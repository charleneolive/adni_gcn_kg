import torch
from torch import nn
import torch.nn.functional as F
from model_utils import sum_sparse, adj, pca, enrich

class RGCN(nn.Module):
    """
    We use a classic RGCN, with embeddings as inputs (instead of the one-hot inputs of rgcn.py)
    """

    def __init__(self, triples, n, r, insize, hidden, numcls, device, link_prediction, bases=None):
        '''
        params:
        triples: triples
        n: number of entities
        r: number of relations
        insize: input size dimensions (feature  vector dim)
        hidden: embeddings of 2nd RGCN
        numcls: number of classes for entity classification
        bases: number of bases in basis decomposition
        '''

        super().__init__()
        
        self.reg_param = 0

        self.insize = insize
        self.hidden = hidden
        self.bases = bases
        self.numcls = numcls
        self.link_prediction = link_prediction

        self.triples = enrich(triples, n, r, device) # return triples, inverses and self loops

        # horizontally and vertically stacked versions of the adjacency graph
        
        # relation & inverse relation + self relation
        hor_ind, hor_size = adj(self.triples, n, 2*r+1, device, vertical=False) # horizontal mapping of relation
        ver_ind, ver_size = adj(self.triples, n, 2*r+1, device, vertical=True) # vertical mapping of relation

        _, rn = hor_size # rn: r x n
        r = rn // n # number of relations

        vals = torch.ones(ver_ind.size(0), dtype=torch.float, device = device) # ver_ind
        vals = vals / sum_sparse(ver_ind, vals, ver_size, device)

        hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        self.register_buffer('hor_graph', hor_graph)

        ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
        self.register_buffer('ver_graph', ver_graph)

        # layer 1 weights
        if bases is None: # weights = number of relations x input size x hidden
            self.weights1 = nn.Parameter(torch.FloatTensor(r, insize, hidden))
            nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else: 
            # self.comps1: matrix to map relations to number of bases 
            self.comps1 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps1, gain=nn.init.calculate_gain('relu'))
            #self.bases1: replacement for weight matrix
            self.bases1 = nn.Parameter(torch.FloatTensor(bases, insize, hidden))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 weights 
        if bases is None:
            # number of relations x hidden size x number of classes
            self.weights2 = nn.Parameter(torch.FloatTensor(r, hidden, numcls) )
            nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None
        else:
            self.comps2 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(bases, hidden, numcls))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(hidden).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(numcls).zero_())
        
        self.w_relation = nn.Parameter(torch.Tensor(r, numcls)) #number of relations x number of hidden features
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, features):

        # size of node representation per layer: f -> e -> c
        n, rn = self.hor_graph.size()
        r = rn // n
        e = self.hidden
        b, c = self.bases, self.numcls

        n, f = features.size() # f is size of feature => e.g. 16

        ## Layer 1
        # r x n * num of predicates 
        h = torch.mm(self.ver_graph, features) # sparse mm (ver_graph => (number of relations x number of nodes) x num of entities ) x (number of entities x embeddiing size)
        h = h.view(r, n, f) # number of relations x number of entities x embedding size

        if self.bases1 is not None: # get weights through basis decompositino
            weights = torch.einsum('rb, bij -> rij', self.comps1, self.bases1)
            # weights = torch.mm(self.comps1, self.bases1.view(b, n*e)).view(r, n, e)
        else:
            weights = self.weights1

        assert weights.size() == (r, f, e) # number of relations, input size  x hidden size 

        # Apply weights and sum over relations
        h = torch.bmm(h, weights).sum(dim=0) # output h: number of nodes x hidden size  => sum over all relations

        assert h.size() == (n, e)

        h = F.relu(h + self.bias1) #activation

        ## Layer 2

        # Multiply adjacencies by hidden
        # https://arxiv.org/pdf/1609.02907.pdf
        h = torch.mm(self.ver_graph, h) # sparse mm => need adjacency matrix again: https://github.com/tkipf/relational-gcn/blob/master/rgcn/train.py
        h = h.view(r, n, e) # new dim for the relations

        if self.bases2 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps2, self.bases2)
            # weights = torch.mm(self.comps2, self.bases2.view(b, e * c)).view(r, e, c)
        else:
            weights = self.weights2

        # Apply weights, sum over relations
        # h = torch.einsum('rhc, rnh -> nc', weights, h)
        h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, c)
        
        if self.link_prediction==True:

            return h 
        else:
            
            return h + self.bias2
    
    def calc_score(self, embedding, triplets):
        '''
        DistMult
        embedding: number of entities x feature vector
        
        '''
        s = embedding[triplets[:,0]] # query for subject
        r = self.w_relation[triplets[:,1]] # query for relation
        o = embedding[triplets[:,2]] # query for object
        score = torch.sum(s * r * o, dim=1) # same dimensions for all the embeddings for subject, relaiton & object
        # calculate score. => we will learn the relation matrix. Also embeddings because they are weights
        return score
    
    def regularization_loss(self, embedding):
        # l2 norm of the weights
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))
    
    def get_loss(self, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)

        # labels is whether it is a corrupted or non-corrupted triple
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

    def penalty(self, p=2):

        assert p==2

        if self.bases is None:
            return self.weights1.pow(2).sum()

        return self.comps1.pow(p).sum() + self.bases1.pow(p).sum()

