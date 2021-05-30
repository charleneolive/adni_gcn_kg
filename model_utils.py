import fire, sys, tqdm
import os
import cv2
import gzip
import numpy as np
import pandas as pd
from skimage import io as skio
from collections import Counter
from collections import OrderedDict
from sklearn.decomposition import PCA
from squeezenet import SqueezeNetwork
from SFCNnet import SFCNNetwork
import time
import torch
from torch import nn
from utils.data_utils import PrepareDataset, PrepareDataset2
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F

import transformers as tf
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms

def score_distmult_bc(data, node_embeddings, edge_embeddings):
    '''
    compute dismult score
    '''
    si, pi, oi = data

    s = node_embeddings[si, :]
    p = edge_embeddings[pi, :]
    o = node_embeddings[oi, :]

    if len(s.size()) == len(p.size()) == len(o.size()): # optimizations for common broadcasting
        if pi.size(-1) == 1 and oi.size(-1) == 1:
            singles = p * o # ignoring batch dimensions, this is a single vector
            return torch.matmul(s, singles.transpose(-1, -2)).squeeze(-1)

        if si.size(-1) == 1 and oi.size(-1) == 1:
            singles = s * o
            return torch.matmul(p, singles.transpose(-1, -2)).squeeze(-1)

        if si.size(-1) == 1 and pi.size(-1) == 1:
            singles = s * p
            return torch.matmul(o, singles.transpose(-1, -2)).squeeze(-1)

    return torch.sum(s * p * o, dim = -1)

def compute_ranks_fast(eval_set, node_embeddings, edge_embeddings, batch_size=16):
    '''
    compute rank for link prediction task
    '''
#     idx_begin, idx_end = splits[eval_split][0], splits[eval_split][1]
#     eval_set = data[idx_begin:idx_end]

#     true_heads, true_tails = truedicts(data) if filtered else (None, None)

    num_facts = eval_set.shape[0] # number of triples to evaluate
    num_nodes = node_embeddings.shape[0] # number of nodes in the graph - also the same as  the number of entities
    num_batches = int((num_facts + batch_size-1)//batch_size) # number of batches => add batch size to account for last small batch
    ranks = torch.empty((num_facts*2), dtype=torch.int64)
    
    for head in [False, True]:  # head or tail prediction
        offset = int(head) * num_facts # ??
        # evaluate by batches
        for batch_id in range(num_batches):
            batch_begin = batch_id * batch_size
            batch_end = min(num_facts, (batch_id+1) * batch_size)

            batch_idx = (int(head) * num_batches) + batch_id + 1

            batch_data = eval_set[batch_begin:batch_end]
            batch_num_facts = batch_data.shape[0] # number of triples in batch

            # compute the full score matrix (filter later)
            bases   = batch_data[:, 1:] if head else batch_data[:, :2] # bases of corrupted heads or tails
            targets = batch_data[:, 0]  if head else batch_data[:, 2] # whether it is tail corrupted or head corrupted

            # collect the triples for which to compute scores
            
            
#             # bases represent the parts of the triple which we will not modify
            bexp = bases.view(batch_num_facts, 1, 2).expand(batch_num_facts, #tensor expanded to higher number of dimensions
                                                            num_nodes, 2) # so that it can then be tested with all combinations of corrupted node.
            # ar evenly spaced nodes => test all the other possible entities => I think num_nodes is just to select all the nodes in the graph?? => actually just need batch facts??
            ar   = torch.arange(num_nodes).view(1, num_nodes, 1).expand(batch_num_facts, num_nodes, 1) #same goes for nodes => expand to make it of size number of triples x number of nodes
                                                                        
            candidates = torch.cat([ar, bexp] if head else [bexp, ar], dim=2)  # size of batch_num_facts x num_nodes x 3
            
            scores = score_distmult_bc((candidates[:,:,0],
                                        candidates[:,:,1],
                                        candidates[:,:,2]),
                                       node_embeddings, #embedding is 
                                       edge_embeddings).to('cpu')

            # Select the true scores => targets are the true values 
            true_scores = scores[torch.arange(batch_num_facts), targets] # scores of uncorrupted
            # get number of scores which are greater than the true score, i.e. if the true score is high, then we get a low rank=> high MRR
            batch_ranks = torch.sum(scores > true_scores.view(batch_num_facts, 1), dim=1, dtype=torch.int64)
            # -- This is the "optimistic" rank (assuming it's sorted to the front of the ties) - get number of scores which tie with true score
            num_ties = torch.sum(scores == true_scores.view(batch_num_facts, 1), dim=1, dtype=torch.int64)

            # Account for ties (put the true example halfway down the ties)
            batch_ranks = batch_ranks + (num_ties - 1) // 2

            ranks[offset+batch_begin:offset+batch_end] = batch_ranks

    return ranks + 1

def sum_sparse(indices, values, size, device, row=True):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries
    params:
    indices: pred x sub
    vals: pred
    ver_size: size of adjacency matrix (vertical)
    
    remember only 1 relation between each subject and predicate hence each row will have only 1 value
    
    num predicates = num subjects

    :return:
    """

    ST = torch.cuda.sparse.FloatTensor if indices.is_cuda else torch.sparse.FloatTensor #memory efficient

    assert len(indices.size()) == 2

    k, r = indices.size() # k: pred, r: sub

    if not row:
        # transpose the matrix
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=device) # size(1) number of predicates 

    smatrix = ST(indices.t(), values, size=size) #ST: sparse tensor => sub * pred
    sums = torch.mm(smatrix, ones) # row/column sums => get subjects x 1 => values are the sums of predicates 

    sums = sums[indices[:, 0]] # index with indices of predicates

    assert sums.size() == (k, 1) # should be number of predicates 

    return sums.view(k)

def adj(triples, num_nodes, num_rels, device, vertical=True):
    """
     Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
     relations are stacked vertically).

     :param triples: all the triples
     :num_nodes: number of nodes
     :num_rels: number of relations
     :return: sparse tensor
    """
    r, n = num_rels, num_nodes
    size = (r * n, n) if vertical else (n, r * n) # if vertical, height is num_reals*nodes

    from_indices = []
    upto_indices = []

    for fr, rel, to in triples:

        offset = rel.item() * n # add offset to relation => by n because this is the number of nodes

        if vertical: # fr x to matrix
            fr = offset + fr.item() # the relation is the position on the graph, i.e. the "offset"
        else:
            to = offset + to.item()

        from_indices.append(fr)
        upto_indices.append(to)

    indices = torch.tensor([from_indices, upto_indices], dtype=torch.long, device=device)
    assert indices.size(1) == len(triples) # i.e. 
    assert indices[0, :].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[1, :].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices.t(), size #transpose indices: pred x sub

def pca(tensor, target_dim, device):
    """
    Applies PCA to a torch matrix to reduce it to the target dimension
    """

    n, f = tensor.size()
    if n < 25: # no point in PCA, just clip
        res = tensor[:, :target_dim]
    else:
        if device == "cuda":
            tensor = tensor.to('cpu')
        model = PCA(n_components=target_dim, whiten=True)

        res = model.fit_transform(tensor)
        res =  torch.from_numpy(res)
        res = res.to(device)

    return res

# def enrich(triples : torch.Tensor, n : int, r: int, device):
#     '''
#     n: number of entities
#     r: number of relations
#     '''

#     # inverses will multiply relation by 2 (now we have twice the number of relations)
#     inverses = torch.cat([
#         triples[:, 2:],
#         triples[:, 1:2] + r, # number of relation
#         triples[:, :1]
#     ], dim=1)

#     # self loops will add 1 more relation => hence fill value is 2*r
#     selfloops = torch.cat([
#         torch.arange(n, dtype=torch.long,  device=device)[:, None],
#         torch.full((n, 1), dtype=torch.long, fill_value=2*r), #tensor of size (n,1) and filled by filled value
#         torch.arange(n, dtype=torch.long, device=device)[:, None],
#     ], dim=1)

#     return torch.cat([triples, inverses, selfloops], dim=0)


def enrich(triples : torch.Tensor, n : int, r: int, device):
    '''
    n: number of entities
    r: number of relations
    '''

    # inverses will multiply relation by 2 (now we have twice the number of relations)
    inverses = torch.cat([
        triples[:, 2:],
        triples[:, 1:2] + r, # number of relation
        triples[:, :1]
    ], dim=1)

    # self loops will add 1 more relation => hence fill value is 2*r
    selfloops = torch.cat([
        torch.arange(n, dtype=torch.long,  device=device)[:, None],
        torch.full((n, 1), dtype=torch.long, fill_value=2*r, device=device), #tensor of size (n,1) and filled by filled value
        torch.arange(n, dtype=torch.long, device=device)[:, None]], dim=1)

    return torch.cat([triples, inverses, selfloops], dim=0)

def bert_emb(strings, device, bs_chars):
    # Sort by length and reverse the sort after computing embeddings
    # (this will speed up computation of the embeddings, by reducing the amount of padding required)

    indexed = list(enumerate(strings))
    indexed.sort(key=lambda p:len(p[1]))

    embeddings = bert_emb_([s for _, s in indexed], bs_chars, device)
    indices = torch.tensor([i for i, _ in indexed])
    _, iindices = indices.sort()

    return embeddings[iindices]


def bert_emb_(strings, bs_chars, device):
    
    MNAME="emilyalsentzer/Bio_ClinicalBERT"

    bmodel = AutoModel.from_pretrained(MNAME)
    btok = AutoTokenizer.from_pretrained(MNAME)

    pbar = tqdm.tqdm(total=len(strings))

    outs = []
    fr = 0
    while fr < len(strings):

        to = fr
        bs = 0
        while bs < bs_chars and to < len(strings):
            bs += len(strings[to])
            to += 1
            # -- add strings to the batch until it puts us over bs_chars

        strbatch = strings[fr:to]
    
        try:
            batch = btok(strbatch, padding=True, truncation=True, return_tensors="pt")
        except:
            print(strbatch)
            sys.exit()
        #-- tokenizer automatically prepends the CLS token
        inputs, mask = batch['input_ids'], batch['attention_mask']
        inputs, mask = inputs.to(device), mask.to(device)

        bmodel.to(device)
        out = bmodel(inputs, mask)

        outs.append(out[0][:, 0, :].to('cpu')) # use only the CLS token

        pbar.update(len(strbatch))
        fr = to

    return torch.cat(outs, dim=0)


def squeeze_emb(images, sample_size, sample_duration, device, bs=1):
    image_embeddings = []
    # def squeeze_emb(images, bs=4, device):
    prep = transforms.Compose([
        transforms.ToTensor(),
    ])
    images = np.array(images)
    nimages = len(images)

    dataset = PrepareDataset(images, transform=prep)
    imagegen =  DataLoader(dataset, batch_size=bs, shuffle=False)

    squeezenet = SqueezeNetwork(sample_size=sample_size, sample_duration=sample_duration)
    squeezenet.to(device)
    for batch in tqdm.tqdm(imagegen, total=nimages // bs):
        bn, c, d, h, w = batch.size()
        batch = batch.float().to(device)

        with torch.no_grad():
            out = squeezenet(batch)
        image_embeddings.append(out.view(bn, -1).to('cpu'))
            # print(image_embeddings[-1].size())
            
    return torch.cat(image_embeddings, dim=0)

def sfcn_emb(images, device, bs=1):
    image_embeddings = []
    # def squeeze_emb(images, bs=4, device):
    prep = transforms.Compose([
        transforms.ToTensor(),
    ])
    images = np.array(images)
    nimages = len(images)

    dataset = PrepareDataset2(images, transform=prep)
    # want to follow the order => shuffle = False
    imagegen =  DataLoader(dataset, batch_size=bs, shuffle=False)

    sfcn_net = SFCNNetwork()
    sfcn_net.to(device)
    for batch in tqdm.tqdm(imagegen, total=nimages // bs):
        bn, c, d, h, w = batch.size()
        batch = batch.float().to(device)
        print(batch.size())
        with torch.no_grad():
            out = sfcn_net(batch)
        image_embeddings.append(out.view(bn, -1).to('cpu'))
            # print(image_embeddings[-1].size())
            
    return torch.cat(image_embeddings, dim=0)