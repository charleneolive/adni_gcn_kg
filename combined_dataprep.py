import pandas as pd
import numpy as np
import collections
# import lmdb
# import msgpack, msgpack_numpy
import glob
import os
import re
import datetime
from datetime import datetime
from sklearn import preprocessing
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

from utils.dataprep_utils import check_duplicate, read_triple, patientInfo, findPatientInfo, get_all_categories

DATA_PATH2_ppmi = "./processed_data/pd_data"

DATA_PATH2_adni = "./processed_data/adni_data"
DATA_PATH2 = "./processed_data/combined"
split_size = 0.2

df_ppmi = pd.read_pickle(os.path.join(DATA_PATH2_ppmi,"intermediate.pkl"))
df_adni = pd.read_pickle(os.path.join(DATA_PATH2_adni,"intermediate.pkl"))

sample_image = df_ppmi[df_ppmi["hasRelation"]=="hasImage"]['tail'].iloc[0]
sample_image2 = df_adni[df_adni["hasRelation"]=="hasImage"]['tail'].iloc[0]

df_research= pd.concat([df_ppmi[df_ppmi["hasRelation"]=="hasResearchGroup"],df_adni[df_adni["hasRelation"]=="hasResearchGroup"]])
df_research = df_research.replace("Control", "CN")

df_research.loc[(df_research['tail'] == "PD") | (df_research['tail'] == "SWEDD") | (df_research['tail'] == "AD") | (df_research['tail'] == "LMCI") | (df_research['tail'] == "MCI"), 'new_tail'] = 'Diseased' 
df_research.loc[(df_research['tail'] == "CN") | (df_research['tail'] == "EMCI") | (df_research['tail'] == "SMC"), 'new_tail'] = 'Non-Diseased' 

df_ppmi = df_ppmi[df_ppmi.hasRelation != "hasResearchGroup"]
df_adni = df_adni[df_adni.hasRelation != "hasResearchGroup"]
all_dfs = pd.concat([df_ppmi, df_adni])

relations_df = pd.DataFrame(all_dfs.hasRelation.unique(), columns=["label"])
relations_df.index.name = "index"
relations_df.to_csv(os.path.join(DATA_PATH2,"relations.int.csv"))

all_dfs2 = all_dfs.explode('tail')
all_dfs2.to_csv(os.path.join(DATA_PATH2, "triples.csv"), index=False)

relations_df = pd.DataFrame(all_dfs.hasRelation.unique(), columns=["label"])
relations_df.index.name = "index"
relations_df.to_csv(os.path.join(DATA_PATH2,"relations.int.csv"))

all_dfs2 = all_dfs.explode('tail')
all_dfs2.to_csv(os.path.join(DATA_PATH2, "triples.csv"), index=False)

'''
Create training, validation and test set
'''
all_keys = {}
all_keys['researchGroup'] = df_research['new_tail'].unique().tolist()
entity2id = {}
relation2id = {}
id2relation = {}
id2entity = {}

annotations = []
label = []
index = []
class_name = []

relation_counter = 0
entity_counter = 0
for idx,row in all_dfs2.iterrows():
    triplet = [row['subject_name'],row['hasRelation'],row['tail']]
    e1, r, e2 = triplet
    # if the subject number is not in dictionary entity2id, then add to dictionary and assign an ID
    if e1 not in entity2id:
        entity2id[e1] = entity_counter
        id2entity[entity_counter] = e1
        entity_counter += 1
        annotations.append("hasName")
        label.append(e1)
        index.append(entity_counter)
        class_name.append(all_keys['researchGroup'].index(df_research[df_research["subject_name"]==e1]['new_tail'].tolist()[0]))
    # if the relation is not in dictionary entity2id, then add to dictionary and assign an ID
    if e2 not in entity2id:
        entity2id[e2] = entity_counter
        id2entity[entity_counter] = e2
        entity_counter += 1
        annotations.append(r)
        label.append(e2)
    # if the relation number is not in dictionary relation2id, then add to dictionary and assign an ID
    if r not in relation2id:
        relation2id[r] = relation_counter
        id2relation[relation_counter] = r
        relation_counter += 1

nodes_df = pd.DataFrame({"annotation":annotations, "label":label})
nodes_df.index.name = "index"
nodes_df.to_csv(os.path.join(DATA_PATH2,"nodes.int.csv"))

data = pd.DataFrame({'index':index, 'class':class_name})
train_val_data, test_data = train_test_split(data, test_size = split_size, random_state=0)
train_data, val_data= train_test_split(train_val_data, test_size = split_size, random_state=0)
train_data.to_csv(os.path.join(DATA_PATH2, "train.csv"), index=False)
val_data.to_csv(os.path.join(DATA_PATH2, "val.csv"), index=False)
test_data.to_csv(os.path.join(DATA_PATH2, "test.csv"), index=False)

triples = read_triple(all_dfs2, entity2id, relation2id)
np.savetxt(os.path.join(DATA_PATH2,"triples.txt"), triples, fmt='%s')