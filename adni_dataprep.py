import pandas as pd
import numpy as np
import collections
import glob
import os
from sklearn import preprocessing
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from utils.dataprep_utils import check_duplicate, read_triple, patientInfo, findPatientInfo, get_all_categories

BASE_PATH = "/data/data_repo/neuro_img/ADNI/mri/" # image & data path
data_list = glob.glob(os.path.join(BASE_PATH, "metadata","*.xml"))
IMAGE_PATH = os.path.join(BASE_PATH, "MRI_N3_from_DTI_ECC_merged")
SUBJECT_PATH = glob.glob(os.path.join(IMAGE_PATH,"*"))
DATA_PATH = glob.glob(os.path.join(IMAGE_PATH,"*","*","*","*","*.nii"))
DATA_PATH2 = "./processed_data/adni_data" # where to save data
split_size = 0.2

IMAGES = [os.path.basename(path) for path in DATA_PATH]
# no duplicates in images
assert check_duplicate(IMAGES) == False

'''
group images according to subject
'''
subject_mri_dict = {}
'''
Get datafrrame of subject name and corresponding mri scans
'''

for path in SUBJECT_PATH:
    subject_name = os.path.basename(path)
    associated_mri = glob.glob(os.path.join(path,"*","*","*","*.nii"))
    subject_mri_dict[subject_name] = associated_mri

df = pd.DataFrame(subject_mri_dict.items(), columns=['subject_name','image_name'])  
df.to_csv(os.path.join(DATA_PATH2,"all_data.csv"), index=False)

interested_elements = ['researchGroup','subjectSex','subjectAge','weightKg', 'APOE A1', 'APOE A2', 'MMSCORE', 'GDTOTAL', 'CDGLOBAL']
relation_elements = {'researchGroup': 'hasResearchGroup',
                     'subjectSex':'hasSubjectSex',
                     'subjectAge':'hasSubjectAge',
                     'weightKg':'hasWeightKg',
                    'APOE A1':'hasAPOEA1',
                    'APOE A2': 'hasAPOEA2',
                     'MMSCORE': 'hasMMSCORE',
                     'GDTOTAL':'hasGDTOTAL',
                     'CDGLOBAL':'hasCDGLOBAL'
                    }
categorical_variables = ['researchGroup', 'subjectSex', 'APOE A1', 'APOE A2']

# build dictionary for patient info as well, load image to create relation
patient_dict = {key: [] for key in interested_elements}
relation_dict = {key: [] for key in relation_elements}
name_dict = {key: [] for key in interested_elements}
all_keys = {key: [] for key in categorical_variables}
image_relation_list=[]

file1 = open("{}/element_not_available.txt".format(DATA_PATH2),"a")

for idx,row in df.iterrows():
    new_patient_info=findPatientInfo(row, data_list)
    all_keys = get_all_categories(all_keys, new_patient_info, interested_elements, categorical_variables)
    image_relation_list.append('hasImage')
    
    for idx,element in enumerate(interested_elements):
        if element in list(new_patient_info.keys()):
            patient_dict[element].append(new_patient_info[element])
            relation_dict[element].append(relation_elements[element])
            name_dict[element].append(row['subject_name'])
        else:
            file1.write('Element {} not in keys. Patient is {}\n'.format(element, row['subject_name']))
            
'''
CREATE DATAFRAMES FOR ALL THE ENTITIES
'''
dataframes_dict = {}
for s,p,o in zip(name_dict.keys(), relation_dict.keys(), patient_dict.keys()):
    d = {'subject_name': name_dict[s], 'hasRelation': relation_dict[p], 'tail': patient_dict[o]}        
    df_temp = pd.DataFrame(data=d)
    dataframes_dict[o] = df_temp


df['hasRelation'] = image_relation_list
df = df.rename(columns={"image_name":"tail"})
dataframes_dict['Image'] = df

intermediate = pd.concat(dataframes_dict.values())
intermediate2 = pd.concat((intermediate, df))
intermediate2.to_csv(os.path.join(DATA_PATH2, "intermediate.csv"), index=False)
intermediate2.to_pickle(os.path.join(DATA_PATH2, "intermediate.pkl"))

df_research = dataframes_dict.pop('researchGroup')

'''
this is if you just want to use adni data - map cn, smc and emci to normal/early AD, and mci, ad and lmci to AD
'''

df_research.loc[(df_research['tail'] == "CN") | (df_research['tail'] == "SMC") | (df_research['tail'] == "EMCI"), 'new_tail'] = 'Normal/EarlyAD' 
df_research.loc[(df_research['tail'] == "MCI") | (df_research['tail'] == "AD") | (df_research['tail'] == "LMCI"), 'new_tail'] = 'AD' 

all_dfs = pd.concat(dataframes_dict.values())
relations_df = pd.DataFrame(all_dfs.hasRelation.unique(), columns=["label"])
relations_df.index.name = "index"
relations_df.to_csv(os.path.join(DATA_PATH2,"relations.int.csv"))
all_dfs2 = all_dfs.explode('tail')
all_dfs2.to_csv(os.path.join(DATA_PATH2, "triples.csv"), index=False)

'''
Create training and validation set
'''
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

'''
save files
'''
        
nodes_df = pd.DataFrame({"annotation":annotations, "label":label})
nodes_df.index.name = "index"
nodes_df.to_csv(os.path.join(DATA_PATH2,"nodes.int.csv"))

data = pd.DataFrame({'index':index, 'class':class_name})
train_val_data, test_data = train_test_split(data, test_size = split_size)
train_data, val_data= train_test_split(train_val_data, test_size = split_size)
train_data.to_csv(os.path.join(DATA_PATH2, "train.csv"), index=False)
val_data.to_csv(os.path.join(DATA_PATH2, "val.csv"), index=False)
test_data.to_csv(os.path.join(DATA_PATH2, "test.csv"), index=False)

triples = read_triple(all_dfs2, entity2id, relation2id)
np.savetxt(os.path.join(DATA_PATH2,"triples.txt"), triples, fmt='%s')