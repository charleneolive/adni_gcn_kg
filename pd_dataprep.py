'''
Last updated: 9 May 2021
I match the metadata and image through the acquisition and info date. For the current medical conditions, I just take any notes which are written in the same year
'''

import pandas as pd
import numpy as np
import collections
import glob
import os
import re
import datetime
from datetime import datetime
from sklearn import preprocessing
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

from utils.dataprep_utils import check_duplicate, read_triple, patientInfo, findPatientInfo, get_all_categories

BASE_PATH = "/data/data_repo/neuro_img/PPMI"

data_file = pd.read_csv(os.path.join(BASE_PATH,"meta","PPMI_subj_info.csv"))
IMAGE_PATH = os.path.join(BASE_PATH, "mri")
SUBJECT_PATH = glob.glob(os.path.join(IMAGE_PATH,"*"))
DATA_PATH = glob.glob(os.path.join(IMAGE_PATH,"*","*sMRI.nii"))
DATA_PATH2 = "./processed_data/pd_data"
split_size = 0.2

# match acq date and infodt
# get demographic relations
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
counter = 1
for path in SUBJECT_PATH:
    counter+=1
    name_parts = os.path.basename(path).split("_")
    subject_name = name_parts[0]
    associated_mri = glob.glob(os.path.join(path,subject_name+"*sMRI.nii"))
    if subject_name not in subject_mri_dict.keys():
        subject_mri_dict[subject_name] = [associated_mri]
    else:
        subject_mri_dict[subject_name].append(associated_mri)

df = pd.DataFrame(subject_mri_dict.items(), columns=['PATNO','image_name'])  

df.to_csv(os.path.join(DATA_PATH2,"all_data.csv"), index=False)

df_demo = pd.DataFrame(data_file)
df_demo = df_demo.rename(columns={"Subject":"PATNO", "Group": "researchGroup", "Sex": "subjectSex", "Age": "subjectAge"})

files = {"Epworth_Sleepiness_Scale.csv":["ESS1", "ESS2", "ESS3", "ESS4", "ESS5", "ESS6", "ESS7", "ESS8"], 
         "University_of_Pennsylvania_Smell_ID_Test.csv": "NORMATIVE_SCORE",
        "Montreal_Cognitive_Assessment__MoCA_.csv":"MCATOT",
        "State-Trait_Anxiety_Inventory.csv":None,
        "REM_Sleep_Disorder_Questionnaire.csv": ["DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL", "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN", "DRMREMEM", "SLPDSTRB", "neuro_condiiton"],
        "Current_Medical_Conditions_Log.csv":"CONDTERM"}

map_files = {"Epworth_Sleepiness_Scale.csv":"ESS", 
         "University_of_Pennsylvania_Smell_ID_Test.csv": "UPSIT",
        "Montreal_Cognitive_Assessment__MoCA_.csv":"MoCa",
        "State-Trait_Anxiety_Inventory.csv":"STAI",
        "REM_Sleep_Disorder_Questionnaire.csv": "REM",
        "Current_Medical_Conditions_Log.csv":"CONDTERM"}

neuro_scores = ["STROKE","HETRA","PARKISM","RLS","NARCLPSY","DEPRS","EPILEPSY","BRNINFM","CNSOTH","CNSOTH"]

df_dict = {}
for key,value in files.items():
    one_df = pd.read_csv(os.path.join(BASE_PATH, "meta", "PPMI_meta_all", key))
    print(key)
    
    if key == "State-Trait_Anxiety_Inventory.csv":
        value = [s for s in one_df.columns.tolist() if s.startswith("STAI")]
        
    if key == "REM_Sleep_Disorder_Questionnaire.csv":
        one_df["neuro_condition"] = one_df.loc[:,neuro_scores].any(axis=1)
        one_df["neuro_condiiton"] = np.where(one_df["neuro_condition"]==True, 1, 0)
    
    if (key == "University_of_Pennsylvania_Smell_ID_Test.csv") | (key == "Montreal_Cognitive_Assessment__MoCA_.csv") | (key == "Current_Medical_Conditions_Log.csv"):
        one_df['object'] = one_df.loc[:,value]
    
    else:
        one_df['object'] = one_df.loc[:,value].sum(axis=1).astype(int)
    df_dict[map_files[key]] = one_df
    

def getEntity(df_dict, element, patient_name, info_date):
    df = df_dict[element][(df_dict[element]["PATNO"]==int(patient_name)) & (df_dict[element]["INFODT"]==str(info_date))]
    
    if len(df)>0:
        if df["object"].dtype == "float64":
            return int(df["object"])
        elif df["object"].dtype == "int64":
            return int(df["object"])
        elif df["object"].dtype == "object":
            return df["object"].tolist()[0]
        else:
            print(df["object"])
    else:
        return None
    
    
interested_elements = ['researchGroup','subjectSex','subjectAge','ESS','UPSIT','MoCa','STAI','REM','CONDTERM','Image']
relation_elements = {'researchGroup': 'hasResearchGroup',
                     'subjectSex':'hasSubjectSex',
                     'subjectAge':'hasSubjectAge',
                     'ESS': 'hasESS',
                     'UPSIT': 'hasUPSIT',
                     'MoCa': 'hasMoCa',
                     'STAI': 'hasSTAI',
                     'REM': 'hasREM',
                     'CONDTERM':'hasCONDTERM',
                     'Image': 'hasImage'
                    }
categorical_variables = ['researchGroup', 'subjectSex', 'subjectAge', 'ESS', 'UPSIT', 'MoCA', 'STAI', 'REM', 'CONDTERM']

# build dictionary for patient info as well, load image to create relation
patient_dict = {key: [] for key in interested_elements}
relation_dict = {key: [] for key in relation_elements}
name_dict = {key: [] for key in interested_elements}
all_keys = {key: [] for key in categorical_variables}
image_relation_list=[]

file1 = open("{}/element_not_available.txt".format(DATA_PATH2),"a")
'''
take the timepoint where there is the image
'''

for patient in df_demo["PATNO"].unique():
    patient_name = patient
    
    new_patient_info = {}
    if str(patient_name) in df["PATNO"].tolist(): # patient needs to have an image
        one_image = df[df["PATNO"]==str(patient_name)]["image_name"].tolist()[0][0][0]
        match = re.search(r'\d{4}-\d{2}-\d{2}', one_image)
        date = datetime.strptime(match.group(), '%Y-%m-%d').date()
        
        new_patient_info['Image'] = one_image
        # convert to new date format
        new_date = str(date.month) +'/'+ str(date.day).zfill(2)+'/'+ str(date.year)
        
        new_patient_info['subjectAge'] = int(df_demo[(df_demo['PATNO']==patient_name) & (df_demo['Acq Date']==new_date) & (df_demo['Modality']=='MRI')]['subjectAge'])
        new_patient_info['subjectSex'] = df_demo[(df_demo['PATNO']==patient_name) & (df_demo['Acq Date']==new_date) & (df_demo['Modality']=='MRI')]['subjectSex'].tolist()[0]
        new_patient_info['researchGroup'] = df_demo[(df_demo['PATNO']==patient_name) & (df_demo['Acq Date']==new_date) & (df_demo['Modality']=='MRI')]['researchGroup'].tolist()[0]
        
        # connect to ESS:
        info_date = str(date.month).zfill(2) +'/'+ str(date.year)
        if getEntity(df_dict, "ESS", patient_name, info_date) is not None: new_patient_info["ESS"] = getEntity(df_dict, "ESS", patient_name, info_date)
        if getEntity(df_dict, "UPSIT", patient_name, info_date) is not None: new_patient_info["UPSIT"] = getEntity(df_dict, "UPSIT", patient_name, info_date)
        if getEntity(df_dict, "MoCa", patient_name, info_date) is not None: new_patient_info["MoCa"] = getEntity(df_dict, "MoCa", patient_name, info_date)
        if getEntity(df_dict, "STAI", patient_name, info_date) is not None: new_patient_info["STAI"] = getEntity(df_dict, "STAI", patient_name, info_date)
        if getEntity(df_dict, "REM", patient_name, info_date) is not None: new_patient_info["REM"] = getEntity(df_dict, "REM", patient_name, info_date)
        
        temp = df_dict['CONDTERM'][df_dict['CONDTERM']['PATNO']==patient_name]
        if len(temp[temp['ORIG_ENTRY'].str.contains(str(date.year))]["CONDTERM"].tolist())>0:
            records = temp[temp['ORIG_ENTRY'].str.contains(str(date.year))]["CONDTERM"].tolist()
            new_patient_info["CONDTERM"] = ','.join(records)
        all_keys = get_all_categories(all_keys, new_patient_info, interested_elements, categorical_variables)
    
    for idx,element in enumerate(interested_elements):
        if element in list(new_patient_info.keys()):
            patient_dict[element].append(new_patient_info[element])
            relation_dict[element].append(relation_elements[element])
            name_dict[element].append(patient_name)
        else:
            file1.write('Element {} not in keys. Patient is {}\n'.format(element, patient_name))
            
            
'''
CREATE DATAFRAMES FOR ALL THE ENTITIES
'''
dataframes_dict = {}
for s,p,o in zip(name_dict.keys(), relation_dict.keys(), patient_dict.keys()):
    d = {'subject_name': name_dict[s], 'hasRelation': relation_dict[p], 'tail': patient_dict[o]}        
    df_temp = pd.DataFrame(data=d)
    dataframes_dict[o] = df_temp

    
intermediate = pd.concat(dataframes_dict.values())
intermediate.to_csv(os.path.join(DATA_PATH2, "intermediate.csv"), index=False)
intermediate.to_pickle(os.path.join(DATA_PATH2, "intermediate.pkl"))

'''
if using only ppmi data, then group PD & SWEDD as diseased, Control as Control
'''

df_research = dataframes_dict.pop('researchGroup')
df_research.loc[(df_research['tail'] == "PD") | (df_research['tail'] == "SWEDD"), 'new_tail'] = 'Diseased' 
df_research.loc[df_research['tail'] == "Control" , 'new_tail'] = 'Control' 

all_dfs = pd.concat(dataframes_dict.values())
relations_df = pd.DataFrame(all_dfs.hasRelation.unique(), columns=["label"])
relations_df.index.name = "index"
relations_df.to_csv(os.path.join(DATA_PATH2,"relations.int.csv"))
all_dfs2 = all_dfs.explode('tail')

all_dfs2.to_csv(os.path.join(DATA_PATH2, "triples.csv"), index=False)

'''
Create training, validation and test set
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
train_data.to_csv(os.path.join(DATA_PATH2,"train.csv"), index=False)
val_data.to_csv(os.path.join(DATA_PATH2,"val.csv"), index=False)
test_data.to_csv(os.path.join(DATA_PATH2,"test.csv"), index=False)

triples = read_triple(all_dfs2, entity2id, relation2id)
np.savetxt(os.path.join(DATA_PATH2,"triples.txt"), triples, fmt='%s')
