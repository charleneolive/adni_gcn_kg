import pandas as pd
import numpy as np
import collections
# import lmdb
# import msgpack, msgpack_numpy
import torch
import glob
import os
from sklearn import preprocessing
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

'''
see find patient info for caveat
'''

def get_all_categories(all_keys, patient_info, interested_elements, categorical_variables):
    
    for key, value in patient_info.items():
        if key in categorical_variables and value not in all_keys[key]:
            all_keys[key].append(value)
        
    return all_keys

def one_hot_encode(all_keys):
    '''
    One hot encode categorical variables if needed
    '''
    new_all_keys = {}
    for key in all_keys.keys():
        all_keys[key].sort()
        new_all_keys[key] = all_keys[key]
        ohe = preprocessing.OneHotEncoder()
        categories = np.array(all_keys[key]).reshape(-1,1)
        ohe.fit(categories)
        matrix = ohe.transform(categories).todense()
        all_keys[key] = torch.tensor(matrix)
    return all_keys

def check_duplicate(PATH):
    
    '''
    function to check that there are no duplicate images
    '''
    a_list = [os.path.basename(file) for file in PATH]
    a_set = set(a_list)
    contains_duplicates = len(a_list) != len(a_set)
    return contains_duplicates

def read_triple(df, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    for idx,row in df.iterrows():
        s, p, o = row['subject_name'], row['hasRelation'], row['tail']
        triples.append((entity2id[s], relation2id[p], entity2id[o]))
    return triples

def patientInfo(xml_path):
    '''
    extract patient info from xml path
    '''
    one_xml = ET.parse(xml_path)
    root = one_xml.getroot()


    # find repeat elements
    all_elements = []
    repeat_elements = []
    for element in root.iter():
        if element.tag in all_elements:
            if element.tag not in repeat_elements:
                repeat_elements.append(element.tag)
        else:
            all_elements.append(element.tag)

    # get info. For repeat elements, use the item. 

    omitted_elements = []
    patient_info = {}
    for element in root.iter():
        if not element.text.isspace() and not element.tag in repeat_elements:
            patient_info[element.tag] = element.text
        elif not element.text.isspace() and element.tag in repeat_elements:
            try:
                (key, value) = element.items()[0]
                patient_info[value] = element.text
            except:
                if element.tag not in omitted_elements:
                    omitted_elements.append(element.tag)
                    
    return patient_info


def findPatientInfo(row, data_list):
    '''
    get corresponding patient info => use only the first image if the patient has multiple images
    '''
    
    img_name = row['image_name']
    sub_name = row['subject_name']
    name_parts = os.path.splitext(os.path.basename(img_name[0]))[0].split("_")

    construct_substring = name_parts[-2]+"_"+name_parts[-1]
    xml_file = [file for file in data_list if construct_substring in file]

    patient_info = patientInfo(xml_file[0])


#     for key, value in patient_info.items():
#         if (key in interested_elements) and (key not in categorical_variables):
#             new_patient_info[key] = value
#         elif (key in interested_elements) and (key in categorical_variables):
#             new_patient_info[key] = all_keys[key][new_all_keys[key].index(value)]
    return patient_info