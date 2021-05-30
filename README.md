# Multimodal Knowledge Graph for Link Prediction and Node Classification in ADNI and PPMI datasets

## About
View Final_Report to get idea of this project. 

Code adapted from: https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling
Paper reference: End-to-End Entity Classification on Multimodal Knowledge Graphs: https://arxiv.org/abs/2003.12383

## Codes and their explanations
1. <code> adni_dataprep.py </code> Preparation of ADNI dataset => we also prepared the dataset further if you only want to use adni dataset
2. <code> pd_dataprep.py </code> Preparation of PPMI dataset => we also prepared the dataset further if you only want to use ppmi dataset
3. <code>combined_dataprep.py</code>: Preparation of ADNI and PPMI dataset
4. <code>config.yaml</code>: config file for adni_pd_link_NC.ipynb
5. <code>adni_pd_link_NC.ipynb</code>: Jupyter notebook to run dataset through link prediction and node classification
6. utils folder, consisting of
- <code>dataprep_utils.py</code>: functions needed for preparing dataset for adni_dataprep.py and pd_dataprep.py
- <code>load.py</code>: loading and preparing dataset for tasks of node classification and link prediction
- <code>data_utils.py</code>: classes to prepare images for 3D SqueezeNet 
7. <code>model_utils.py</code>: functions used in model development
8. <code>SFCNnet.py</code>: classes defining the SCFN network
9. <code>squeezenet.py</code>: classes defining the SqueezeNet network
10. <code>rgcn.py</code>: classes defining rgcn network.
11. <code>Combined.ipynb</code>: Jupyter notebook to prepare ADNI and PPMI datasets for ablation studies

## Other References
I used several pretrained models in this project
1. <code>SFCN</code>: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain
2. <code> 3D-Squeezenet </code>: https://github.com/okankop/Efficient-3DCNNs
3. <code> Bio-Clinical BERT </code>: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

## Data Source
I got the data from ADNI (https://adni.loni.usc.edu/) and PPMI (https://www.ppmi-info.org/). If you like, please request the data from there directly. 

## How to run the codes
1. Run adni_dataprep.py
2. Run pd_dataprep.py 
3. Run combined_dataprep.py (need both step 1 and 2 to be run first)
4. Change config.yaml to reflect updated parameters like model path, embedding paths etc
5. Run adni_pd_link_NC.ipynb