import os
import torch
import random
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.decomposition import PCA

def get_ss_attribute(molecule_list, raw_feature_root, ss_types='GCIP_G', dim =100):
    with open(f'{raw_feature_root}/{ss_types}.pkl', 'rb') as file:
        data = pickle.load(file)

    index_list = data['index']
    smiles_list = data['smiles']
    features_list = data['x']

    features_dict = {smiles_list[i]: features_list[i] for i in range(len(smiles_list))}

    selected_features = [features_dict[smiles] for smiles in features_dict]

    pca = PCA(n_components=dim)
    fp_PCA = pca.fit_transform(selected_features)

    ss_attributes_list = fp_PCA

    return ss_attributes_list

for dataset in ['tox21', 'sider', 'muv']:
     for feat in ['CGIP_G', 'GraphMVP', 'IEM_3d_10conf', 'MoleBERT', 'molformer', 'unimol_10conf', 'VideoMol_3d_1conf']:
         for dim in [100,200,300]:
            if dataset == 'tox21':
                raw_filename = "data/tox21.csv"
                tasks = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
                             "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
                             "SR-HSE", "SR-MMP", "SR-p53", ]
            elif dataset == 'sider':
                raw_filename = "data/sider.csv"
                tasks = ["SIDER1", "SIDER2", "SIDER3", "SIDER4", "SIDER5", "SIDER6", "SIDER7", "SIDER8", "SIDER9", "SIDER10",
                        "SIDER11", "SIDER12", "SIDER13", "SIDER14", "SIDER15", "SIDER16", "SIDER17", "SIDER18", "SIDER19", "SIDER20",
                        "SIDER21", "SIDER22", "SIDER23", "SIDER24", "SIDER25", "SIDER26", "SIDER27"]

            elif dataset == 'muv':
                raw_filename = "data/muv.csv"
                tasks = ["MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689", "MUV-692", "MUV-712", "MUV-713",
                         "MUV-733",
                         "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"]

            elif dataset == 'tdc':
                raw_filename = "data/tdc.csv"
                tasks = ["bbb_martins", "cyp2c9_veith", "pgp_broccatelli", "hia_hou", "cyp2d6_veith", "cyp3a4_veith",
                         "bioavailability_ma", "herg", "ames", "dili"]


            smiles_tasks_df = pd.read_csv(raw_filename)
            smilesList = smiles_tasks_df.smiles.values



            remained_smiles = []
            for smiles in smilesList:
                try:
                    remained_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
                except:
                    pass

            smiles_tasks_df['remained_smiles'] = remained_smiles
            remained_df = smiles_tasks_df

            raw_feature_root = f'./self-supervised_attributes/{dataset}'
            ss_attributes_list = get_ss_attribute(list(remained_df["remained_smiles"].values), raw_feature_root, ss_types=feat, dim = dim)
            cache_file_path = f'./cache/self-supervised_feat/{dim}dim/{dataset}/{feat}.npy'
            print("cache fp data to {}".format(cache_file_path))
            np.save(cache_file_path, ss_attributes_list)