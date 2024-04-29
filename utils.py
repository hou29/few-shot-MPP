import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data, InMemoryDataset, Dataset
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
from rdkit.ML.Descriptors import MoleculeDescriptors


class FeaturesGeneration:
    # RDKit descriptors -->
    def __init__(self, nbits: int = 1024, long_bits: int = 16384):
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

        # dictionary
        self.fp_func_dict = {'ecfp0': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits),
                             'ecfp2': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits),
                             'ecfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits),
                             'ecfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits),
                             'fcfp2': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True,
                                                                                      nBits=nbits),
                             'fcfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True,
                                                                                      nBits=nbits),
                             'fcfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True,
                                                                                      nBits=nbits),
                             'lecfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=long_bits),
                             'lecfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=long_bits),
                             'lfcfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True,
                                                                                       nBits=long_bits),
                             'lfcfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True,
                                                                                       nBits=long_bits),
                             'maccs': lambda m: MACCSkeys.GenMACCSKeys(m),
                             'hashap': lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits),
                             'hashtt': lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m,
                                                                                                                  nBits=nbits),
                             'avalon': lambda m: fpAvalon.GetAvalonFP(m, nbits),
                             'laval': lambda m: fpAvalon.GetAvalonFP(m, long_bits),
                             'rdk5': lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2),
                             'rdk6': lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2),
                             'rdk7': lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2),
                             'rdkDes': lambda m: calc.CalcDescriptors(m)}

    def get_fingerprints(self, smiles_list: list, fp_name: str) -> np.array:
        """获得分子指纹fingerprint

        Args:
            smiles_list: list(smiles序列)
            fp_name: fingerprint分子指纹名称

        Returns:

        """
        fingerprints = []
        not_found = []

        for smi in tqdm(smiles_list, desc=f"generating fp: {fp_name}"):
            try:
                m = Chem.MolFromSmiles(smi)
                fp = self.fp_func_dict[fp_name](m)
                fingerprints.append(np.array(fp))
            except ValueError:
                not_found.append(smi)
                if fp_name == 'rdkDes':
                    tpatf_arr = np.empty(len(fingerprints[0]), dtype=np.float32)
                else:
                    tpatf_arr = np.empty(len(fingerprints[0]), dtype=np.float32)

                fingerprints.append(tpatf_arr)

        if fp_name == 'rdkDes':
            x = np.array(fingerprints)
            x[x == np.inf] = np.nan
            ndf = pd.DataFrame.from_records(x)
            # [ndf[col].fillna(ndf[col].mean(), inplace=True) for col in ndf.columns]
            x = ndf.iloc[:, 0:].values
            # x = x.astype(np.float32)
            x = np.nan_to_num(x)
            # x = x.astype(np.float64)
        else:
            fp_array = (np.array(fingerprints, dtype=object))
            x = np.vstack(fp_array).astype(np.float32)
            imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
            imp_median.fit(x)
            x = imp_median.transform(x)

        final_array = x

        return final_array


def get_attribute(molecule_list, fp_types=['ecfp2'], k_cluster=100, pool="concat",cluster_or_pca="cluster"):

    featuresGeneration = FeaturesGeneration()

    if cluster_or_pca == "cluster":
        kmeans_predictors = {}
        fp_infos = {}
        for fp_type in fp_types:
            molecule_fp = featuresGeneration.get_fingerprints(molecule_list, fp_name=fp_type)
            kmeans_fp = KMeans(n_clusters=k_cluster, random_state=0)
            kmeans_fp.fit(molecule_fp)
            kmeans_predictors[fp_type] = kmeans_fp
            fp_infos[fp_type] = molecule_fp

        attributes_list = []

        for idx, fp_type in enumerate(fp_types):
            fp = fp_infos[fp_type]
            labels = kmeans_predictors[fp_type].predict(fp)
            one_hot_labels = np.zeros((labels.shape[0], k_cluster))
            one_hot_labels[np.arange(labels.shape[0]), labels] = 1
            attribute = one_hot_labels

            if idx == 0:
                attributes_list = attribute
            elif pool == "sum":
                attributes_list = [s1 + s2 for s1, s2 in zip(attributes_list, attribute)]
            elif pool == "concat":
                attributes_list = [s1.tolist() + s2.tolist() for s1, s2 in zip(attributes_list, attribute)]

            attributes_list = np.array(attributes_list)

    else:
        pca = PCA(n_components=100)
        attributes_list = []

        for idx, fp_type in enumerate(fp_types):
            molecule_fp = featuresGeneration.get_fingerprints(molecule_list, fp_name=fp_type)
            fp_PCA = pca.fit_transform(molecule_fp)

            attribute = fp_PCA

            if idx == 0:
                attributes_list = attribute
            elif pool == "sum":
                attributes_list = [s1 + s2 for s1, s2 in zip(attributes_list, attribute)]
            elif pool == "concat":
                attributes_list = [s1.tolist() + s2.tolist() for s1, s2 in zip(attributes_list, attribute)]

            attributes_list = np.array(attributes_list)

    return attributes_list
