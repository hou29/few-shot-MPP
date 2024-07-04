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
import pickle



def load_dataset_scaffold(path, dataset='hiv', seed=628, tasks=None):
    save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test

    pyg_dataset = MultiDataset(root=path, dataset=dataset, tasks=tasks)
    df = pd.read_csv(os.path.join(path, 'raw/{}.csv'.format(dataset)))
    smilesList = df.smiles.values
    print("number of all smiles: ", len(smilesList))
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            remained_smiles.append(smiles)
        except:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    df = df[df["smiles"].isin(remained_smiles)].reset_index()

    trn_id, val_id, test_id, weights = scaffold_randomized_spliting(df, tasks=tasks, random_seed=seed)
    trn, val, test = pyg_dataset[torch.LongTensor(trn_id)], \
                     pyg_dataset[torch.LongTensor(val_id)], \
                     pyg_dataset[torch.LongTensor(test_id)]
    trn.weights = weights

    torch.save([trn, val, test], save_path)
    return load_dataset_scaffold(path, dataset, seed, tasks)

# copy from xiong et al. attentivefp
class ScaffoldGenerator(object):
    """
    Generate molecular scaffolds.
    Parameters
    ----------
    include_chirality : : bool, optional (default False)
        Include chirality in scaffolds.
    """

    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol):
        """
        Get Murcko scaffolds for molecules.
        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.
        Parameters
        ----------
        mols : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality)

# copy from xiong et al. attentivefp
def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)
    return scaffold

# copy from xiong et al. attentivefp
def split(scaffolds_dict, smiles_tasks_df, tasks, weights, sample_size, random_seed=0):
    count = 0
    minor_count = 0
    minor_class = np.argmax(weights[0])  # weights are inverse of the ratio
    minor_ratio = 1 / weights[0][minor_class]
    optimal_count = 0.1 * len(smiles_tasks_df)
    while (count < optimal_count * 0.9 or count > optimal_count * 1.1) \
            or (minor_count < minor_ratio * optimal_count * 0.9 \
                or minor_count > minor_ratio * optimal_count * 1.1):
        random_seed += 1
        random.seed(random_seed)
        scaffold = random.sample(list(scaffolds_dict.keys()), sample_size)
        count = sum([len(scaffolds_dict[scaffold]) for scaffold in scaffold])
        index = [index for scaffold in scaffold for index in scaffolds_dict[scaffold]]
        minor_count = len(smiles_tasks_df.iloc[index, :][smiles_tasks_df[tasks[0]] == minor_class])
    #     print(random)
    return scaffold, index

def scaffold_randomized_spliting(smiles_tasks_df, tasks=['HIV_active'], random_seed=8):
    weights = []
    for i, task in enumerate(tasks):
        negative_df = smiles_tasks_df[smiles_tasks_df[task] == 0][["smiles", task]]
        positive_df = smiles_tasks_df[smiles_tasks_df[task] == 1][["smiles", task]]
        weights.append([(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0], \
                        (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]])
    print('The dataset weights are', weights)
    print('generating scaffold......')
    scaffold_list = []
    all_scaffolds_dict = {}
    for index, smiles in enumerate(smiles_tasks_df['smiles']):
        scaffold = generate_scaffold(smiles)
        scaffold_list.append(scaffold)
        if scaffold not in all_scaffolds_dict:
            all_scaffolds_dict[scaffold] = [index]
        else:
            all_scaffolds_dict[scaffold].append(index)
    #     smiles_tasks_df['scaffold'] = scaffold_list

    samples_size = int(len(all_scaffolds_dict.keys()) * 0.1)
    test_scaffold, test_index = split(all_scaffolds_dict, smiles_tasks_df, tasks, weights, samples_size,
                                      random_seed=random_seed)
    training_scaffolds_dict = {x: all_scaffolds_dict[x] for x in all_scaffolds_dict.keys() if x not in test_scaffold}
    valid_scaffold, valid_index = split(training_scaffolds_dict, smiles_tasks_df, tasks, weights, samples_size,
                                        random_seed=random_seed)

    training_scaffolds_dict = {x: training_scaffolds_dict[x] for x in training_scaffolds_dict.keys() if
                               x not in valid_scaffold}
    train_index = []
    for ele in training_scaffolds_dict.values():
        train_index += ele
    assert len(train_index) + len(valid_index) + len(test_index) == len(smiles_tasks_df)

    return train_index, valid_index, test_index, weights


def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_attr(mol, explicit_H=True, use_chirality=True):
    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        # if atom.GetDegree()>5:
        #     print(Chem.MolToSmiles(mol))
        #     print(atom.GetSymbol())
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
             ]) + onehot_encoding(atom.GetDegree(),
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + onehot_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            #                 print(one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')])
            except:
                results = results + [0, 0] + [atom.HasProp('_ChiralityPossible')]
        feat.append(results)

    return np.array(feat)


def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)

class MultiDataset(InMemoryDataset):

    def __init__(self, root, dataset, tasks, transform=None, pre_transform=None, pre_filter=None):
        self.tasks = tasks
        self.dataset = dataset

        self.weights = 0
        super(MultiDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # os.remove(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        print("number of all smiles: ", len(smilesList))
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
                remained_smiles.append(smiles)
            except:
                print("not successfully processed smiles: ", smiles)
                pass
        print("number of successfully processed smiles: ", len(remained_smiles))

        df = df[df["smiles"].isin(remained_smiles)].reset_index()
        target = df[self.tasks].values
        smilesList = df.smiles.values
        data_list = []

        for i, smi in enumerate(tqdm(smilesList)):

            mol = MolFromSmiles(smi)
            data = self.mol2graph(mol)

            if data is not None:
                label = target[i]
                label[np.isnan(label)] = 6
                data.y = torch.LongTensor([label])
                if self.dataset == 'esol' or self.dataset == 'freesolv' or self.dataset == 'lipophilicity':
                    data.y = torch.FloatTensor([label])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def mol2graph(self, mol):
        if mol is None: return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        # pos = torch.FloatTensor(geom)
        data = Data(
            x=torch.FloatTensor(node_attr),
            # pos=pos,
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            y=None  # None as a placeholder
        )
        return data


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
        # for smi in smiles_list:
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


#输入：一个smiles的list；输出：一个属性tensor（这里的属性是聚类获得的）
def get_attribute_0(molecule_list, fp_types=['ecfp2'], k_cluster=100):
    ###聚类训练
    #调用FeaturesGeneration类
    featuresGeneration = FeaturesGeneration()
    # 聚类数量
    k = k_cluster
    molecule_list = [item[0] for item in molecule_list]

    """ 对所有的分子根据不同的指纹聚类"""
    # #ecfp0
    # molecule_ecfp0 = featuresGeneration.get_fingerprints(molecule_list, fp_name='ecfp0')
    # kmeans_ecfp0 = KMeans(n_clusters=k, random_state=0)
    # kmeans_ecfp0.fit(molecule_ecfp0)
    # #ecfp2
    # molecule_ecfp2 = featuresGeneration.get_fingerprints(molecule_list, fp_name='ecfp2')
    # kmeans_ecfp2 = KMeans(n_clusters=k, random_state=0)
    # kmeans_ecfp2.fit(molecule_ecfp2)
    #mol = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    attributes_list = [] # 一个SMILES的list的属性，数据类型是str

    for fp_type in fp_types:
        fp = featuresGeneration.get_fingerprints(molecule_list, fp_name=fp_type)
        # if fp_type == 'ecfp0':
        #     labels = kmeans_ecfp0.predict(fp)
        # elif fp_type == 'ecfp2':
        #     labels = kmeans_ecfp2.predict(fp)
        #     one_hot_labels = np.eye(k)[int(labels[0])].astype(int)
        #     one_hot_labels = ''.join(map(str, one_hot_labels))
        if fp_type == 'maccs':
            fp = [sub_array[1:] for sub_array in fp]
            fp = torch.tensor(fp)
            #attributes_list

    return fp


#输入：一个list(一个数据集所有的smiles）；输出：一个数组（每个分子对应的指纹，这里的属性是聚类获得的）
def get_fp_attribute(molecule_list, fp_types=['ecfp2'], k_cluster=100, pool="concat",cluster_or_pca="cluster"):

    #调用FeaturesGeneration类
    featuresGeneration = FeaturesGeneration()

    if cluster_or_pca == "cluster":
        #对所有的分子根据不同的指纹聚类，共使用 15 种指纹
        kmeans_predictors = {}
        fp_infos = {}
        for fp_type in fp_types:
            molecule_fp = featuresGeneration.get_fingerprints(molecule_list, fp_name=fp_type)
            # if === "kmeans":
            kmeans_fp = KMeans(n_clusters=k_cluster, random_state=0)
            kmeans_fp.fit(molecule_fp)
            kmeans_predictors[fp_type] = kmeans_fp
            fp_infos[fp_type] = molecule_fp


        attributes_list = [] # 一个SMILES的list的属性，数据类型是str

        for idx, fp_type in enumerate(fp_types):
            fp = fp_infos[fp_type]
            labels = kmeans_predictors[fp_type].predict(fp)

            # if fp_type == 'maccs':
            #     #fp = [sub_array[1:] for sub_array in fp]
            #     labels = kmeans_maccs.predict(fp)
            # elif fp_type == 'ecfp0':
            #     labels = kmeans_ecfp0.predict(fp)
            # elif fp_type == 'ecfp2':
            #     labels = kmeans_ecfp2.predict(fp)
            # elif fp_type == 'ecfp4':
            #     labels = kmeans_ecfp4.predict(fp)
            # elif fp_type == 'ecfp6':
            #     labels = kmeans_ecfp6.predict(fp)
            # elif fp_type == 'fcfp2':
            #     labels = kmeans_fcfp2.predict(fp)
            # elif fp_type == 'fcfp4':
            #     labels = kmeans_fcfp4.predict(fp)
            # elif fp_type == 'fcfp6':
            #     labels = kmeans_fcfp6.predict(fp)
            # elif fp_type == 'rdk5':
            #     labels = kmeans_rdk5.predict(fp)
            # elif fp_type == 'rdk6':
            #     labels = kmeans_rdk6.predict(fp)
            # elif fp_type == 'rdk7':
            #     labels = kmeans_rdk7.predict(fp)
            # elif fp_type == 'hashtt':
            #     labels = kmeans_hashtt.predict(fp)
            # elif fp_type == 'hashap':
            #     labels = kmeans_hashap.predict(fp)
            # elif fp_type == 'avalon':
            #     labels = kmeans_avalon.predict(fp)
            # elif fp_type == 'rdkDes':
            #    labels = kmeans_rdkDes.predict(fp)
            one_hot_labels = np.zeros((labels.shape[0], k_cluster))
            one_hot_labels[np.arange(labels.shape[0]), labels] = 1# 使用fancy indexing来设置每行中相应的元素为1
            attribute = one_hot_labels

            if idx == 0:
                attributes_list = attribute
            elif pool == "sum":
                attributes_list = [s1 + s2 for s1, s2 in zip(attributes_list, attribute)]
            elif pool == "concat":
                attributes_list = [s1.tolist() + s2.tolist() for s1, s2 in zip(attributes_list, attribute)]

            attributes_list = np.array(attributes_list)

    else:
        #计算所有的分子 PCA降维后的15种指纹
        pca = PCA(n_components=100)
        attributes_list = []  # 一个list保存所有分子的最终的属性，数据类型是str

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

def get_ss_attribute(molecule_list, raw_feature_root, ss_types='GCIP_G'):
    attributes_list = []
    pca = PCA(n_components=100)
    attributes_list = []  # 一个list保存所有分子的最终的属性，数据类型是str

    with open(f'{raw_feature_root}/{ss_types}.pkl', 'rb') as file:
        data = pickle.load(file)

    index_list = data['index']
    smiles_list = data['smiles']
    features_list = data['x']

    # 创建一个字典，以便快速查找分子的特征
    features_dict = {smiles_list[i]: features_list[i] for i in range(len(smiles_list))}

    # 提取分子列表中所有分子的特征
    selected_features = [features_dict[smiles] for smiles in molecule_list if smiles in features_dict]

    pca = PCA(n_components=100)
    fp_PCA = pca.fit_transform(selected_features)

    ss_attributes_list = fp_PCA

    return ss_attributes_list
