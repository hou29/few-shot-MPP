import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain


# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


# 构造一个数据集中所有分子的 dict , smiles:{graph_data}
def smiles_to_graph_dicts(smiles_list,filename):

    atom_info = {}
    bond_info = {}
    bond_index = {}

    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        molecule_graph_data = mol_to_graph_data_obj_simple(molecule)
        atom_info[smiles] = molecule_graph_data.x
        bond_info[smiles] = molecule_graph_data.edge_attr
        bond_index[smiles] = molecule_graph_data.edge_index

    # 分别对应所有的分子的原子特征、键特征、键索引
    feature_dicts = {
        'atom_info': atom_info,
        'bond_info': bond_info,
        'bond_index': bond_index
    }
    pickle.dump(feature_dicts, open(filename, "wb"))
    print('feature dicts file saved as ' + filename)
    return feature_dicts


def get_graph_data(smilesList, graph_dicts, device="cpu"): # 在模型的前向传播时使用，获取来自batch的 smiles list 的 graph_data list
    x_atom = []
    x_bonds = []
    x_bond_index = []
    for smiles in smilesList:
        if isinstance(smiles, tuple):
            smiles = smiles[0]
        #smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        x_atom.append(graph_dicts['atom_info'][smiles])
        x_bonds.append(graph_dicts['bond_info'][smiles])
        x_bond_index.append(graph_dicts['bond_index'][smiles])
    atom_mask_index = torch.from_numpy(np.concatenate([[item[0]]*item[1] for item in zip(range(len(x_atom)), [item.shape[0] for item in x_atom])]))
    combine_x_atom = torch.vstack(x_atom)
    combine_x_bonds = torch.vstack(x_bonds)
    combine_x_bond_index = torch.hstack([item[0] + item[1] for item in zip(np.cumsum([0] + [item.shape[0] for item in x_atom])[:-1], x_bond_index)])
    return combine_x_atom.to(device), combine_x_bonds.to(device), combine_x_bond_index.to(device), atom_mask_index.to(device)