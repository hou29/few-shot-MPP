import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import random
import numpy as np
from utils import get_attribute,FeaturesGeneration



class MyDataset(data.Dataset):
    """
    smiles_tasks_csv: lbaels on every task of all molecules(DataFrame)
    batchsz: how mang times get_item() can call
    tasks: task list(str)
    task_num: the tasks num of get_item() returns

    """
    def __init__(self, smiles_tasks_csv, k_spt_pos, k_spt_neg, k_query, tasks, task_num, batchsz,
                 fp_type=['maccs'], is_cache=False, cache_file_prefix="", logger=None, pool="concat", cluster_or_pca="cluster" ):
        super(MyDataset, self).__init__()
        self.print = print if logger is None else logger.info
        self.smiles_tasks_csv = smiles_tasks_csv

        self.k_spt_pos = k_spt_pos  # k-shot positive
        self.k_spt_neg = k_spt_neg  # k-shot negative
        self.k_query = k_query  # for evaluation
        self.tasks = tasks
        self.task_num = task_num
        self.batchsz = batchsz  #这里指epoch
        self.all_smi = np.array(list(self.smiles_tasks_csv["remained_smiles"].values))
        import time
        if is_cache:
            fp_str = "_".join(fp_type)
            cache_file_path = f"{cache_file_prefix}{fp_str}.npy"
            cache_file_root = os.path.split(cache_file_path)[0]
            if len(cache_file_root) != 0 and not os.path.exists(cache_file_root):
                os.makedirs(cache_file_root)
            if not os.path.exists(cache_file_path):
                self.all_fps = get_attribute(list(self.smiles_tasks_csv["remained_smiles"].values), fp_type, pool=pool, cluster_or_pca=cluster_or_pca )
                self.print("cache fp data to {}".format(cache_file_path))
                np.save(cache_file_path, self.all_fps)
            else:
                self.print("using cache file: {}".format(cache_file_path))
                self.all_fps = np.load(cache_file_path)
        else:
            self.all_fps = get_attribute(list(self.smiles_tasks_csv["remained_smiles"].values), fp_type, pool=pool)
        self.batchs_data = []
        self.create_batch2()

    def select_query(self, cls_label, index_list,):
        test_idx = np.random.choice(len(index_list), self.k_query, False)
        np.random.shuffle(test_idx)
        query_x = list(self.all_smi[index_list[test_idx]])
        query_y = list(cls_label[index_list[test_idx]])

        # query_x = np.array(query_x)
        query_y = np.array(query_y)

        return query_x, query_y


    def __getitem__(self, item):
        x_spt, y_spt, a_spt, x_qry, y_qry, a_qry = self.batchs_data[item]
        return x_spt, y_spt, a_spt, x_qry, y_qry, a_qry

    def __len__(self):
        return len(self.batchs_data)