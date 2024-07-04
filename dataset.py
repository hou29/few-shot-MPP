import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import random
import numpy as np
from utils import get_fp_attribute, get_ss_attribute, FeaturesGeneration



class MyDataset(data.Dataset):
    """
    smiles_tasks_csv: lbaels on every task of all molecules(DataFrame)
    batchsz: how mang times get_item() can call
    tasks: task list(str)
    task_num: the tasks num of get_item() returns

    """
    def __init__(self, smiles_tasks_csv, k_spt_pos, k_spt_neg, k_query, tasks, task_num, batchsz, raw_feature_root, attribute_type='fp',
                 fp_type=['maccs'], ss_type='CGIP_G', cache_file_prefix="", logger=None, pool="concat", cluster_or_pca="cluster" ):
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

        if attribute_type == 'fp':
            fp_str = "_".join(fp_type)
            cache_file_path = f"{cache_file_prefix}{fp_str}.npy"
            cache_file_root = os.path.split(cache_file_path)[0]
            if len(cache_file_root) != 0 and not os.path.exists(cache_file_root):
                os.makedirs(cache_file_root)
            if not os.path.exists(cache_file_path):
                self.all_fps = get_fp_attribute(list(self.smiles_tasks_csv["remained_smiles"].values), fp_type, pool=pool, cluster_or_pca=cluster_or_pca )
                self.print("cache fp data to {}".format(cache_file_path))
                np.save(cache_file_path, self.all_fps)
            else:
                self.print("using cache file: {}".format(cache_file_path))
                self.all_fps = np.load(cache_file_path)
        else:
            cache_file_path = f"{cache_file_prefix}/{ss_type}.npy"
            cache_file_root = os.path.split(cache_file_path)[0]
            if len(cache_file_root) != 0 and not os.path.exists(cache_file_root):
                os.makedirs(cache_file_root)
            if not os.path.exists(cache_file_path):
                self.all_fps = get_ss_attribute(list(self.smiles_tasks_csv["remained_smiles"].values), raw_feature_root, ss_type)
                self.print("cache self-supervised attribute to {}".format(cache_file_path))
                np.save(cache_file_path, self.all_ss)
            else:
                self.print("using cache file: {}".format(cache_file_path))
                self.all_fps = np.load(cache_file_path)

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

    def create_batch2(self):
        """
        create batch for meta-learning.
        batchsz: how many batchs
        self.tasks: task list(str)
        """
        for b in range(self.batchsz):  # for each batch
            # 1.select n_way classes randomly
            # selected_cls = random.choices(self.tasks, k=self.task_num)  #duplicate tasks
            selected_cls = random.sample(self.tasks, k=self.task_num)  # no duplicate task, selected_cls是一个num list
            np.random.shuffle(selected_cls)
            support_x = []
            support_y = []
            support_a = []
            query_x = []
            query_y = []
            query_a = []
            for cls in selected_cls:# for every task
                # 2. select k_shot + k_query for each class
                cls_label = np.array(self.smiles_tasks_csv[cls])# cla_label: all labels of molecules for task cls
                cls_support_x = []
                cls_support_y = []
                cls_support_a = []
                cls_query_x = []
                cls_query_y = []
                cls_query_a = []
                negative_index = np.where(cls_label == 0)[0]
                negative_idx = np.random.choice(len(negative_index), self.k_spt_neg + self.k_query //2, False)
                np.random.shuffle(negative_idx)
                negative_all = list(self.all_smi[negative_index[negative_idx]])
                negative_label_all = list(cls_label[negative_index[negative_idx]])
                negative_attribute_all = list(self.all_fps[negative_index[negative_idx]])
                cls_support_x.extend(negative_all[:self.k_spt_neg])
                cls_support_y.extend(negative_label_all[:self.k_spt_neg])
                cls_support_a.extend(negative_attribute_all[:self.k_spt_neg])
                cls_query_x.extend(negative_all[self.k_spt_neg:])
                cls_query_y.extend(negative_label_all[self.k_spt_neg:])
                cls_query_a.extend(negative_attribute_all[self.k_spt_neg:])


                positive_index = np.where(cls_label == 1)[0]
                if (len(positive_index) < self.k_spt_pos + self.k_query //2):
                    support_positive_idx = np.random.choice(self.k_spt_pos, self.k_spt_pos, False)
                    query_positive_idx = np.random.choice(np.arange(self.k_spt_pos, len(positive_index)), self.k_query // 2, True)
                    positive_idx = np.concatenate((support_positive_idx, query_positive_idx))
                else:
                    positive_idx = np.random.choice(len(positive_index), self.k_spt_pos + self.k_query //2, False)
                np.random.shuffle(positive_idx)
                positive_all = list(self.all_smi[positive_index[positive_idx]])
                positive_label_all = list(cls_label[positive_index[positive_idx]])
                positive_attribute_all = list(self.all_fps[positive_index[positive_idx]])
                cls_support_x.extend(positive_all[:self.k_spt_pos])
                cls_support_y.extend(positive_label_all[:self.k_spt_pos])
                cls_support_a.extend(positive_attribute_all[:self.k_spt_pos])
                cls_query_x.extend(positive_all[self.k_spt_pos:])
                cls_query_y.extend(positive_label_all[self.k_spt_pos:])
                cls_query_a.extend(positive_attribute_all[self.k_spt_pos:])

                c = list(zip(cls_query_x, cls_query_y, cls_query_a))
                random.shuffle(c)
                cls_query_x[:], cls_query_y[:], cls_query_a[:] = zip(*c)

                support_x.append(cls_support_x)
                support_y.append(cls_support_y)
                support_a.append(cls_support_a)
                query_x.append(cls_query_x)
                query_y.append(cls_query_y)
                query_a.append(cls_query_a)
            # support_x = np.array(support_x)
            support_y = np.array(support_y)
            support_a = np.array(support_a)
            # query_x = np.array(query_x)
            query_y = np.array(query_y)
            query_a = np.array(query_a)

            self.batchs_data.append([support_x, support_y, support_a, query_x, query_y, query_a])

    def __getitem__(self, item):
        x_spt, y_spt, a_spt, x_qry, y_qry, a_qry = self.batchs_data[item]
        return x_spt, y_spt, a_spt, x_qry, y_qry, a_qry

    def __len__(self):
        return len(self.batchs_data)
