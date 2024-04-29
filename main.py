import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("/home/lzeng/hll/attribute-graph/")
from dataset import *
import random
import sys
import argparse
import statistics
import pandas as pd
import time, datetime
from prototype import get_prototypes,prototypical_loss,get_proto_acc
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from logger import Logger
from GNN_models import *
from graph_data_pre import *


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=188, type=int, help='')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--dataset', default='tox21', type=str, choices=['tox21', 'sider', 'muv'])
parser.add_argument('--log_dir', default="./log", type=str, help='log dir')
parser.add_argument('--gpu', default='0', type=str, help='GPU')
parser.add_argument('--attributes', default='ecfp0', type=str, help='If there are multiple attributes, separate them with commas')
parser.add_argument('--fp_cache_root', default='./cache/tox21_fp_PCA/one_attr/', type=str, help='')

parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_tasks', type=int, default=9, help='task num in one batch for training (default:9)')
parser.add_argument('--k_shot', type=int, default=10, help='K-shot')
parser.add_argument('--epochs', type=int, default=100, help='dataset_train duplicate (default: 100)')
parser.add_argument('--num_layer', type=int, default=5, help='number of GNN(GAT) message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=100, help='embedding dimensions (default: 300)')
parser.add_argument('--dropout_ratio', type=float, default=0.2, help='dropout ratio (default: 0.2)')
parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
parser.add_argument('--gnn_type', type=str, default="gat")
parser.add_argument('--pretrained_bool', type=str, default= 'Fasle', help='use pretrained GNN or not')

parser.add_argument('--pool', type=str, default="concat", choices=['sum', 'concat'], help='')
parser.add_argument('--run_time', type=str, default="0", help='one exp run multiple times')
parser.add_argument('--with_attr', type=str, default="True", help='use attributes or not')
parser.add_argument('--cluster_or_pca', type=str, default="cluster", help='if not in cache,use which method to decrease dim')
parser.add_argument('--no_pca', type=str, default="False", help='')
args = parser.parse_args()


args.attributes = args.attributes.split(",")
attributes_str = "_".join(args.attributes)
if args.with_attr == "False":
    attributes_str = 'none'

device = torch.device("cuda:" + args.gpu)

args.saved_model_dir = args.log_dir

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
if not os.path.exists(args.saved_model_dir):
    os.makedirs(args.saved_model_dir)

log_file_name = "{}/gnn_type:{}_lr:{}_{}.log".format(args.log_dir, args.gnn_type, args.lr, attributes_str)
saved_model_path = "{}/gnn_type:{}_lr:{}_{}.pth".format(args.saved_model_dir, args.gnn_type, args.lr, attributes_str)
logger = Logger(filename=log_file_name).get_logger()

seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

k_spt_pos = args.k_shot
k_spt_neg = args.k_shot
k_query = 32

episode = 20

test_batchsz = 20
epochs = 100

if args.dataset == 'tox21':
    raw_filename = "data/tox21.csv"
    tasks = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
             "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
             "SR-HSE", "SR-MMP", "SR-p53", ]
    task_num = 9
    test_task_num = 3
elif args.dataset == 'sider':
    raw_filename = "data/sider.csv"
    tasks = ["SIDER1", "SIDER2", "SIDER3", "SIDER4", "SIDER5", "SIDER6", "SIDER7", "SIDER8", "SIDER9", "SIDER10",
            "SIDER11", "SIDER12", "SIDER13", "SIDER14", "SIDER15", "SIDER16", "SIDER17", "SIDER18", "SIDER19", "SIDER20",
            "SIDER21", "SIDER22", "SIDER23", "SIDER24", "SIDER25", "SIDER26", "SIDER27"]
    task_num = 21
    test_task_num = 6
elif args.dataset == 'muv':
    raw_filename = "data/muv.csv"
    tasks = ["MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689", "MUV-692", "MUV-712", "MUV-713", "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"]
    task_num = 12
    test_task_num = 5


smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.smiles.values
logger.info("number of all smiles: {}".format(len(smilesList)))

remained_smiles = []
for smiles in smilesList:
    try:
        remained_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles),  isomericSmiles=True))
    except:
        logger.info(smiles)
        pass
logger.info("number of successfully processed smiles: {}".format(len(remained_smiles)))

feature_filename = raw_filename.replace('.csv', '.pickle')
if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb"))
else:
    feature_dicts = smiles_to_graph_dicts(remained_smiles, feature_filename)

smiles_tasks_df['remained_smiles'] = remained_smiles
remained_df = smiles_tasks_df
logger.info("number of remained  smiles: {}".format(len(remained_df.smiles.values)))

data_train = MyDataset(remained_df, k_spt_pos, k_spt_neg, k_query, tasks[:task_num], task_num, epochs, fp_type=args.attributes, is_cache=True, cache_file_prefix=f"{args.fp_cache_root}", logger=logger, pool=args.pool, cluster_or_pca=args.cluster_or_pca)
dataset_train = data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)

data_test = MyDataset(remained_df, k_spt_pos, k_spt_neg, k_query, tasks[task_num:], test_task_num, test_batchsz, fp_type=args.attributes, is_cache=True, cache_file_prefix=f"{args.fp_cache_root}", logger=logger, pool=args.pool, cluster_or_pca=args.cluster_or_pca)
dataset_test = data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)


# 定义模型和优化器
if args.pool == 'sum':
    attr_dim = 100
else:
    attr_dim = len(args.attributes) *100
if args.no_pca == 'True':
    attr_dim = 1024

model = attributes_GNN(args.num_layer, args.emb_dim, attr_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type, with_attr=args.with_attr, pretrained_bool=args.pretrained_bool, device=device).to(device)#GNN得到的分子只包含分子编码的信息，表示不使用属性信息
meta_optim = torch.optim.Adam(model.parameters(), lr= args.lr)
print(model)
#
# for name, parameters in model.named_parameters():
#     print(name, ':', parameters.size())
result_dict = {
    "test_auc": {
        "best_test_auc": -np.inf,
        "according_std_dev": -np.inf,
        "epoch": -1,
        "step": -1,
        "cor_test_acc": -np.inf,

        "cor_train_loss": -np.inf,
        "cor_train_acc": -np.inf,
        "cor_train_auc": -np.inf,
    },
    "test_acc": {
        "best_test_acc": -np.inf,
        "epoch": -1,
        "step": -1,
        "cor_test_auc": -np.inf,

        "cor_train_loss": -np.inf,
        "cor_train_acc": -np.inf,
        "cor_train_auc": -np.inf,
    },


}
patience_acc = 0
patience_auc = 0
n_patience = 100
flag_early_stop = False
max_step_all_epoch = len(dataset_train) * episode
logger.info(f"max_step_all_epoch: {max_step_all_epoch}")

for epoch in range(episode):
    for step, (x_spt, y_spt, a_spt, x_qry, y_qry, a_qry) in enumerate(dataset_train):
        task_num = len(x_spt)
        shot = len(x_spt[0])
        query_size = len(x_qry[0])
        _,_,_,attributes_size = a_spt.size()
        y_spt = y_spt.view(task_num, shot).long().to(device)
        y_qry = y_qry.view(task_num, query_size).long().to(device)
        a_spt = a_spt.view(task_num, shot,attributes_size).to(device)
        a_qry = a_qry.view(task_num, query_size, attributes_size).to(device)

        start_time = datetime.datetime.now()
        task_num = len(x_spt)
        support_embeddings_list = []
        query_embeddings_list = []
        for i in range(task_num):
            support_atom, support_bonds, support_bond_index, support_atom_mask_index = get_graph_data(x_spt[i], feature_dicts, device=device)
            support_embeddings = model(support_atom, support_bond_index, support_bonds, support_atom_mask_index, a_spt[i])
            support_embeddings_list.append(support_embeddings)
            query_atom, query_bonds, query_bond_index, query_atom_mask_index = get_graph_data(x_qry[i], feature_dicts, device=device)
            query_embeddings = model(query_atom, query_bond_index, query_bonds, query_atom_mask_index, a_qry[i])
            query_embeddings_list.append(query_embeddings)
        support_embeddings = torch.stack(support_embeddings_list)
        if args.k_shot == 1:
            prototypes = support_embeddings
        else:
            prototypes = get_prototypes(support_embeddings, y_spt)

        query_embeddings = torch.stack(query_embeddings_list)
        train_loss = prototypical_loss(prototypes, query_embeddings, y_qry)
        train_acc, train_auc = get_proto_acc(prototypes, query_embeddings, y_qry)

        meta_optim.zero_grad()
        train_loss.backward()
        meta_optim.step()

        # print("="*100)
        # print(list(model.parameters())[0].mean())
        if (step+1) % 5 == 0:
            start_time = datetime.datetime.now()
            # logger.info(start_time, "    @@@@@@@@@@@@@@@@@@@@")
            logger.info("epoch: {}; step: {}".format(epoch, step))
            logger.info("meta-train_loss: {}; meta-train_acc: {}; meta-train_auc: {}".format(train_loss.item(), train_acc.item(), train_auc.item()))
            # print(loss)

            # evaluation on test
            test_acc_list, test_auc_list = [],[]
            for x_spt, y_spt, a_spt, x_qry, y_qry ,a_qry in dataset_test:
                task_num = len(x_spt)
                shot = len(x_spt[0])
                query_size = len(x_qry[0])
                _, _, _, attributes_size = a_spt.size()
                y_spt = y_spt.view(task_num, shot).long().to(device)
                y_qry = y_qry.view(task_num, query_size).long().to(device)
                a_spt = a_spt.view(task_num, shot, attributes_size).to(device)
                a_qry = a_qry.view(task_num, query_size, attributes_size).to(device)

                support_embeddings_list = []
                query_embeddings_list = []
                for task_index, (x_spt_one, y_spt_one, a_spt_one, x_qry_one, y_qry_one, a_qry_one) in enumerate(zip(x_spt, y_spt, a_spt, x_qry, y_qry ,a_qry)):
                    support_atom, support_bonds, support_bond_index, support_atom_mask_index = get_graph_data(x_spt_one, feature_dicts, device=device)
                    support_embeddings = model(support_atom, support_bond_index, support_bonds, support_atom_mask_index, a_spt_one)
                    support_embeddings_list.append(support_embeddings)
                    query_atom, query_bonds, query_bond_index, query_atom_mask_index = get_graph_data(x_qry_one, feature_dicts, device=device)
                    query_embeddings = model(query_atom, query_bond_index, query_bonds, query_atom_mask_index, a_qry_one)
                    query_embeddings_list.append(query_embeddings)
                support_embeddings = torch.stack(support_embeddings_list)
                if args.k_shot == 1:
                    prototypes = support_embeddings
                else:
                    prototypes = get_prototypes(support_embeddings, y_spt)

                query_embeddings = torch.stack(query_embeddings_list)
                test_acc, test_auc = get_proto_acc(prototypes, query_embeddings, y_qry)
                test_acc_list.append(test_acc.item())
                test_auc_list.append(test_auc.item())


            mean_acc = sum(test_acc_list)/len(test_acc_list)
            mean_auc = sum(test_auc_list)/len(test_auc_list)
            std_dev = statistics.stdev(test_auc_list)
            logger.info(' test_acc：{};test_auc：{}；std{}'.format(mean_acc, mean_auc, std_dev))

            if mean_acc > result_dict["test_acc"]["best_test_acc"]:
                result_dict["test_acc"] = {
                    "best_test_acc": mean_acc,
                    "epoch": epoch,
                    "step": step,
                    "cor_test_auc": mean_auc,

                    "cor_train_loss": train_loss,
                    "cor_train_acc": train_acc,
                    "cor_train_auc": train_auc,
                }
                patience_acc = 0
            else:
                patience_acc += 1

            if mean_auc > result_dict["test_auc"]["best_test_auc"]:
                result_dict["test_auc"] = {
                    "best_test_auc": mean_auc,
                    "according_std_dev": std_dev,
                    "epoch": epoch,
                    "step": step,
                    "cor_test_acc": mean_acc,

                    "cor_train_loss": train_loss,
                    "cor_train_acc": train_acc,
                    "cor_train_auc": train_auc,
                }
                patience_auc = 0
                if os.path.exists(saved_model_path):
                    os.remove(saved_model_path)
                torch.save(model.state_dict(), saved_model_path)
            else:
                patience_auc += 1
                
        if min(patience_acc, patience_auc) > n_patience:
            flag_early_stop = True
            break
            
    if flag_early_stop:
        break

logger.info(result_dict)