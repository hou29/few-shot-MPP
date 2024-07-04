import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.metrics import auc as PR_AUC


__all__ = ['get_num_samples', 'get_prototypes', 'prototypical_loss']


def get_num_samples(targets, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, 2))
        targets = targets.to(torch.int64)
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def get_prototypes(embeddings, targets, dist_metric="L2"):
    """Compute the prototypes (the mean vector of the embedded training/support 
    points belonging to its class) for each classes in the task.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor 
        has shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has 
        shape `(batch_size, num_examples)`.

    num_classes : int
        Number of classes in the task.

    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1) # batch_size:how many tasks,
    
    num_samples = get_num_samples(targets, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, 2, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings).cuda()

    embeddings = embeddings.reshape(embeddings.size(0), 2, int(embeddings.size(1)/2), embeddings.size(2))

    #############   求均值获得 class embedding  ##############
    #prototypes = torch.mean(embeddings, dim=2)

    #############   求加权的 class embedding  ##############
    if dist_metric == "L2":
        # 求每个分子到其他分子的距离之和
        dist = torch.zeros(embeddings.size(0), embeddings.size(1), embeddings.size(2),device=embeddings.device)
        for i in range(embeddings.size(0)):
            for j in range(embeddings.size(1)):
                for k in range(embeddings.size(2)):
                    dist[i][j][k] += torch.sum((embeddings[i][j][k].unsqueeze(0) - embeddings[i][j]) ** 2)
        # 求每个分子的attn score
        attn = torch.zeros(embeddings.size(0), embeddings.size(1), embeddings.size(2),device=embeddings.device)
        for i in range(embeddings.size(0)):
            for j in range(embeddings.size(1)):
                all_dist = torch.sum(dist[i][j])#一个类别support集中的分子的距离总和
                attn_low = torch.sum(all_dist / dist[i][j])
                attn[i][j] = (all_dist / dist[i][j]) / attn_low

    elif dist_metric == "cosine_sim":
        # 求每个分子到其他分子的cosine_sim之和
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(2), embeddings.unsqueeze(3), dim=-1)# 计算余弦相似度
        dist = cosine_sim.sum(dim=3)# 将结果累加到 dist 中

        # 求每个分子的attn score( 权重归一化
        all_dists = torch.sum(dist, dim=2)# 计算每个类别的总距离
        attn = dist / all_dists.unsqueeze(2)# 计算注意力权重

    embeddings = embeddings * attn.unsqueeze(3)
    embeddings = embeddings.reshape(embeddings.size(0), 2*embeddings.size(2), embeddings.size(3))

    prototypes.scatter_add_(1, indices.long().cuda(), embeddings.cuda()).div_(num_samples.cuda())

    return prototypes


def prototypical_loss(prototypes, embeddings, targets, dist_metric="cosine_sim", **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical 
    network, on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(batch_size, num_classes, embedding_size)`.

    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(batch_size, num_examples)`.

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    if dist_metric == "L2":
        squared_distances = torch.sum((prototypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1)
        return F.cross_entropy(-squared_distances, targets, **kwargs)

    elif dist_metric == "cosine_sim":
        similarities = torch.einsum('a i c, a j c -> a j i', embeddings, prototypes) #[task_num, 2, query_num],每个query sample在两个类别上的概率
    # ###cos相似度
    # similarities = torch.Tensor(embeddings.shape[0], embeddings.shape[1], prototypes.shape[1]).to(embeddings.device)
    # torch.matmul(embeddings,prototypes)
    # for i in range(embeddings.shape[0]):
    #     for j in range(prototypes.shape[1]):
    #         similarity = F.cosine_similarity(embeddings[i], torch.unsqueeze(prototypes[i][j], 0).repeat(embeddings.shape[1], 1))
    #         similarities[i, :, j] = similarity
    #
    # #targets = torch.unsqueeze(targets, 2)
    # print(similarities.shape)
        return F.cross_entropy(similarities.cuda(), targets.long().cuda(), **kwargs)


def get_proto_acc(prototypes, embeddings, targets, dist_metric="cosine_sim"):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    num_task = targets.shape[0]

    if dist_metric == "cosine_sim":
        #计算acc
        similarities = torch.einsum('a i c, a j c -> a j i', embeddings, prototypes)
        _, predictions = torch.max(similarities, dim=1)
        acc = torch.mean(predictions.eq(targets).float())
        mean_auc = np.mean(
            [roc_auc_score(targets[i].cpu().numpy(), torch.softmax(similarities.permute(0, 2, 1), dim=-1)[i, :, 1].detach().cpu().numpy())
             for i in range(num_task)])

        f1 = np.mean([f1_score(targets[i].cpu().numpy(), predictions[i].cpu().numpy()) for i in range(num_task)])

        # 计算 Precision-Recall 曲线
        precision_recall_list = [precision_recall_curve(targets[i].cpu().numpy(), torch.softmax(similarities.permute(0, 2, 1), dim=-1)[i, :, 1].detach().cpu().numpy()) for i in range(num_task)]
        # 计算 PR-AUC
        pr_auc = np.mean([PR_AUC(precision_recall_list[i][1], precision_recall_list[i][0]) for i in range(num_task)])


    elif dist_metric == "L2":
        #计算 acc
        sq_distances = torch.sum((prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1)
        _, predictions = torch.min(sq_distances, dim=-1)
        acc = torch.mean(predictions.eq(targets).float())
        # 计算 auc
        auc = np.mean([roc_auc_score(targets[i].cpu().numpy(), torch.softmax(-sq_distances, dim=-1)[i, :, 1].detach().cpu().numpy()) for i in range(num_task)])

    return mean_auc, f1, pr_auc
