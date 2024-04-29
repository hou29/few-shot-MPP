import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score


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

    if dist_metric == "L2":
        dist = torch.zeros(embeddings.size(0), embeddings.size(1), embeddings.size(2),device=embeddings.device)
        for i in range(embeddings.size(0)):
            for j in range(embeddings.size(1)):
                for k in range(embeddings.size(2)):
                    dist[i][j][k] += torch.sum((embeddings[i][j][k].unsqueeze(0) - embeddings[i][j]) ** 2)

        attn = torch.zeros(embeddings.size(0), embeddings.size(1), embeddings.size(2),device=embeddings.device)
        for i in range(embeddings.size(0)):
            for j in range(embeddings.size(1)):
                all_dist = torch.sum(dist[i][j])
                attn_low = torch.sum(all_dist / dist[i][j])
                attn[i][j] = (all_dist / dist[i][j]) / attn_low

    elif dist_metric == "cosine_sim":
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(2), embeddings.unsqueeze(3), dim=-1)
        dist = cosine_sim.sum(dim=3)

        all_dists = torch.sum(dist, dim=2)
        attn = dist / all_dists.unsqueeze(2)

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
        similarities = torch.einsum('a i c, a j c -> a j i', embeddings, prototypes)
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
        similarities = torch.einsum('a i c, a j c -> a j i', embeddings, prototypes)
        _, predictions = torch.max(similarities, dim=1)
        acc = torch.mean(predictions.eq(targets).float())
        auc = np.mean(
            [roc_auc_score(targets[i].cpu().numpy(), torch.softmax(similarities.permute(0, 2, 1), dim=-1)[i, :, 1].detach().cpu().numpy())
             for i in range(num_task)])
    elif dist_metric == "L2":
        sq_distances = torch.sum((prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1)
        _, predictions = torch.min(sq_distances, dim=-1)
        acc = torch.mean(predictions.eq(targets).float())
        auc = np.mean([roc_auc_score(targets[i].cpu().numpy(), torch.softmax(-sq_distances, dim=-1)[i, :, 1].detach().cpu().numpy()) for i in range(num_task)])

    return acc, auc