import os
import numpy as np
import random
import torch
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from torch_sparse import SparseTensor
from torch.utils import data
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix


def preprocess_features(features):
    """Row-normalize feature matrix """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct, correct / len(labels)


def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  '''
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:,1]
    # try:
    #     auc_out = roc_auc_score(labels, pos_probs)
    # except:
    #     auc_out = 0
    auc_out = roc_auc_score(labels, pos_probs)
    return auc_out, pos_probs

def compute_metrics(y_true, y_pred, average='binary'):
    """
    计算 AUC, F1, Sensitivity (Recall), Specificity 指标
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param average: 'binary' 或 'macro' 或 'weighted'，决定了 F1 分数的计算方式
    :return: 包含 AUC, F1, Sensitivity 和 Specificity 的字典
    """

    # F1 Score
    f1 = f1_score(y_true, y_pred, average=average)

    # 混淆矩阵用于计算 Sensitivity 和 Specificity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()  # tn: True Negative, fp: False Positive, fn: False Negative, tp: True Positive

    # Sensitivity (召回率) = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0

    # Specificity (特异性) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    return {'F1': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}


def setup_seed(rs):
    """
    set random seed for reproducing experiments
    :param rs: random seed
    :return: None
    """
    os.environ['PYTHONSEED'] = str(rs)
    np.random.seed(rs)
    random.seed(rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train, num_workers=0)


def get_adj(feature, k=10):
    n_nodes = feature.shape[0]
    d = torch.cdist(feature, feature, p=2)
    sigma = torch.mean(d)
    sim = torch.exp(- d ** 2 / (2 * sigma ** 2)).sort()
    idx = sim.indices[:, -k:]
    wei = sim.values[:, -k:]

    assert n_nodes, k == idx.shape
    assert n_nodes, k == wei.shape

    I = torch.unsqueeze(torch.arange(n_nodes), dim=1).repeat(1, k).view(n_nodes * k).long()
    J = idx.reshape(n_nodes * k).long()
    V = wei.reshape(n_nodes * k)

    edge_weights = V.type(torch.FloatTensor)
    edge_index = SparseTensor(row=I, col=J)

    return edge_index, edge_weights

def construct_knn_graph(features, k=5, distance_metric='euclidean'):
    """
    构建基于 KNN 的图，返回邻接矩阵 (edge_index) 和边权重 (edge_weight)

    参数：
        features (torch.Tensor): 形状为 (num_samples, num_features) 的特征矩阵
        k (int): 每个节点的 K 个最近邻
        distance_metric (str): 用于计算距离的度量方法，默认为欧几里得距离

    返回：
        edge_index (torch.Tensor): 形状为 (2, num_edges) 的邻接矩阵
        edge_weight (torch.Tensor): 形状为 (num_edges,) 的边权重
    """
    # 将 features 转换为 numpy 数组以供 KNN 使用
    features_numpy = features

    # 使用 KNN 找到每个节点的 K 个最近邻
    neighbors = NearestNeighbors(n_neighbors=k, metric=distance_metric)
    neighbors.fit(features_numpy)  # 用特征数据训练 KNN

    # 获取 K 个最近邻的索引
    distances, indices = neighbors.kneighbors(features_numpy)

    edge_index = []
    edge_weight = []

    sigma = np.std(distances)  # 全局尺度参数

    # 对每个节点，获取 K 个邻居并构建双向边
    for i in range(features.shape[0]):  # 对每个节点
        for j in range(k):  # 对每个邻居
            # 添加边（i -> j）
            edge_index.append([i, indices[i, j]])

            edge_weight.append(1.0 / (distances[i, j] + 1e-5))  # 边权重（倒数距离）

    # 转换为 PyTorch 张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return edge_index, edge_weight
