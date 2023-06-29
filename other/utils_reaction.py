import profile

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import argparse
import numpy as np
import scipy.sparse as sp
import math
from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix, dense_to_sparse, is_undirected, coalesce
from torch_geometric.utils import contains_isolated_nodes, degree, remove_self_loops, k_hop_subgraph
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikipediaNetwork
import torch.nn.functional as F
import sys
import os.path
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os.path as osp
from tqdm import tqdm
import mat73
from time import time
import cProfile
import wrapt
from line_profiler import LineProfiler
import pstats
import io
from numba import jit
import scipy.io as sio
# profiler = LineProfiler()


cur_dir = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append('%s/software/' % par_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def profile(func):
#     def inner(*args, **kwargs):
#         profiler.add_function(func)
#         profiler.enable_by_count()
#         return func(*args, **kwargs)
#     return inner
#
#
# def print_stats():
#     profiler.print_stats()

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def obtain_a_p(data):
    # the adjacency matrix
    index = torch.cat((data.node_idx, data.edge_idx), dim=0)
    value = torch.ones_like(data.edge_idx).view(-1).to(torch.float)

    i = torch.sparse_coo_tensor(index, value, (data.node_num, data.edge_num))
    a = torch.sparse.mm(i, i.t())
    a = a.coalesce()
    data.edge_index_a, data.edge_weight_a = a.indices().to(torch.long), a.values().to(torch.float)
    data.edge_index_a, data.edge_weight_a = remove_self_loops(data.edge_index_a, data.edge_weight_a)

    # the intersection profile
    p = torch.sparse.mm(i.t(), i)
    p = p.coalesce()
    data.edge_index_p, data.edge_weight_p = p.indices().to(torch.long), p.values().to(torch.float)
    data.edge_index_p, data.edge_weight_p = remove_self_loops(data.edge_index_p, data.edge_weight_p)
    data.it = i.t()

    return data


def ego_graph_minus(data, edge, h, args):
    # the ego-graph in the adjaceny matarix
    num_nodes_a = torch.max(data.edge_index_a) + 1
    sub_nodes, edge_index_a, mapping, edge_mask = k_hop_subgraph(edge.view(-1), args.num_hops, data.edge_index_a,
                                                                 relabel_nodes=True, num_nodes=num_nodes_a)
    num_nodes_a = torch.max(edge_index_a) + 1
    node_mask_a = torch.zeros(num_nodes_a, dtype=torch.bool)

    # mask of the nodes contained in a hyperedge
    node_mask_a[mapping] = True
    edge_weight_a = data.edge_weight_a[edge_mask].view(-1, 1)

    # mask of the edges run among the nodes
    edge_mask_row = torch.zeros(edge_index_a.size(1), dtype=torch.bool)
    edge_mask_col = torch.zeros(edge_index_a.size(1), dtype=torch.bool)

    row, col = edge_index_a
    torch.index_select(node_mask_a, 0, row, out=edge_mask_row)
    torch.index_select(node_mask_a, 0, col, out=edge_mask_col)
    edge_mask_a = torch.logical_and(edge_mask_row, edge_mask_col)

    data_a = Data(edge_index=edge_index_a, edge_attr=edge_weight_a, num_nodes=num_nodes_a)
    data_a.edge_weight = edge_weight_a
    data_a.edge_mask = edge_mask_a
    data_a.node_mask = node_mask_a

    return data_a


def ego_graph_plus(data, edge, args):
    # add a hyperedge in the adjacency matrix
    node_paris = torch.combinations(edge.view(-1)).transpose(0, 1)
    edge_index_a = torch.cat((data.edge_index_a, node_paris), dim=1)
    edge_index_a = torch.cat((edge_index_a, node_paris[[1, 0], :]), dim=1)
    edge_weight_a = torch.cat((data.edge_weight_a, torch.ones(node_paris.size(1) * 2)), dim=0)
    edge_index_a, edge_weight_a = coalesce(edge_index_a, edge_weight_a)

    num_nodes_a = torch.max(edge_index_a) + 1
    sub_nodes, edge_index_a, mapping, edge_mask = k_hop_subgraph(edge.view(-1), args.num_hops, edge_index_a,
                                                                 relabel_nodes=True, num_nodes=num_nodes_a)
    num_nodes_a = torch.max(edge_index_a) + 1
    node_mask_a = torch.zeros(num_nodes_a, dtype=torch.bool)
    node_mask_a[mapping] = True
    edge_mask_row = torch.zeros(edge_index_a.size(1), dtype=torch.bool)
    edge_mask_col = torch.zeros(edge_index_a.size(1), dtype=torch.bool)

    row, col = edge_index_a
    torch.index_select(node_mask_a, 0, row, out=edge_mask_row)
    torch.index_select(node_mask_a, 0, col, out=edge_mask_col)
    edge_mask_a = torch.logical_and(edge_mask_row, edge_mask_col)  # node pairs run inside the hyper edge
    edge_weight_a = edge_weight_a[edge_mask].view(-1, 1)

    data_a = Data(edge_index=edge_index_a, edge_attr=edge_weight_a, num_nodes=num_nodes_a)
    data_a.edge_weight = edge_weight_a
    data_a.edge_mask = edge_mask_a
    data_a.node_mask = node_mask_a

    return data_a


def load_splitted_data(args):
    par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_name = args.data + "_split_" + args.data_split
    data_dir = os.path.join(par_dir, "data/{}.mat".format(data_name))

    data = mat73.loadmat(data_dir)

    S = data['train']
    vertices = list(range(S.shape[0]))
    train_data = []
    for e in range(S.shape[1]):
        tS = np.squeeze(np.asarray(S[:, e]))
        lx = np.array(np.nonzero(tS)).tolist()
        lx = sum(lx, [])
        train_data.append(lx)
        # if len(lx) > 1:
        #     train_data.append(lx)
        # else:
        #     pass

    train_label = np.ones((len(train_data),), dtype=int)

    U = data['test']
    vertices = list(range(U.shape[0]))
    test_data = []
    for e in range(U.shape[1]):
        tU = np.squeeze(np.asarray(U[:, e]))
        lx = np.array(np.nonzero(tU)).tolist()
        lx = sum(lx, [])
        test_data.append(lx)


    test_label = data['labels'].astype(int)
    print(data['missnumber'])


    # Convert to Torch tensors
    train_label = torch.tensor(train_label, dtype=torch.long)
    test_lb = torch.tensor(test_label, dtype=torch.long)

    pos_ind = torch.where(train_label == 1)[0]
    neg_ind = torch.where(train_label == 0)[0]
    pos_val_size = int(pos_ind.size(0) * args.val_ratio)
    neg_val_size = int(neg_ind.size(0) * args.val_ratio)
    is_train = torch.ones_like(train_label).to(torch.bool)
    perm = torch.randperm(pos_ind.size(0))
    is_train[pos_ind[perm[:pos_val_size]]] = False
    perm = torch.randperm(neg_ind.size(0))
    is_train[neg_ind[perm[:neg_val_size]]] = False

    train_edges = list()
    val_edges = list()
    test_edges = list()
    train_lb = list()
    val_lb = list()
    train_pos_id = list()
    val_pos_id = list()

    # contruct the observed hypergraph
    edge_id = -1
    num_nodes = 0
    node_idx, edge_idx = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    for ind, edge in enumerate(train_data):
        nodes = torch.tensor(edge).view(1, -1).to(torch.long)
        max_node = torch.max(nodes)
        if max_node > num_nodes:
            num_nodes = max_node
        if is_train[ind]:
            train_edges.append(nodes)
            train_lb.append(train_label[ind])
            if train_label[ind] == 1:
                edge_id += 1
                edges = torch.full(nodes.size(), edge_id).to(torch.long)
                node_idx = torch.cat((node_idx, nodes), dim=1)
                edge_idx = torch.cat((edge_idx, edges), dim=1)
                train_pos_id.append(edge_id)
        else:
            val_edges.append(nodes)
            val_lb.append(train_label[ind])
            if train_label[ind] == 1:
                edge_id += 1
                edges = torch.full(nodes.size(), edge_id).to(torch.long)
                node_idx = torch.cat((node_idx, nodes), dim=1)
                edge_idx = torch.cat((edge_idx, edges), dim=1)
                val_pos_id.append(edge_id)

    for ind, edge in enumerate(test_data):
        max_node = torch.max(nodes)
        if max_node > num_nodes:
            num_nodes = max_node
        nodes = torch.tensor(edge).view(1, -1).to(torch.long)
        test_edges.append(nodes)

    data = Data()
    data.node_idx = node_idx
    data.edge_idx = edge_idx
    data.node_num = int(num_nodes + 1)
    data.edge_num = int(edge_id + 1)
    data = obtain_a_p(data)

    train_lb = torch.tensor(train_lb).to(torch.long)
    val_lb = torch.tensor(val_lb).to(torch.long)

    return data, train_edges, val_edges, test_edges, train_lb, val_lb, test_lb, train_pos_id, val_pos_id


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    # adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def obtain_walk_profile(data, args):
    walk_profile = list()
    row, col = data.edge_index
    row, col = row.view(-1), col.view(-1)
    if data.edge_index.size(1) == 0:
        return torch.zeros(1, args.walk_len * 10)

    g = sp.csr_matrix((data.edge_weight.view(-1), (row, col)), shape=(data.num_nodes, data.num_nodes))
    mat_p = sys_normalized_adjacency(g)
    mat_p = sparse_mx_to_torch_sparse_tensor(mat_p).cuda()
    x_p = torch.eye(data.num_nodes, dtype=torch.float32).cuda()


    # for adjacency matrix, the edge weight substract by one
    edge_weight = data.edge_weight
    edge_weight[data.edge_mask] -= 1.0

    g1 = sp.csr_matrix((edge_weight.view(-1), (row, col)), shape=(data.num_nodes, data.num_nodes))
    mat_m1 = sys_normalized_adjacency(g1)
    mat_m1 = sparse_mx_to_torch_sparse_tensor(mat_m1).cuda()
    x_m1 = torch.eye(data.num_nodes, dtype=torch.float32).cuda()


    edge_weight = data.edge_weight
    edge_weight[data.edge_mask] = 0

    g2 = sp.csr_matrix((edge_weight.view(-1), (row, col)), shape=(data.num_nodes, data.num_nodes))
    mat_m2 = sys_normalized_adjacency(g2)
    mat_m2 = sparse_mx_to_torch_sparse_tensor(mat_m2).cuda()
    x_m2 = torch.eye(data.num_nodes, dtype=torch.float32).cuda()

    # mat_p = mat_p.to_dense()
    for i in range(args.walk_len):
        x_p = torch.spmm(mat_p, x_p)
        x_m1 = torch.spmm(mat_m1, x_m1)
        x_m2 = torch.spmm(mat_m2, x_m2)

        walk_profile.append(torch.diagonal(x_p)[data.node_mask].mean())
        walk_profile.append(x_p[data.node_mask, :][:, data.node_mask].mean())
        walk_profile.append(torch.diagonal(x_m1)[data.node_mask].mean())
        walk_profile.append(x_m1[data.node_mask, :][:, data.node_mask].mean())
        walk_profile.append(torch.diagonal(x_p).mean() - torch.diagonal(x_m1).mean())
        walk_profile.append(torch.diagonal(x_m2)[data.node_mask].mean())
        walk_profile.append(x_m2[data.node_mask, :][:, data.node_mask].mean())
        walk_profile.append(torch.diagonal(x_p).mean() - torch.diagonal(x_m2).mean())

    #walk_profile = torch.tensor(walk_profile).view(1, -1)
    walk_profile = torch.tensor(walk_profile, dtype=torch.float32).view(1, -1)
    return walk_profile




def prepare_data(args):
    data, train_edges, val_edges, test_edges, train_lb, val_lb, test_lb, train_pos_ids, val_pos_ids = load_splitted_data(
        args)

    # Construct train, val and test data loader.
    set_random_seed(args.seed)
    train_data = torch.tensor([])
    val_data = torch.tensor([])
    test_data = torch.tensor([])

    print(len(train_edges))
    train_pos_id = -1
    train_labels = []
    for ind, edge in enumerate(tqdm(train_edges)):
        if edge.shape[1]==1:
            continue
        else:
            if train_lb[ind] == 1:
                train_labels.append(1)
                train_pos_id = train_pos_ids.pop(0)
                data_a = ego_graph_minus(data, edge, train_pos_id, args)
            else:
                train_labels.append(0)
                data_a = ego_graph_plus(data, edge, args)
            walk_profile = obtain_walk_profile(data_a, args)
            train_data = torch.cat((train_data, walk_profile), dim=0)

    val_labels = []
    for ind, edge in enumerate(tqdm(val_edges)):
        if edge.shape[1]==1:
            continue
        else:
            if val_lb[ind] == 1:
                val_labels.append(1)
                val_pos_id = val_pos_ids.pop(0)
                data_a = ego_graph_minus(data, edge, val_pos_id, args)
            else:
                val_labels.append(0)
                data_a = ego_graph_plus(data, edge, args)
            walk_profile = obtain_walk_profile(data_a, args)
            val_data = torch.cat((val_data, walk_profile), dim=0)

    test_labels = []
    for ind, edge in enumerate(tqdm(test_edges)):
        if edge.shape[1] ==1:
            continue
        else:
            test_labels.append(test_lb[ind])
            data_a = ego_graph_plus(data, edge, args)
            walk_profile = obtain_walk_profile(data_a, args)
            test_data = torch.cat((test_data, walk_profile), dim=0)

    train_lb = torch.tensor(train_labels).to(torch.long)
    val_lb = torch.tensor(val_labels).to(torch.long)
    test_lb = torch.tensor(test_labels).to(torch.long)


    return train_data, val_data, test_data, train_lb, val_lb, test_lb




