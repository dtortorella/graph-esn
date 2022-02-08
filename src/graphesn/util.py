import statistics
from typing import Union, List

import torch.linalg
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor

from graphesn import DynamicData

__all__ = ['compute_graph_alpha', 'approximate_graph_alpha',
           'compute_dynamic_graph_alpha', 'compute_dynamic_weighted_graph_alpha']


def compute_graph_alpha(edge_index: Adj, edge_weight: OptTensor = None):
    """
    Spectral norm of a graph

    :param edge_index: Graph adjacency matrix
    :param edge_weight: Edges weight (optional)
    :return: Spectral norm
    """
    return float(torch.linalg.matrix_norm(to_dense_adj(edge_index, edge_attr=edge_weight), ord=2))


def approximate_graph_alpha(adj: SparseTensor, max_iterations: int = 1000, threshold: float = 1e-6):
    """
    Spectral norm of a graph via power method iteration

    :param adj: Sparse adjacency matrix
    :param max_iterations: Maximum number of power iterations
    :param threshold: Convergence threshold for dominant eigenvector
    :return: Spectral radius
    """
    u = torch.rand(adj.size(0), 1).to(adj.device())
    u /= torch.linalg.vector_norm(u)
    for _ in range(max_iterations):
        v = adj.matmul(u)
        v /= torch.linalg.vector_norm(v)
        if torch.linalg.vector_norm(v - u) < threshold:
            break
        else:
            u = v
    alpha = (adj.matmul(v).t() @ v) / (v.t() @ v)
    return alpha.item()


def compute_dynamic_graph_alpha(data_list: Union[DynamicData, List[Adj]], ignore_disconnected: bool = True):
    """
    Geometric mean of the spectral norms for a dynamic graph

    :param data_list: A dynamic graph or a list of adjacency matrices
    :param ignore_disconnected: Whether to ignore timesteps without edges (default true)
    :return: Geometric mean of alphas
    """
    if isinstance(data_list, DynamicData):
        data_list = [data_list[t].edge_index for t in range(data_list.num_timesteps)]
    alphas = [compute_graph_alpha(edge_index) for edge_index in data_list if
              not ignore_disconnected or edge_index.shape[1] > 0]
    return statistics.geometric_mean(alphas)


def compute_dynamic_weighted_graph_alpha(data: DynamicData, ignore_disconnected: bool = True):
    """
    Geometric mean of the spectral norms for a dynamic weighted graph

    :param data: A dynamic graph or a list of adjacency matrices
    :param ignore_disconnected: Whether to ignore timesteps without edges (default true)
    :return: Geometric mean of alphas
    """
    alphas = [compute_graph_alpha(data[t].edge_index, data[t].edge_weight) for t in range(data.num_timesteps) if
              not ignore_disconnected or data[t].edge_index.shape[1] > 0]
    return statistics.geometric_mean(alphas)
