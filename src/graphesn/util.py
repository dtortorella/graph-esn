import statistics
from typing import Union, List

import torch.linalg
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import to_dense_adj

from graphesn import DynamicData

__all__ = ['compute_graph_alpha', 'compute_graph_alpha', 'compute_dynamic_weighted_graph_alpha']


def compute_graph_alpha(edge_index: Adj, edge_weight: OptTensor = None):
    """
    Spectral norm of a graph

    :param edge_index: Graph adjacency matrix
    :param edge_weight: Edges weight (optional)
    :return: Spectral norm
    """
    return float(torch.linalg.matrix_norm(to_dense_adj(edge_index, edge_attr=edge_weight), ord=2))


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
