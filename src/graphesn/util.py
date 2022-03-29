import statistics
from typing import Union, List, Optional

import torch.linalg
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor

from graphesn import DynamicData

__all__ = ['graph_spectral_norm', 'approximate_graph_spectral_radius',
           'compute_dynamic_graph_alpha', 'compute_dynamic_weighted_graph_alpha',
           'distance_to_proximity', 'to_sparse_adjacency']


def graph_spectral_norm(edge_index: Adj, edge_weight: OptTensor = None):
    """
    Spectral norm of a graph

    :param edge_index: Graph adjacency matrix
    :param edge_weight: Edges weight (optional)
    :return: Spectral norm
    """
    return float(torch.linalg.matrix_norm(to_dense_adj(edge_index, edge_attr=edge_weight), ord=2))


def approximate_graph_spectral_radius(adj: SparseTensor, max_iterations: int = 1000, threshold: float = 1e-6):
    """
    Spectral radius of a graph via power method iteration

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
    alphas = [graph_spectral_norm(edge_index) for edge_index in data_list if
              not ignore_disconnected or edge_index.shape[1] > 0]
    return statistics.geometric_mean(alphas)


def compute_dynamic_weighted_graph_alpha(data: DynamicData, ignore_disconnected: bool = True):
    """
    Geometric mean of the spectral norms for a dynamic weighted graph

    :param data: A dynamic graph or a list of adjacency matrices
    :param ignore_disconnected: Whether to ignore timesteps without edges (default true)
    :return: Geometric mean of alphas
    """
    alphas = [graph_spectral_norm(data[t].edge_index, data[t].edge_weight) for t in range(data.num_timesteps) if
              not ignore_disconnected or data[t].edge_index.shape[1] > 0]
    return statistics.geometric_mean(alphas)


def distance_to_proximity(edge_weight: Union[Tensor, List[Tensor]]):
    """
    Convert edge weights from distances to proximities via max-min normalization

    :param edge_weight: Edge weights (single tensor or temporal list of tensors)
    :return: Proximity edge weights
    """
    if type(edge_weight) is list:
        return [distance_to_proximity(edge_weight_t) for edge_weight_t in edge_weight]
    else:
        max_weight, min_weight = edge_weight.max(), edge_weight.min()
        return 1 - (edge_weight - min_weight) / (max_weight - min_weight)


def to_sparse_adjacency(edge_index: Adj, edge_weight: OptTensor = None,
                        num_nodes: Optional[int] = None) -> SparseTensor:
    """
    Convert edge index and weights to sparse adjacency tensor

    :param edge_index: Edge index
    :param edge_weight: Edges weight (optional)
    :param num_nodes: Number of nodes (optional, but *strongly* advised)
    :return: Sparse adjacency tensor
    """
    return SparseTensor.from_edge_index(edge_index, edge_weight,
                                        (num_nodes, num_nodes) if num_nodes is not None else None)
