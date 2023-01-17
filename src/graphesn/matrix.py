import random
from typing import Optional, Tuple, Union

import torch
from torch import Size, Tensor

__all__ = ['uniform', 'normal', 'ring', 'orthogonal', 'symmetric', 'antisymmetric', 'diagonal', 'binary', 'ones',
           'zeros', 'sparse_uniform', 'sparse_normal', 'sparse_binary', 'sparse_ones', 'rescale_']


def uniform(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
            scale: Optional[float] = None, range: Tuple[float, float] = (-1, 1)) -> Tensor:
    """
    Uniform random tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :param range: Range of uniform distribution (default [-1, +1])
    :return: A random tensor
    """
    W = torch.empty(size).uniform_(*range)
    rescale_(W, rho, sigma, scale)
    return W.data


def normal(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
           scale: Optional[float] = None) -> Tensor:
    """
    Normal random tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A random tensor
    """
    W = torch.empty(size).normal_(mean=0, std=1)
    rescale_(W, rho, sigma, scale)
    return W.data


def ring(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
         scale: Optional[float] = None) -> Tensor:
    """
    Ring matrix

    See:
    C. Gallicchio & A. Micheli (2020). Ring Reservoir Neural Networks for Graphs.
    In 2020 International Joint Conference on Neural Networks (IJCNN), IEEE.
    https://doi.org/10.1109/IJCNN48605.2020.9206723

    :param size: Size of tensor (must be square)
    :param rho: Spectral radius (equivalent to others)
    :param sigma: Spectral norm (equivalent to others)
    :param scale: Simple rescaling of the matrix (equivalent to others)
    :return: A re-scaled ring matrix
    """
    assert (len(size) == 2) and (size[0] == size[1])
    assert any(arg is not None for arg in [rho, sigma, scale])
    if scale is None:
        scale = rho if sigma is None else sigma
    W = torch.eye(size[0]).roll(1, 0) * scale
    return W.data


def orthogonal(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
               scale: Optional[float] = None) -> Tensor:
    """
    Orthogonal matrix

    See:
    F. Mezzadri (2007). How to Generate Random Matrices from the Classical Compact Groups.
    Notices of the American Mathematical Society, 54(5), pp. 592-604.
    https://www.ams.org/notices/200705/fea-mezzadri-web.pdf

    :param size: Size of tensor (if not square, generates a semi-orthogonal matrix)
    :param rho: Spectral radius (equivalent to others)
    :param sigma: Spectral norm (equivalent to others)
    :param scale: Simple rescaling of the matrix (equivalent to others)
    :return: A re-scaled orthogonal matrix
    """
    assert any(arg is not None for arg in [rho, sigma, scale])
    if scale is None:
        scale = rho if sigma is None else sigma
    W = torch.empty(size)
    torch.nn.init.orthogonal_(W, scale)
    return W.data


def symmetric(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
              scale: Optional[float] = None) -> Tensor:
    """
    Symmetric random tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A random tensor
    """
    W = torch.empty(size).uniform_(-1, 1)
    W = (W + W.t()) / 2
    rescale_(W, rho, sigma, scale)
    return W.data


def antisymmetric(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
                  scale: Optional[float] = None) -> Tensor:
    """
    Antisymmetric random tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A random tensor
    """
    W = torch.empty(size).uniform_(-1, 1)
    W = (W - W.t()) / 2
    rescale_(W, rho, sigma, scale)
    return W.data


def diagonal(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
             scale: Optional[float] = None, range: Tuple[float, float] = (-1, 1)) -> Tensor:
    """
    Diagonal uniform random tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :param range: Range of uniform distribution (default [-1, +1])
    :return: A random tensor
    """
    assert size[0] == size[1]
    W = torch.empty(size[0]).uniform_(*range).diag()
    rescale_(W, rho, sigma, scale)
    return W.data


def binary(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
           scale: Optional[float] = None) -> Tensor:
    """
    Binary random tensor (±1)

    Used as input matrix for Ring Reservoirs. See:
    C. Gallicchio & A. Micheli (2020). Ring Reservoir Neural Networks for Graphs.
    In 2020 International Joint Conference on Neural Networks (IJCNN), IEEE.
    https://doi.org/10.1109/IJCNN48605.2020.9206723

    :param size: Size of tensor
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A random tensor
    """
    W = torch.empty(size).uniform_(-1, 1).sign()
    rescale_(W, rho, sigma, scale)
    return W.data


def ones(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
         scale: Optional[float] = None) -> Tensor:
    """
    Ones tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A random tensor
    """
    W = torch.ones(size)
    rescale_(W, rho, sigma, scale)
    return W.data


def zeros(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
          scale: Optional[float] = None) -> Tensor:
    """
    Zeros tensor

    Rescaling is meaningless in this case.

    :param size: Size of tensor
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A random tensor
    """
    W = torch.zeros(size)
    return W.data


def sparse_uniform(size: Size, connections: Union[int, float],
                   rho: Optional[float] = None, sigma: Optional[float] = None,
                   scale: Optional[float] = None) -> Tensor:
    """
    Sparse uniform random tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param connections: Non-zero weights per row (absolute or ratio)
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A sparse random tensor
    """
    assert len(size) == 2
    if type(connections) is float:
        connections = int(connections * size[1])
    rows = sum(([i] * connections for i in range(size[0])), [])
    cols = sum((random.sample(range(size[1]), connections) for _ in range(size[0])), [])
    values = torch.empty(len(rows)).uniform_(-1, 1)
    W = torch.sparse_coo_tensor([rows, cols], values, size).coalesce()
    rescale_(W, rho, sigma, scale)
    return W


def sparse_normal(size: Size, connections: Union[int, float],
                  rho: Optional[float] = None, sigma: Optional[float] = None,
                  scale: Optional[float] = None) -> Tensor:
    """
    Sparse normal random tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param connections: Non-zero weights per row (absolute or ratio)
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A sparse random tensor
    """
    assert len(size) == 2
    if type(connections) is float:
        connections = int(connections * size[1])
    rows = sum(([i] * connections for i in range(size[0])), [])
    cols = sum((random.sample(range(size[1]), connections) for _ in range(size[0])), [])
    values = torch.empty(len(rows)).normal_(mean=0, std=1)
    W = torch.sparse_coo_tensor([rows, cols], values, size).coalesce()
    rescale_(W, rho, sigma, scale)
    return W


def sparse_binary(size: Size, connections: Union[int, float],
                  rho: Optional[float] = None, sigma: Optional[float] = None,
                  scale: Optional[float] = None) -> Tensor:
    """
    Sparse binary random tensor (±1)

    Used as input matrix for Ring Reservoirs. See:
    C. Gallicchio & A. Micheli (2020). Ring Reservoir Neural Networks for Graphs.
    In 2020 International Joint Conference on Neural Networks (IJCNN), IEEE.
    https://doi.org/10.1109/IJCNN48605.2020.9206723

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param connections: Non-zero weights per row (absolute or ratio)
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A sparse random tensor
    """
    assert len(size) == 2
    if type(connections) is float:
        connections = int(connections * size[1])
    rows = sum(([i] * connections for i in range(size[0])), [])
    cols = sum((random.sample(range(size[1]), connections) for _ in range(size[0])), [])
    values = torch.empty(len(rows)).uniform_(-1, 1).sign()
    W = torch.sparse_coo_tensor([rows, cols], values, size).coalesce()
    rescale_(W, rho, sigma, scale)
    return W


def sparse_ones(size: Size, connections: Union[int, float],
                rho: Optional[float] = None, sigma: Optional[float] = None,
                scale: Optional[float] = None) -> Tensor:
    """
    Sparse ones random tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param connections: Non-zero weights per row (absolute or ratio)
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A sparse random tensor
    """
    assert len(size) == 2
    if type(connections) is float:
        connections = int(connections * size[1])
    rows = sum(([i] * connections for i in range(size[0])), [])
    cols = sum((random.sample(range(size[1]), connections) for _ in range(size[0])), [])
    values = torch.ones(len(rows))
    W = torch.sparse_coo_tensor([rows, cols], values, size).coalesce()
    rescale_(W, rho, sigma, scale)
    return W


def rescale_(W: Tensor, rho: Optional[float] = None, sigma: Optional[float] = None,
             scale: Optional[float] = None) -> Tensor:
    """
    Rescale a matrix in-place

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param W: Matrix to rescale
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: Rescaled matrix
    """
    if rho is not None:
        return W.div_(torch.linalg.eigvals(W.to_dense()).abs().max()).mul_(rho)
    elif sigma is not None:
        return W.div_(torch.linalg.matrix_norm(W.to_dense(), ord=2)).mul_(sigma)
    elif scale is not None:
        return W.mul_(scale)
