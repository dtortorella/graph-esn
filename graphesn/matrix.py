from typing import Optional

import torch
from torch import Size, Tensor

__all__ = ['uniform', 'normal', 'ring', 'rescale_']


def uniform(size: Size, rho: Optional[float] = None, sigma: Optional[float] = None,
            scale: Optional[float] = None) -> Tensor:
    """
    Uniform random tensor

    Can either be rescaled according to spectral radius `rho`, spectral norm `sigma`, or `scale`.

    :param size: Size of tensor
    :param rho: Spectral radius
    :param sigma: Spectral norm
    :param scale: Simple rescaling of the standard random matrix
    :return: A random tensor
    """
    W = torch.empty(size).uniform_(-1, 1)
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
        return W.div_(torch.linalg.eigvals(W).abs().max()).mul_(rho)
    elif sigma is not None:
        return W.div_(torch.linalg.matrix_norm(W, ord=2)).mul_(sigma)
    elif scale is not None:
        return W.mul_(scale)
