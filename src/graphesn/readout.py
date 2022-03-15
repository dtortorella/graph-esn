import sys
from typing import Tuple, Iterator, Optional, List, Callable, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

__all__ = ['Readout', 'fit_readout', 'fit_and_validate_readout']


class Readout(Module):
    """
    A linear readout

    Linear model with bias :math:`y = W x + b`, like Linear in Torch.
    """
    weight: Parameter  # (targets × features)
    bias: Parameter  # (targets)

    def __init__(self, num_features: int, num_targets: int):
        """
        New readout

        :param num_features: Number of input features
        :param num_targets: Number of output targets
        """
        super().__init__()
        self.weight = Parameter(Tensor(num_targets, num_features), requires_grad=False)
        self.bias = Parameter(Tensor(num_targets), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def fit(self, data: Union[Iterator[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]],
            regularization: Union[Optional[float], List[float]] = None,
            validate: Optional[Callable[[Tuple[Tensor, Tensor]], float]] = None,
            verbose: bool = False):
        """
        Fit readout to data

        :param data: Dataset of (features, targets) tuples, or single pair
        :param regularization: Ridge regression lambda, or lambda if validation requested
        :param validate: Validation function, if regularization is to be selected
        :param verbose: Whether to print validation info (default false)
        """
        if not hasattr(data, '__next__'):
            data = iter([data])
        if callable(validate):
            self.weight.data, self.bias.data = fit_and_validate_readout(data, regularization, validate, verbose)
        else:
            self.weight.data, self.bias.data = fit_readout(data, regularization)

    @property
    def num_features(self) -> int:
        """
        Input features

        :return: Number of input features
        """
        return self.weight.shape[1]

    @property
    def num_targets(self) -> int:
        """
        Output targets

        :return: Number of output targets
        """
        return self.weight.shape[0]

    def __repr__(self):
        return f'Readout(features={self.num_features}, targets={self.num_targets})'


def fit_readout(data: Iterator[Tuple[Tensor, Tensor]], regularization: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    """
    Ridge regression for big data

    Fits a linear model :math:`y = W x + b` with regularization.
    See:
    T. Zhang & B. Yang (2017). An exact approach to ridge regression for big data.
    Computational Statistics, 32(3), 909–928. https://doi.org/10.1007/s00180-017-0731-5

    :param data: Batch dataset of pairs (x, y) with samples on rows
    :param regularization: Regularization constant for ridge regression (default null)
    :return: A pair of tensors (W, b)
    """
    # Compute sufficient statistics for regression
    x, y = next(data)
    Syy = y.square().sum(dim=0)  # (targets)
    Sxy = x.t() @ y  # (features × targets)
    Sxx = x.t() @ x  # (features × features)
    Sy = y.sum(dim=0)  # (targets)
    Sx = x.sum(dim=0)  # (features)
    n = float(x.shape[0])  # samples
    for x, y in data:
        Syy += y.square().sum(dim=0)
        Sxy += x.t() @ y
        Sxx += x.t() @ x
        Sy += y.sum(dim=0)
        Sx += x.sum(dim=0)
        n += x.shape[0]
    # Compute ridge matrices
    Vxx = Sxx.diag() - (Sx.square() / n)
    Vyy = Syy - (Sy.square() / n)
    XX = (Sxx - torch.outer(Sx, Sx) / n) / torch.outer(Vxx, Vxx).sqrt()
    Xy = (Sxy - torch.outer(Sx, Sy) / n) / torch.outer(Vxx, Vyy).sqrt()
    if regularization:
        XX += torch.eye(n=XX.shape[0]).to(XX) * regularization
    # Compute weights
    Ws = torch.linalg.solve(XX, Xy)
    W = Ws * torch.sqrt(Vyy.expand_as(Ws) / Vxx.unsqueeze(-1))
    b = (Sy / n) - (Sx / n) @ W
    return W.t(), b


def fit_and_validate_readout(data: Iterator[Tuple[Tensor, Tensor]], regularization_constants: List[float],
                             get_validation_error: Callable[[Tuple[Tensor, Tensor]], float],
                             verbose: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Ridge regression for big data, with efficient regularization selection

    Fits a linear model :math:`y = W x + b` with regularization.
    See:
    T. Zhang & B. Yang (2017). An exact approach to ridge regression for big data.
    Computational Statistics, 32(3), 909–928. https://doi.org/10.1007/s00180-017-0731-5

    :param data: Batch dataset of pairs (x, y) with samples on rows
    :param regularization_constants: Regularization constants for ridge regression (including none)
    :param get_validation_error: Evaluate validation error for a regression pair (W, b)
    :param verbose: Whether to print validation info (default false)
    :return: A pair of tensors (W, b)
    """
    # Compute sufficient statistics for regression
    x, y = next(data)
    Syy = y.square().sum(dim=0)  # (targets)
    Sxy = x.t() @ y  # (features × targets)
    Sxx = x.t() @ x  # (features × features)
    Sy = y.sum(dim=0)  # (targets)
    Sx = x.sum(dim=0)  # (features)
    n = float(x.shape[0])  # samples
    for x, y in data:
        Syy += y.square().sum(dim=0)
        Sxy += x.t() @ y
        Sxx += x.t() @ x
        Sy += y.sum(dim=0)
        Sx += x.sum(dim=0)
        n += x.shape[0]
    # Compute ridge matrices
    Vxx = Sxx.diag() - (Sx.square() / n)
    Vyy = Syy - (Sy.square() / n)
    XX = (Sxx - torch.outer(Sx, Sx) / n) / torch.outer(Vxx, Vxx).sqrt()
    Xy = (Sxy - torch.outer(Sx, Sy) / n) / torch.outer(Vxx, Vyy).sqrt()
    # Compute and select weights
    best_validation_error, best_W, best_b = None, None, None
    for regularization in regularization_constants:
        # Compute weights
        XXr = (XX + torch.eye(n=XX.shape[0]).to(XX) * regularization) if regularization else XX
        Ws = torch.linalg.solve(XXr, Xy)
        W = Ws * torch.sqrt(Vyy.expand_as(Ws) / Vxx.unsqueeze(-1))
        b = (Sy / n) - (Sx / n) @ W
        # Validate, select
        validation_error = get_validation_error((W.t(), b))
        if best_validation_error is None or validation_error < best_validation_error:
            best_validation_error, best_W, best_b = validation_error, W.t(), b
        if verbose:
            print(f'{regularization:e}: {validation_error}', file=sys.stderr)
    return best_W, best_b
