from typing import Optional, Callable, Union

import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_sparse import matmul, SparseTensor

from graphesn import matrix


def initializer(name: str, **options) -> Callable[[Size], Tensor]:
    """
    Gets a random weight initializer

    :param name: Name of the random matrix generator in `graphesn.matrix`
    :param options: Random matrix generator options
    :return: A random weight generator function
    """
    init = getattr(matrix, name)
    return lambda size: init(size, **options)


class ReservoirConvLayer(MessagePassing):
    """
    A Graph ESN convolution layer

    :param input_features: Input dimension
    :param hidden_features: Reservoir dimension
    :param bias: If bias term is present
    :param activation: Activation function (default tanh)
    """
    input_weight: Parameter
    recurrent_weight: Parameter
    leakage: Parameter
    bias: Optional[Parameter]
    activation: Callable[[Tensor], Tensor]

    def __init__(self, input_features: int, hidden_features: int, bias: bool = False,
                 activation: Union[str, Callable[[Tensor], Tensor]] = 'tanh', **kwargs):
        super().__init__(**kwargs)
        self.input_weight = Parameter(torch.empty(hidden_features, input_features), requires_grad=False)
        self.recurrent_weight = Parameter(torch.empty(hidden_features, hidden_features), requires_grad=False)
        self.leakage = Parameter(torch.ones([]), requires_grad=False)
        self.bias = Parameter(torch.empty(hidden_features), requires_grad=False) if bias else None
        self.activation = activation if callable(activation) else getattr(torch, activation)

    def forward(self, edge_index: Adj, input: Tensor, state: Tensor):
        neighbour_aggr = self.propagate(edge_index=edge_index, x=state)
        return self.leakage * self.activation(
            F.linear(input, self.input_weight, self.bias) + F.linear(neighbour_aggr, self.recurrent_weight)) \
               + (1 - self.leakage) * state

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, self.aggr)

    def initialize_parameters(self, recurrent: Callable[[Size], Tensor], input: Callable[[Size], Tensor],
                              bias: Optional[Callable[[Size], Tensor]] = None, leakage: float = 1.0):
        """
        Initialize reservoir weights

        :param recurrent: Random matrix generator for recurrent weight
        :param input: Random matrix generator for input weight
        :param bias: Random matrix generator for bias, if present
        :param leakage: Leakage constant
        """
        self.recurrent_weight.data = recurrent(self.recurrent_weight.shape)
        self.input_weight.data = input(self.input_weight.shape)
        if self.bias is not None:
            self.bias.data = bias(self.bias.shape)
        self.leakage.data = torch.tensor(leakage)
