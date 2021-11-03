from typing import Optional, Callable, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn import Module, Parameter, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_sparse import matmul, SparseTensor

from graphesn import matrix

__all__ = ['initializer', 'ReservoirConvLayer', 'GraphReservoir', 'StaticGraphReservoir', 'TemporalGraphReservoir',
           'DynamicGraphReservoir']


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

    @property
    def in_features(self) -> int:
        """Input dimension"""
        return self.input_weight.shape[1]

    @property
    def out_features(self) -> int:
        """Reservoir state dimension"""
        return self.input_weight.shape[0]


class GraphReservoir(Module):
    """
    Base class for graph reservoirs

    :param num_layers: Reservoir layers
    :param in_features: Size of input
    :param hidden_features: Size of reservoir (i.e. number of hidden units per layer)
    :param bias: Whether bias term is present
    :param kwargs: Other `ReservoirConvLayer` arguments (activation, etc.)
    """
    layers: ModuleList

    def __init__(self, num_layers: int, in_features: int, hidden_features: int, bias: bool = False, **kwargs):
        super().__init__()
        assert num_layers > 0
        self.layers = ModuleList()
        self.layers.append(
            ReservoirConvLayer(input_features=in_features, hidden_features=hidden_features, bias=bias, **kwargs))
        for _ in range(1, num_layers):
            self.layers.append(
                ReservoirConvLayer(input_features=hidden_features, hidden_features=hidden_features, bias=bias,
                                   **kwargs))

    def initialize_parameters(self, recurrent: Callable[[Size], Tensor], input: Callable[[Size], Tensor],
                              bias: Optional[Callable[[Size], Tensor]] = None, leakage: float = 1.0):
        """
        Initialize reservoir weights for all layers

        :param recurrent: Random matrix generator for recurrent weight
        :param input: Random matrix generator for input weight
        :param bias: Random matrix generator for bias, if present
        :param leakage: Leakage constant
        """
        for layer in self.layers:
            layer.initialize_parameters(recurrent=recurrent, input=input, bias=bias, leakage=leakage)

    @property
    def num_layers(self) -> int:
        """Number of reservoir layers"""
        return len(self.layers)

    @property
    def in_features(self) -> int:
        """Input dimension"""
        return self.layers[0].input_weight.shape[1]

    @property
    def out_features(self) -> int:
        """Embedding dimension"""
        raise NotImplementedError()


class StaticGraphReservoir(GraphReservoir):
    """
    Reservoir for static graphs

    :param num_layers: Reservoir layers
    :param in_features: Size of input
    :param hidden_features: Size of reservoir (i.e. number of hidden units per layer)
    :param bias: Whether bias term is present
    :param pooling: Graph pooling function (optional, default no pooling)
    :param fully: Whether to concatenate all layers' encodings, or use just final layer encoding
    :param max_iterations: Maximum number of iterations (optional, default infinity)
    :param epsilon: Convergence condition (default 1e-6)
    """
    pooling: Optional[Callable[[Tensor, Tensor], Tensor]]
    fully: bool
    max_iterations: Optional[int]
    epsilon: float

    def __init__(self, num_layers: int, in_features: int, hidden_features: int, bias: bool = False,
                 pooling: Optional[Callable[[Tensor, Tensor], Tensor]] = None, fully: bool = False,
                 max_iterations: Optional[int] = None, epsilon: float = 1e-6, **kwargs):
        super().__init__(num_layers, in_features, hidden_features, bias, **kwargs)
        self.pooling = pooling
        self.fully = fully
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def forward(self, edge_index: Adj, input: Tensor, initial_state: Optional[Union[List[Tensor], Tensor]] = None,
                batch: Optional[Tensor] = None) -> Tensor:
        """
        Encode input

        :param edge_index: Adjacency
        :param input: Input graph signal (nodes × in_features)
        :param initial_state: Initial state (nodes × hidden_features) for all reservoir layers, default zeros
        :param batch: Batch index (optional)
        :return: Encoding (samples × dim)
        """
        if initial_state is None:
            initial_state = [torch.zeros(input.shape[0], layer.out_features).to(input) for layer in self.layers]
        elif len(initial_state) != self.num_layers and initial_state.dim() == 2:
            initial_state = [initial_state] * self.num_layers
        embeddings = []
        embeddings.append(self._embed(edge_index, input, initial_state[0], self.layers[0]))
        for i in range(1, self.num_layers):
            embeddings.append(self._embed(edge_index, embeddings[-1], initial_state[i], self.layers[i]))
        if self.fully:
            return torch.cat([self.pooling(x, batch) if self.pooling else x for x in embeddings], dim=-1)
        else:
            return self.pooling(embeddings[-1], batch) if self.pooling else embeddings[-1]

    def _embed(self, edge_index: Adj, input: Tensor, initial_state: Tensor, layer: ReservoirConvLayer) -> Tensor:
        """
        Compute node embeddings for a single layer

        :param edge_index: Adjacency
        :param input: Input graph signal (nodes × in_features)
        :param initial_state: Initial state (nodes × hidden_features) for all reservoir layers, default zeros
        :param layer: Reservoir layer
        :return: Encoding (nodes × dim)
        """
        iterations = 0
        old_state = initial_state
        while True:
            state = layer(edge_index, input, old_state)
            if self.max_iterations and iterations >= self.max_iterations:
                break
            if torch.norm(old_state - state) < self.epsilon:
                break
            old_state = state
            iterations += 1
        return state

    @property
    def out_features(self) -> int:
        """Embedding dimension"""
        return sum(layer.out_features for layer in self.layers) if self.fully else self.layers[-1].out_features


class TemporalGraphReservoir(GraphReservoir):
    """
    Reservoir for temporal graphs

    :param num_layers: Reservoir layers
    :param in_features: Size of input
    :param hidden_features: Size of reservoir (i.e. number of hidden units per layer)
    :param bias: Whether bias term is present
    :param pooling: Graph pooling function (optional, default no pooling)
    :param fully: Whether to concatenate all layers' encodings, or use just final layer encoding
    """
    pooling: Optional[Callable[[Tensor, Tensor], Tensor]]
    fully: bool

    def __init__(self, num_layers: int, in_features: int, hidden_features: int, bias: bool = False,
                 pooling: Optional[Callable[[Tensor, Tensor], Tensor]] = None, fully: bool = False, **kwargs):
        super().__init__(num_layers, in_features, hidden_features, bias, **kwargs)
        self.pooling = pooling
        self.fully = fully

    def forward(self, edge_index: Adj, input: Tensor, initial_state: Optional[Union[List[Tensor], Tensor]] = None,
                batch: Optional[Tensor] = None) -> Tensor:
        """
        Encode input

        :param edge_index: Adjacency
        :param input: Input graph signal (time × nodes × in_features)
        :param initial_state: Initial state (nodes × hidden_features) for all reservoir layers, default zeros
        :param batch: Batch index (optional)
        :return: Encoding (samples × dim)
        """
        if initial_state is None:
            state = [torch.zeros(input.shape[1], layer.out_features).to(input) for layer in self.layers]
        elif len(initial_state) != self.num_layers and initial_state.dim() == 2:
            state = [initial_state] * self.num_layers
        else:
            state = initial_state
        for t in range(input.shape[0]):
            state[0] = self.layers[0](edge_index, input[t], state[0])
            for i in range(1, self.num_layers):
                state[i] = self.layers[i](edge_index, state[i - 1], state[i])
        if self.fully:
            return torch.cat([self.pooling(x, batch) if self.pooling else x for x in state], dim=-1)
        else:
            return self.pooling(state[-1], batch) if self.pooling else state[-1]

    @property
    def out_features(self) -> int:
        """Embedding dimension"""
        return sum(layer.out_features for layer in self.layers) if self.fully else self.layers[-1].out_features


class DynamicGraphReservoir(GraphReservoir):
    """
    Reservoir for discrete-time dynamic temporal graphs

    :param num_layers: Reservoir layers
    :param in_features: Size of input
    :param hidden_features: Size of reservoir (i.e. number of hidden units per layer)
    :param bias: Whether bias term is present
    :param pooling: Graph pooling function (optional, default no pooling)
    :param fully: Whether to concatenate all layers' encodings, or use just final layer encoding
    """
    pooling: Optional[Callable[[Tensor, Tensor], Tensor]]
    fully: bool

    def __init__(self, num_layers: int, in_features: int, hidden_features: int, bias: bool = False,
                 pooling: Optional[Callable[[Tensor, Tensor], Tensor]] = None, fully: bool = False, **kwargs):
        super().__init__(num_layers, in_features, hidden_features, bias, **kwargs)
        self.pooling = pooling
        self.fully = fully

    def forward(self, edge_index: List[Adj], input: Tensor, initial_state: Optional[Union[List[Tensor], Tensor]] = None,
                batch: Optional[Tensor] = None) -> Tensor:
        """
        Encode input

        :param edge_index: Sequence of adjacency matrices (time × Adj)
        :param input: Input graph signal (time × nodes × in_features)
        :param initial_state: Initial state (nodes × hidden_features) for all reservoir layers, default zeros
        :param batch: Batch index (optional)
        :return: Encoding (samples × dim)
        """
        if initial_state is None:
            state = [torch.zeros(input.shape[1], layer.out_features).to(input) for layer in self.layers]
        elif len(initial_state) != self.num_layers and initial_state.dim() == 2:
            state = [initial_state] * self.num_layers
        else:
            state = initial_state
        for t in range(input.shape[0]):
            state[0] = self.layers[0](edge_index[t], input[t], state[0])
            for i in range(1, self.num_layers):
                state[i] = self.layers[i](edge_index[t], state[i - 1], state[i])
        if self.fully:
            return torch.cat([self.pooling(x, batch) if self.pooling else x for x in state], dim=-1)
        else:
            return self.pooling(state[-1], batch) if self.pooling else state[-1]

    @property
    def out_features(self) -> int:
        """Embedding dimension"""
        return sum(layer.out_features for layer in self.layers) if self.fully else self.layers[-1].out_features
