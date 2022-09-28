from typing import Optional, Callable, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn import Module, Parameter, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import matmul, SparseTensor

from graphesn import matrix

__all__ = ['initializer', 'ReservoirConvLayer', 'GraphReservoir', 'StaticGraphReservoir', 'DynamicGraphReservoir']


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

    def forward(self, edge_index: Adj, input: Tensor, state: Tensor, edge_weight: OptTensor = None):
        neighbour_aggr = self.propagate(edge_index=edge_index, x=state, edge_weight=edge_weight)
        return self.leakage * self.activation(
            F.linear(input, self.input_weight, self.bias) + F.linear(neighbour_aggr, self.recurrent_weight)) \
               + (1 - self.leakage) * state

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

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
        return self.recurrent_weight.shape[0]

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, bias={self.bias is not None}'


class ReservoirEmbConvLayer(ReservoirConvLayer):
    """
    A Graph ESN convolution layer for categorical input features

    :param input_features: Input categories
    :param hidden_features: Reservoir dimension
    :param bias: If bias term is present
    :param activation: Activation function (default tanh)
    """

    def __init__(self, input_features: int, hidden_features: int, bias: bool = False,
                 activation: Union[str, Callable[[Tensor], Tensor]] = 'tanh', **kwargs):
        super().__init__(input_features, hidden_features, bias, activation, **kwargs)
        self.input_weight = Parameter(torch.empty(input_features, hidden_features), requires_grad=False)

    def forward(self, edge_index: Adj, input: Tensor, state: Tensor, edge_weight: OptTensor = None):
        neighbour_aggr = self.propagate(edge_index=edge_index, x=state, edge_weight=edge_weight)
        return self.leakage * self.activation(
            F.embedding(input, self.input_weight) + F.linear(neighbour_aggr, self.recurrent_weight, self.bias)) \
               + (1 - self.leakage) * state

    @property
    def in_features(self) -> int:
        """Input categories"""
        return self.input_weight.shape[0]


class GraphReservoir(Module):
    """
    Base class for graph reservoirs

    :param num_layers: Reservoir layers
    :param in_features: Size of input
    :param hidden_features: Size of reservoir (i.e. number of hidden units per layer)
    :param bias: Whether bias term is present
    :param categorical_input: Whether input features are categorical
    :param kwargs: Other `ReservoirConvLayer` arguments (activation, etc.)
    """
    layers: ModuleList

    def __init__(self, num_layers: int, in_features: int, hidden_features: int, bias: bool = False,
                 categorical_input: bool = False, **kwargs):
        super().__init__()
        assert num_layers > 0
        self.layers = ModuleList()
        if categorical_input:
            self.layers.append(
                ReservoirEmbConvLayer(input_features=in_features, hidden_features=hidden_features, bias=bias, **kwargs))
        else:
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
    :param categorical_input: Whether input features are categorical
    """
    pooling: Optional[Callable[[Tensor, Tensor], Tensor]]
    fully: bool
    max_iterations: Optional[int]
    epsilon: float

    def __init__(self, num_layers: int, in_features: int, hidden_features: int, bias: bool = False,
                 pooling: Optional[Callable[[Tensor, Tensor], Tensor]] = None, fully: bool = False,
                 max_iterations: Optional[int] = None, epsilon: float = 1e-6, categorical_input: bool = False,
                 **kwargs):
        super().__init__(num_layers, in_features, hidden_features, bias, categorical_input, **kwargs)
        self.pooling = pooling
        self.fully = fully
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def forward(self, edge_index: Adj, input: Tensor, initial_state: Optional[Union[List[Tensor], Tensor]] = None,
                batch: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        """
        Encode input

        :param edge_index: Adjacency
        :param edge_weight: Edges weight (optional)
        :param input: Input graph signal (nodes × in_features)
        :param initial_state: Initial state (nodes × hidden_features) for all reservoir layers, default zeros
        :param batch: Batch index (optional)
        :return: Encoding (samples × dim)
        """
        if initial_state is None:
            initial_state = [torch.zeros(input.shape[0], layer.out_features).to(layer.recurrent_weight) for layer in
                             self.layers]
        elif len(initial_state) != self.num_layers and initial_state.dim() == 2:
            initial_state = [initial_state] * self.num_layers
        embeddings = [self._embed(self.layers[0], edge_index, edge_weight, input, initial_state[0])]
        for i in range(1, self.num_layers):
            embeddings.append(self._embed(self.layers[i], edge_index, edge_weight, embeddings[-1], initial_state[i]))
        if self.fully:
            return torch.cat([self.pooling(x, batch) if self.pooling else x for x in embeddings], dim=-1)
        else:
            return self.pooling(embeddings[-1], batch) if self.pooling else embeddings[-1]

    def _embed(self, layer: ReservoirConvLayer, edge_index: Adj, edge_weight: OptTensor, input: Tensor,
               initial_state: Tensor) -> Tensor:
        """
        Compute node embeddings for a single layer

        :param layer: Reservoir layer
        :param edge_index: Adjacency
        :param edge_weight: Edges weight (optional)
        :param input: Input graph signal (nodes × in_features)
        :param initial_state: Initial state (nodes × hidden_features) for all reservoir layers, default zeros
        :return: Encoding (nodes × dim)
        """
        iterations = 0
        old_state = initial_state
        while True:
            state = layer(edge_index, input, old_state, edge_weight)
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


class DynamicGraphReservoir(GraphReservoir):
    """
    Reservoir for discrete-time dynamic temporal graphs

    :param num_layers: Reservoir layers
    :param in_features: Size of input
    :param hidden_features: Size of reservoir (i.e. number of hidden units per layer)
    :param bias: Whether bias term is present
    :param pooling: Graph pooling function (optional, default no pooling)
    :param fully: Whether to concatenate all layers' encodings, or use just final layer encoding
    :param return_sequences: Return the sequence of states instead of just the final states (default false)
    :param categorical_input: Whether input features are categorical
    """
    pooling: Optional[Callable[[Tensor, Tensor], Tensor]]
    fully: bool

    def __init__(self, num_layers: int, in_features: int, hidden_features: int, bias: bool = False,
                 pooling: Optional[Callable[[Tensor, Tensor], Tensor]] = None, fully: bool = False,
                 return_sequences: bool = False, categorical_input: bool = False, **kwargs):
        super().__init__(num_layers, in_features, hidden_features, bias, categorical_input, **kwargs)
        self.pooling = pooling
        self.fully = fully
        self.return_sequences = return_sequences

    def forward(self, edge_index: Union[List[Adj], Adj], input: Union[Tensor, List[Tensor]],
                initial_state: Optional[Union[List[Tensor], Tensor]] = None,
                batch: OptTensor = None, edge_weight: Optional[Union[List[Tensor], Tensor]] = None,
                mask: Optional[List[Tensor]] = None) -> Tensor:
        """
        Encode input

        :param edge_index: Sequence of adjacency matrices (time × Adj), or Adj in the case of the spatio-temporal setting
        :param edge_weight: Optional sequence of edge weights, or fixed edge weights for the spatio-temporal setting
        :param input: Input graph signal (time × nodes × in_features)
        :param initial_state: Initial state (nodes × hidden_features) for all reservoir layers, default zeros
        :param batch: Batch index (optional)
        :param mask: Sequence of node masks (optional, useful for padding dynamic graphs with different lengths)
        :return: Encoding (samples × dim)
        """
        if initial_state is None:
            num_nodes = input[0].shape[0] if isinstance(input, list) else input.shape[1]
            state = [torch.zeros(num_nodes, layer.out_features).to(layer.recurrent_weight) for layer in self.layers]
        elif len(initial_state) != self.num_layers and initial_state.dim() == 2:
            state = [initial_state.clone() for _ in range(self.num_layers)]
        else:
            state = [x.clone() for x in initial_state]
        if self.return_sequences:
            return self._embed_sequence(edge_index, edge_weight, input, state, mask, batch)
        else:
            return self._embed_final(edge_index, edge_weight, input, state, mask, batch)

    def _embed_final(self, edge_index: Union[List[Adj], Adj], edge_weight: Optional[Union[List[Tensor], Tensor]],
                     input: Union[Tensor, List[Tensor]], state: List[Tensor], mask: Optional[List[Tensor]],
                     batch: OptTensor) -> Tensor:
        """
        Compute final-state embedding

        :param edge_index: Sequence of adjacency matrices (time × Adj), or Adj in the case of the spatio-temporal setting
        :param edge_weight: Optional sequence of edge weights, or fixed edge weights for the spatio-temporal setting
        :param input: Input graph signal (time × nodes × in_features)
        :param state: Initial state (nodes × hidden_features) for all reservoir layers
        :param mask: Sequence of node masks (optional, useful for padding dynamic graphs with different lengths)
        :param batch: Batch index (optional)
        :return: Encoding (samples × dim)
        """
        for t in range(len(input)):
            edge_index_t = edge_index[t] if isinstance(edge_index, list) else edge_index
            edge_weight_t = edge_weight[t] if isinstance(edge_weight, list) else edge_weight
            mask_t = slice(None) if mask is None else mask[t]
            state[0][mask_t] = self.layers[0](edge_index_t, input[t], state[0], edge_weight_t)[mask_t]
            for i in range(1, self.num_layers):
                state[i][mask_t] = self.layers[i](edge_index_t, state[i - 1], state[i], edge_weight_t)[mask_t]
        if self.fully:
            return torch.cat([self.pooling(x, batch) if self.pooling else x for x in state], dim=-1)
        else:
            return self.pooling(state[-1], batch) if self.pooling else state[-1]

    def _embed_sequence(self, edge_index: Union[List[Adj], Adj], edge_weight: Optional[Union[List[Tensor], Tensor]],
                        input: Union[Tensor, List[Tensor]], state: List[Tensor], mask: Optional[List[Tensor]],
                        batch: OptTensor) -> Tensor:
        """
        Compute sequence-to-sequence embedding

        :param edge_index: Sequence of adjacency matrices (time × Adj), or Adj in the case of the spatio-temporal setting
        :param edge_weight: Optional sequence of edge weights, or fixed edge weights for the spatio-temporal setting
        :param input: Input graph signal (time × nodes × in_features)
        :param state: Initial state (nodes × hidden_features) for all reservoir layers
        :param mask: Sequence of node masks (optional, useful for padding dynamic graphs with different lengths)
        :param batch: Batch index (optional)
        :return: Encoding (time × samples × dim)
        """
        size = int(batch.max().item() + 1) if self.pooling else state[0].shape[0]
        embeddings = torch.zeros(len(input), size, self.out_features).to(input[0])
        for t in range(len(input)):
            edge_index_t = edge_index[t] if isinstance(edge_index, list) else edge_index
            edge_weight_t = edge_weight[t] if isinstance(edge_weight, list) else edge_weight
            mask_t = slice(None) if mask is None else mask[t]
            state[0][mask_t] = self.layers[0](edge_index_t, input[t], state[0], edge_weight_t)[mask_t]
            for i in range(1, self.num_layers):
                state[i][mask_t] = self.layers[i](edge_index_t, state[i - 1], state[i], edge_weight_t)[mask_t]
            if self.fully:
                embeddings[t] = torch.cat([self.pooling(x, batch) if self.pooling else x for x in state], dim=-1)
            else:
                embeddings[t] = self.pooling(state[-1], batch) if self.pooling else state[-1]
        return embeddings

    @property
    def out_features(self) -> int:
        """Embedding dimension"""
        return sum(layer.out_features for layer in self.layers) if self.fully else self.layers[-1].out_features

    def extra_repr(self) -> str:
        return f'return_sequences={self.return_sequences}'
