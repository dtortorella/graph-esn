import argparse

import torch
from torch.nn.functional import one_hot
from torch_geometric.datasets import Planetoid
from graphesn import StaticGraphReservoir, initializer, Readout
from graphesn.util import graph_spectral_norm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name')
parser.add_argument('--root', help='root directory for dataset', default='/tmp')
parser.add_argument('--device', help='device for torch computations', default='cpu')
parser.add_argument('--layers', help='reservoir layers', type=int, default=2)
parser.add_argument('--units', help='reservoir units per layer', type=int, default=64)
parser.add_argument('--init', help='random recurrent initializer (uniform, normal, ring)', type=str, default='uniform')
parser.add_argument('--iterations', help='max iterations', type=int, default=None)
parser.add_argument('--epsilon', help='learning rate', type=float, default=1e-6)
parser.add_argument('--rho', help='rho for recurrent matrix initialization', type=float, default=None)
parser.add_argument('--sigma', help='sigma for recurrent matrix initialization', type=float, default=None)
parser.add_argument('--scale', help='scale for input matrix initialization', type=float, default=1.0)
parser.add_argument('--ld', help='readout lambda', type=float, default=1e-3)
parser.add_argument('--fully', help='whether to use all layers embeddings', action='store_true')
parser.add_argument('--bias', help='whether bias term is present', action='store_true')
args = parser.parse_args()

dataset = Planetoid(root=args.root, name=args.dataset)
device = torch.device(args.device)
data = dataset[0].to(device)

alpha = graph_spectral_norm(data.edge_index)
print(f'graph alpha = {float(alpha):.2f}')

reservoir = StaticGraphReservoir(num_layers=args.layers, in_features=dataset.num_features, hidden_features=args.units,
                                 max_iterations=args.iterations, epsilon=args.epsilon, bias=args.bias)
reservoir.initialize_parameters(recurrent=initializer(args.init, rho=args.rho / alpha if args.rho else None, sigma=args.sigma / alpha if args.sigma else None),
                                input=initializer('uniform', scale=1.0),
                                bias=initializer('uniform', scale=0.1))
reservoir.layers[0].initialize_parameters(recurrent=initializer(args.init, rho=args.rho / alpha if args.rho else None, sigma=args.sigma / alpha if args.sigma else None),
                                          input=initializer('uniform', scale=args.scale),
                                          bias=initializer('uniform', scale=0.1))
reservoir = reservoir.to(device)

x = reservoir(data.edge_index, data.x)
y = one_hot(data.y, dataset.num_classes).float()

readout = Readout(num_features=reservoir.out_features, num_targets=dataset.num_classes)
readout.fit((x[data.train_mask], y[data.train_mask]), args.ld)
y_pred = readout(x)

print(f'training accuracy: {(y_pred[data.train_mask].argmax(dim=-1) == data.y[data.train_mask]).float().mean() * 100:.2f}%')
print(f'test accuracy: {(y_pred[data.test_mask].argmax(dim=-1) == data.y[data.test_mask]).float().mean() * 100:.2f}%')
