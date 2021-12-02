# Graph ESN library
Pytorch implementation of echo state networks for static graphs and discrete-time dynamic graphs.

## Installation
Easiest way to get our library is via python package:
```
pip install graphesn
```

## Usage
The library is quite straightforward to use:
```python
from graphesn import StaticGraphReservoir, Readout, initializer
from torch_geometric.data import Data

data = Data(...)

reservoir = StaticGraphReservoir(num_layers=3, in_features=8, hidden_features=16)
reservoir.initialize_parameters(recurrent=initializer('uniform', rho=.9), input=initializer('uniform', scale=1))
embeddings = reservoir(data.edge_index, data.x)

readout = Readout(num_features=reservoir.out_features, num_targets=3)
readout.fit(data=(embeddings, data.y), regularization=1e-3)
predictions = readout(embeddings)
```

## Code outlook
The library is contained in folder `src/graphesn`:
- `reservoir.py` implementation of reservoirs for static and discrete-time dynamic graphs;
- `matrix.py` random matrices generating functions;
- `readout.py` implementation of a linear readout for large-scale ridge regression;
- `data.py` classes to represent temporal and dynamic graphs;
- `dataset.py` some dynamic graph datasets;
- `util.py` general utilities.

The `examples` folder contains demos for our library on some common graph datasets.

## Contributing
***This research software is provided as-is***. We are working on this library in our spare time.

Code is released under the MIT license, see `LICENSE` for details.

If you find a bug, please open an issue to report it, and we will do our best to solve it. For general or technical questions, please email us rather than opening an issue.

## References
* C. Gallicchio, A. Micheli (2010). Graph Echo State Networks. The 2010 International Joint Conference on Neural Networks (IJCNN 2010), pp. 3967–3974.
* C. Gallicchio, A. Micheli (2020). Fast and Deep Graph Neural Networks. The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20).
* C. Gallicchio, A. Micheli (2020). Ring Reservoir Neural Networks for Graphs. The 2020 International Joint Conference on Neural Networks (IJCNN 2020).
* D. Tortorella, A. Micheli (2021). Dynamic Graph Echo State Networks. Proceedings of the 29th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2021), pp. 99–104.
