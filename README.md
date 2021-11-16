# Graph ESN library
Pytorch implementation of echo state networks for static graphs and discrete-time dynamic graphs.

## Content Summmary
- ``` examples/planetoid ``` contains an example of the library usage in the static graph domain.
- ``` src/graphesn/reservoir ``` contains the implementations of the echo state networks for static graphs and discrete-time dynamic graphs.
- ``` src/graphesn/readout ``` contains the implementation of a linear readout.
- ``` src/graphesn/matrix ``` contains useful matrix operations.

## Installation
``` python3 -m pip install graphesn ```

## References
* C. Gallicchio, A. Micheli (2010). Graph Echo State Networks. The 2010 International Joint Conference on Neural Networks (IJCNN 2010), pp. 3967–3974.
* C. Gallicchio, A. Micheli (2020). Fast and Deep Graph Neural Networks. The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20).
* C. Gallicchio, A. Micheli (2020). Ring Reservoir Neural Networks for Graphs. The 2020 International Joint Conference on Neural Networks (IJCNN 2020).
* D. Tortorella, A. Micheli (2021). Dynamic Graph Echo State Networks. Proceedings of the 29th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2021), pp. 99–104.

## License
graph-esn is MIT licensed, as written in the LICENSE file.

## Contributing
**This research software is provided as-is**. We are working on this library in our spare time.

If you find a bug, please open an issue to report it, and we will do our best to solve it. For generic/technical questions, please email us rather than opening an issue.