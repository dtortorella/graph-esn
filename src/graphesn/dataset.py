import os.path
from typing import Union, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import download_url, extract_zip
from tqdm import tqdm

from graphesn.data import DynamicData


class TGKDataset:
    """
    Susceptible-Infected dynamic graphs, from the "Temporal Graph Kernels" paper

    See:
    Oettershagen, Kriege, Morris, Mutzel (2020). Temporal Graph Kernels for Classifying Dissemination Processes.
    Proceedings of the 2020 SIAM International Conference on Data Mining, pp. 496–504.
    Available at http://arxiv.org/abs/1911.05496

    :param name: Dataset name (e.g. 'dblp_ct1')
    :param root: Root directory for data
    """

    def __init__(self, name: str, root: str = '/tmp'):
        self.name = name
        self.root = root
        path = self._check_and_download()
        self.data = self._load_data(path)

    def _check_and_download(self) -> str:
        path = os.path.join(self.root, self.name)
        if not os.path.isdir(path):
            zip_file = download_url(f'https://www.chrsmrrs.com/graphkerneldatasets/{self.name}.zip', self.root)
            extract_zip(zip_file, self.root)
            os.unlink(zip_file)
        return path

    def _load_data(self, path: str) -> List[DynamicData]:
        data = []
        edges = np.loadtxt(os.path.join(path, f'{self.name}_A.txt'), delimiter=',', dtype=np.int64) - 1
        indicator = np.loadtxt(os.path.join(path, f'{self.name}_graph_indicator.txt'), dtype=np.int64) - 1
        timestamps = np.loadtxt(os.path.join(path, f'{self.name}_edge_attributes.txt'), dtype=np.int64)
        y = np.loadtxt(os.path.join(path, f'{self.name}_graph_labels.txt'), dtype=np.int64)
        df = pd.read_csv(os.path.join(path, f'{self.name}_node_labels.txt'), names=['t0', 'x0', 't1', 'x1'],
                         dtype={'t0': np.int64, 'x0': np.float32, 't1': 'Int64', 'x1': np.float32})
        t1, x1 = df.t1.to_numpy(dtype=np.int64, na_value=-1), df.x1.to_numpy(dtype=np.float32)
        T = max(t1.max(), timestamps.max()) + 1
        change_mask = (t1 >= 0)
        x = np.zeros((len(t1), T), dtype=np.float32)
        x[change_mask, t1[change_mask]] = x1[change_mask]
        x = np.cumsum(x, axis=1)
        pos = np.cumsum(np.bincount(indicator + 1))
        empty_adj = torch.empty((2, 0), dtype=torch.int64)
        for sample in tqdm(np.unique(indicator)):
            edge_slice = np.bitwise_and(edges[:, 0] >= pos[sample], edges[:, 0] < pos[sample + 1])
            edge_index = [empty_adj] * T
            for t in np.unique(timestamps[edge_slice]):
                time_edge_slice = np.bitwise_and(edge_slice, timestamps == t)
                edge_index[t] = torch.tensor(edges[time_edge_slice] - pos[sample]).t()
            unmasked = np.unique(
                np.concatenate([timestamps[edge_slice], t1[[sample]]]) if t1[sample] >= 0 else timestamps[edge_slice])
            mask = np.zeros((T, pos[sample + 1] - pos[sample]), dtype=bool)
            mask[unmasked, :] = True
            data.append(DynamicData(static_keys=['y'], edge_index=edge_index, y=torch.tensor(y[sample]),
                                    x=torch.tensor(x[pos[sample]:pos[sample + 1]].T).unsqueeze(-1),
                                    mask=torch.tensor(mask)))
        return data

    def __getitem__(self, item: Union[int, Iterable[int]]):
        if hasattr(item, '__iter__'):
            return [self.data[index] for index in item]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'{self.name}({len(self.data)})'
