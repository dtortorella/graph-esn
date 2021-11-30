import os.path
from typing import Union, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import download_url, extract_zip
from torch_geometric_temporal import TwitterTennisDatasetLoader, PedalMeDatasetLoader, WikiMathsDatasetLoader
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from tqdm import tqdm

from graphesn.data import DynamicData, TemporalData

__all__ = ['TGKDataset', 'chickenpox_dataset', 'twitter_tennis_dataset', 'pedalme_dataset', 'wiki_maths_dataset']


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


def chickenpox_dataset(target_lags: int = 8, feature_lags: bool = True) -> TemporalData:
    """
    Chickenpox temporal graph dataset

    See:
    Chickenpox Cases in Hungary: a Benchmark Dataset for Spatiotemporal Signal Processing with Graph Neural Networks.
    Benedek Rozemberczki, Paul Scherer, Oliver Kiss, Rik Sarkar, Tamas Ferenci (2021).
    https://arxiv.org/abs/2102.08100

    :param target_lags: Prediction target time-steps lag
    :param feature_lags: If true, include lagged features (default true)
    :return: A single temporal graph
    """
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset(lags=target_lags)
    return TemporalData(temporal_keys=['x', 'y'],
                        edge_index=torch.from_numpy(dataset.edge_index),
                        x=torch.stack([torch.from_numpy(x.astype('float32')) if feature_lags else torch.from_numpy(
                            x[:, 0].astype('float32')).unsqueeze(dim=-1) for x in dataset.features], dim=0),
                        y=torch.stack(
                            [torch.from_numpy(y.astype('float32')).unsqueeze(dim=-1) for y in dataset.targets], dim=0))


def twitter_tennis_dataset(event_id: str = 'rg17', num_nodes: Optional[int] = None, feature_mode: str = 'encoded',
                           target_offset: int = 1) -> DynamicData:
    """
    Twitter tennis dynamic graph dataset

    See:
    F. Béres, R. Pálovics, A. Oláh, et al. Temporal walk based centrality metric for graph streams.
    Applied Network Science 3, 32 (2018). https://doi.org/10.1007/s41109-018-0080-5

    :param event_id: Name of the event, 'rg17' or 'uo17'
    :param num_nodes: Select top nodes (optional, default all)
    :param feature_mode: Can be None for raw features, or 'encoded' for one hot encoding, or 'diagonal' for identity matrix
    :param target_offset: Prediction off-set for node labels (default 1)
    :return: A single dynamic graph
    """
    loader = TwitterTennisDatasetLoader(event_id=event_id, N=num_nodes, feature_mode=feature_mode,
                                        target_offset=target_offset)
    dataset = loader.get_dataset()
    return DynamicData(edge_index=[torch.from_numpy(edge_index) for edge_index in dataset.edge_indices],
                       edge_weight=[torch.from_numpy(edge_weight.astype('float32')) for edge_weight in
                                    dataset.edge_weights],
                       x=torch.stack([torch.from_numpy(x.astype('float32')) for x in dataset.features], dim=0),
                       y=torch.stack(
                           [torch.from_numpy(y.astype('float32')).unsqueeze(dim=-1) for y in dataset.targets], dim=0))


def pedalme_dataset(target_lags: int = 8, feature_lags: bool = True) -> TemporalData:
    """
    PedalMe temporal graph dataset

    See:
    B. Rozemberczki, P. Scherer, Y. He, et al. (2021). PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models.
    Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM '21), pp. 4564–4573.
    https://doi.org/10.1145/3459637.3482014

    :param target_lags: Prediction target time-steps lag
    :param feature_lags: If true, include lagged features (default true)
    :return: A single temporal graph
    """
    loader = PedalMeDatasetLoader()
    dataset = loader.get_dataset(lags=target_lags)
    return TemporalData(temporal_keys=['x', 'y'],
                        edge_index=torch.from_numpy(dataset.edge_index),
                        edge_weight=torch.from_numpy(dataset.edge_weight.astype('float32')),
                        x=torch.stack([torch.from_numpy(x.astype('float32')) if feature_lags else torch.from_numpy(
                            x[:, 0].astype('float32')).unsqueeze(dim=-1) for x in dataset.features], dim=0),
                        y=torch.stack(
                            [torch.from_numpy(y.astype('float32')).unsqueeze(dim=-1) for y in dataset.targets], dim=0))


def wiki_maths_dataset(target_lags: int = 8, feature_lags: bool = True) -> TemporalData:
    """
    Wiki Maths temporal graph dataset

    See:
    B. Rozemberczki, P. Scherer, Y. He, et al. (2021). PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models.
    Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM '21), pp. 4564–4573.
    https://doi.org/10.1145/3459637.3482014

    :param target_lags: Prediction target time-steps lag
    :param feature_lags: If true, include lagged features (default true)
    :return: A single temporal graph
    """
    loader = WikiMathsDatasetLoader()
    dataset = loader.get_dataset(lags=target_lags)
    return TemporalData(temporal_keys=['x', 'y'],
                        edge_index=torch.from_numpy(dataset.edge_index),
                        edge_weight=torch.from_numpy(dataset.edge_weight.astype('float32')),
                        x=torch.stack([torch.from_numpy(x.astype('float32')) if feature_lags else torch.from_numpy(
                            x[:, 0].astype('float32')).unsqueeze(dim=-1) for x in dataset.features], dim=0),
                        y=torch.stack(
                            [torch.from_numpy(y.astype('float32')).unsqueeze(dim=-1) for y in dataset.targets], dim=0))
