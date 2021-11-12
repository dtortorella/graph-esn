from __future__ import annotations

from collections import UserList
from typing import List, Any

from torch import Tensor
from torch_geometric.data import Data, Batch


class TemporalData(Data):
    """
    Data object describing a temporal graph.
    Attribute declared as temporal keys have first dimension as time dimension;
    second dimension assumed as node (or graph) dimension.

    :param temporal_keys: Which keys have a first temporal dimension (default only 'x')

    Example::
        >>> data = TemporalData(temporal_keys=['x', 'y'], x=torch.rand(10,7,3), edge_index=torch.tensor(...), y=torch.rand(10,1))
        # Temporal graph having 10 time-steps, 7 nodes, 3 features per node, and a sequence of graph attributes for each time-step.
        >>> Batch.from_data_list([data, data])
        TemporalDataBatch(x=[10, 14, 3], edge_index=[2, 12], y=[10, 2], batch=[14], ptr=[3])
    """

    def __init__(self, temporal_keys: List[str] = ['x'], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temporal_keys = temporal_keys

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in self._temporal_keys:
            return 1
        else:
            return super().__cat_dim__(key, value, args, kwargs)

    @property
    def num_timesteps(self) -> int:
        return self[self._temporal_keys[0]].shape[0]

    @staticmethod
    def from_data(data: Data, temporal_keys: List[str] = ['x']) -> TemporalData:
        return TemporalData(temporal_keys=temporal_keys, **data.to_dict())


class DynamicData(object):
    def __init__(self, static_keys: List[str] = [], **stuff):
        self._static_keys = static_keys
        self._storage = stuff

    def __getitem__(self, t: int):
        return Data(**{key: value if key in self._static_keys else value[t] for key, value in self._storage.items()})

    def __getattr__(self, key: str):
        if key in self.keys:
            return self._storage[key]
        else:
            raise AttributeError(f"No attribute '{key}'")

    @property
    def keys(self) -> List[str]:
        return list(self._storage.keys())

    @property
    def num_timesteps(self) -> int:
        return len(self._storage[[key for key in self.keys if key not in self._static_keys][0]])

    def to(self, *args, **kwargs):
        for key in self.keys:
            if isinstance(self._storage[key], Tensor):
                self._storage[key] = self._storage[key].to(*args, **kwargs)
            else:
                for t in range(len(self._storage[key])):
                    self._storage[key][t] = self._storage[key][t].to(*args, **kwargs)
        return self

    def __repr__(self):
        return f"DynamicData({', '.join(self.keys)})"


class DynamicBatch(UserList):
    def __init__(self, data: List[DynamicData]):
        super().__init__()
        self.data = [Batch.from_data_list([sample[t] for sample in data]) for t in range(data[0].num_timesteps)]

    def to(self, *args, **kwargs):
        for t in range(len(self)):
            self.data[t] = self.data[t].to(*args, **kwargs)
        return self

    def __repr__(self):
        return f'DynamicBatch[{len(self)}]'
