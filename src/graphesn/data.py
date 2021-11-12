from __future__ import annotations

from typing import List, Any

from torch_geometric.data import Data


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
