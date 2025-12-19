from __future__ import annotations
import pickle

from pathlib import Path
from collections import namedtuple
from contextlib import contextmanager

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Literal, Dict, Tuple, Union, Optional, List
import bisect

import numpy as np
from numpy import ndarray
from numpy.lib.format import open_memmap

import torch
from torch import tensor, is_tensor, arange, from_numpy, stack, broadcast_tensors
from torch.utils.data import Dataset, DataLoader

from einops import rearrange
import einx

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# types

PRIMITIVE_TYPES = Union[int, float, bool]

PRIMITIVE_TYPE_STR = Literal['int', 'float', 'bool']

FIELD_TYPE = Dict[
    str,
    Union[
        str,
        Tuple[PRIMITIVE_TYPE_STR, Union[int, Tuple[int, ...]]],
        Tuple[PRIMITIVE_TYPE_STR, Union[int, Tuple[int, ...]], PRIMITIVE_TYPES]
    ]
]

# classes

class ReplayBuffer:

    @beartype
    def __init__(
        self,
        folder: str | Path,
        max_episodes: int,
        max_timesteps: int,
        fields: FIELD_TYPE,
        meta_fields: FIELD_TYPE = dict(),
        circular = False,
        overwrite = False
    ):
        # folder for data

        if not isinstance(folder, Path):
            folder = Path(folder)

        folder.mkdir(exist_ok = True, parents = True)

        self.folder = folder
        assert folder.is_dir()

        # save the hyperparameters, so it can be rehydrated from a json file

        config_path = folder / 'data.pkl'

        if not config_path.exists():
            config = dict(
                max_episodes = max_episodes,
                max_timesteps = max_timesteps,
                fields = fields,
                meta_fields = meta_fields
            )

            with open(str(config_path), 'wb') as data:
                pickle.dump(config, data)

        # keeping track of episode length

        num_episodes_path = folder / 'num_episodes.state.npy'
        episode_index_path = folder / 'episode_index.state.npy'
        timestep_index_path = folder / 'timestep_index.state.npy'

        self._num_episodes = open_memmap(str(num_episodes_path), mode = 'w+' if not num_episodes_path.exists() or overwrite else 'r+', dtype = np.int32, shape = ())
        self._episode_index = open_memmap(str(episode_index_path), mode = 'w+' if not episode_index_path.exists() or overwrite else 'r+', dtype = np.int32, shape = ())
        self._timestep_index = open_memmap(str(timestep_index_path), mode = 'w+' if not timestep_index_path.exists() or overwrite else 'r+', dtype = np.int32, shape = ())

        if overwrite:
            self.num_episodes = 0
            self.episode_index = 0
            self.timestep_index = 0

        # auto infer the max episodes and max timesteps by grabbing a random data memmap file

        if not exists(max_episodes) or not exists(max_timesteps):
            field_name, field_info = next(iter(fields.items()))
            filepath = folder / f'{file_name}.data.npy'
            assert filepath.exists(), f'if not instantiating buffer from existing folder'

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps

        assert 'episode_lens' not in meta_fields

        meta_fields.update(episode_lens = 'int')

        def field_info_to_shape_dtype(field_info):
            if isinstance(field_info, str):
                field_info = (field_info, (), None)

            elif is_bearable(field_info, tuple[PRIMITIVE_TYPE_STR, int | tuple[int, ...]]):
                field_info = (*field_info, None)

            dtype_str, shape, default_value = field_info

            dtype = dict(int = np.int32, float = np.float32, bool = np.bool_)[dtype_str]

            return shape, dtype, default_value

        # create the memmap for individual data tracks

        self.shapes = dict()
        self.dtypes = dict()
        self.data = dict()
        self.fieldnames = set(fields.keys())

        for field_name, field_info in fields.items():

            shape, dtype, default_value = field_info_to_shape_dtype(field_info)

            # memmap file

            filepath = folder / f'{field_name}.data.npy'

            if isinstance(shape, int):
                shape = (shape,)

            mode = 'r+' if filepath.exists() and not overwrite else 'w+'
            memmap = open_memmap(str(filepath), mode = mode, dtype = dtype, shape = (max_episodes, max_timesteps, *shape))

            if exists(default_value):
                memmap[:] = default_value

            self.data[field_name] = memmap
            self.shapes[field_name] = shape
            self.dtypes[field_name] = dtype

        self.memory_namedtuple = namedtuple('Memory', list(fields.keys()))

        # meta data

        self.meta_shapes = dict()
        self.meta_dtypes = dict()
        self.meta_data = dict()
        self.meta_fieldnames = set(meta_fields.keys())

        for field_name, field_info in meta_fields.items():

            shape, dtype, default_value = field_info_to_shape_dtype(field_info)

            # memmap file

            filepath = folder / f'{field_name}.meta.data.npy'

            if isinstance(shape, int):
                shape = (shape,)

            mode = 'r+' if filepath.exists() else 'w+'
            memmap = open_memmap(str(filepath), mode = mode, dtype = dtype, shape = (max_episodes, *shape))

            if exists(default_value):
                memmap[:] = default_value

            self.meta_data[field_name] = memmap
            self.meta_shapes[field_name] = shape
            self.meta_dtypes[field_name] = dtype

        # whether the buffer should loop back around - for online policy opt related

        self.circular = circular

    @property
    def num_episodes(self):
        return self._num_episodes.item()

    @num_episodes.setter
    def num_episodes(self, value):
        self._num_episodes[()] = value
        self._num_episodes.flush()

    @property
    def episode_index(self):
        return self._episode_index.item()

    @episode_index.setter
    def episode_index(self, value):
        self._episode_index[()] = value
        self._episode_index.flush()

    @property
    def timestep_index(self):
        return self._timestep_index.item()

    @timestep_index.setter
    def timestep_index(self, value):
        self._timestep_index[()] = value
        self._timestep_index.flush()

    @classmethod
    def from_config(cls, folder, config_name = 'data.pkl'):
        filepath = folder / config_name
        assert filepath.exists()

        with open(str(filepath), 'rb') as data:
            config = pickle.load(data)

        return cls(folder, **config)

    @property
    def episode_lens(self):
        return self.meta_data['episode_lens']

    def __len__(self):
        return (self.episode_lens > 0).sum().item()

    def reset_(self):
        self.episode_lens[:] = 0
        self.num_episodes = 0
        self.episode_index = 0
        self.timestep_index = 0

    def advance_episode(self):

        assert self.circular or self.num_episodes < self.max_episodes

        self.episode_lens[self.episode_index] = self.timestep_index
        
        self.episode_index = (self.episode_index + 1) % self.max_episodes
        self.timestep_index = 0

        self.num_episodes += 1
        self.num_episodes = min(self.max_episodes, self.num_episodes)

    def flush(self):

        for memmap in self.data.values():
            memmap.flush()

        for memmap in self.meta_data.values():
            memmap.flush()

        self._num_episodes.flush()
        self._episode_index.flush()
        self._timestep_index.flush()

    @contextmanager
    def one_episode(
        self,
        **meta_data
    ):
        assert self.circular or self.num_episodes < self.max_episodes

        for name, metadata in meta_data.items():
            self.store_meta_datapoint(self.episode_index, name, metadata)

        yield

        self.advance_episode()
        self.flush()

    @beartype
    def store_datapoint(
        self,
        episode_index: int,
        timestep_index: int,
        name: str,
        datapoint: torch.Tensor | ndarray | PRIMITIVE_TYPES
    ):
        assert 0 <= episode_index < self.max_episodes
        assert 0 <= timestep_index < self.max_timesteps

        if is_bearable(datapoint, PRIMITIVE_TYPES):
            datapoint = tensor(datapoint)

        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        assert name in self.fieldnames, f'invalid field name {name} - must be one of {self.fieldnames}'

        assert datapoint.shape == self.shapes[name], f'invalid shape {datapoint.shape} - shape must be {self.shapes[name]}'

        self.data[name][self.episode_index, self.timestep_index] = datapoint

    @beartype
    def store_meta_datapoint(
        self,
        episode_index: int,
        name: str,
        datapoint: torch.Tensor | ndarray | PRIMITIVE_TYPES
    ):
        assert 0 <= episode_index < self.max_episodes

        if is_bearable(datapoint, PRIMITIVE_TYPES):
            datapoint = tensor(datapoint)

        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        assert name in self.meta_fieldnames, f'invalid field name {name} - must be one of {self.fieldnames}'

        assert datapoint.shape == self.meta_shapes[name], f'invalid shape {datapoint.shape} - shape must be {self.shapes[name]}'

        self.meta_data[name][self.episode_index] = datapoint

    def store(
        self,
        **data
    ):
        assert is_bearable(data, dict[str, torch.Tensor | ndarray])

        assert not self.timestep_index >= self.max_timesteps, 'you exceeded the `max_timesteps` set on the replay buffer'

        for name, datapoint in data.items():

            self.store_datapoint(self.episode_index, self.timestep_index, name, datapoint)

        self.timestep_index += 1

        # determine what fields are missing and just default to nothing

        missing_fields = self.fieldnames - set(data.keys())

        for missing_field in missing_fields:
            dtype = self.dtypes[missing_field]
            shape = self.shapes[missing_field]
            data.update(**{missing_field: np.zeros(shape, dtype = dtype)})

        # store

        return self.memory_namedtuple(**data)

class ReplayDataset(Dataset):
    def __init__(
        self,
        experiences: ReplayBuffer,
        task_id: int | None = None,
        fields: list[str] | None = None,
        fieldname_map: dict[str, str] = dict(),
        return_indices = False
    ):
        self.experiences = experiences
        self.return_indices = return_indices

        episode_ids = arange(experiences.max_episodes)
        episode_lens = from_numpy(experiences.episode_lens)

        max_episode_len = episode_lens.amax().item()

        valid_mask = (experiences.episode_lens > 0) & ~experiences.meta_data['invalidated']

        # filter by task id

        if exists(task_id):
            is_task_id = experiences.meta_data['task_id'] == task_id
            valid_mask = valid_mask & is_task_id

        valid_episodes = episode_ids[valid_mask]
        self.valid_episodes = valid_episodes
        valid_episode_lens = episode_lens[valid_mask]

        timesteps = arange(max_episode_len)

        episode_timesteps = stack(broadcast_tensors(
            rearrange(valid_episodes, 'e -> e 1'),
            rearrange(timesteps, 't -> 1 t')
        ), dim = -1)

        valid_timesteps = einx.less('j, i -> i j', timesteps, valid_episode_lens)

        # filter by invalidated - bytedance's filtered BC method

        if 'invalidated' in experiences.data:
            timestep_invalidated = experiences.data['invalidated'][valid_episodes, :max_episode_len]

            valid_timesteps = valid_timesteps & ~timestep_invalidated

        self.timepoints = episode_timesteps[valid_timesteps]

        self.fields = default(fields, list(experiences.fieldnames))

        self.fieldname_map = fieldname_map

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, idx):
        episode_id, timestep_index = self.timepoints[idx].unbind(dim = -1)

        step_data = dict()

        for field in self.fields:
            data = self.experiences.data[field]

            model_kwarg_name = self.fieldname_map.get(field, field)

            step_data[model_kwarg_name] = data[episode_id, timestep_index]

        if self.return_indices:
            step_data['indices'] = self.timepoints[idx]

        return step_data

class JoinedReplayDataset(Dataset):
    def __init__(
        self,
        datasets: list[ReplayDataset],
        meta_buffer: ReplayBuffer
    ):
        super().__init__()
        self.datasets = datasets
        self.meta_buffer = meta_buffer

        meta_episode_offset = 0
        self.meta_episode_offsets = []

        for dataset in datasets:
            self.meta_episode_offsets.append(meta_episode_offset)
            meta_episode_offset += len(dataset.valid_episodes)

        from torch.utils.data import ConcatDataset
        self.concat_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.concat_dataset.cumulative_sizes, idx)
        
        local_idx = idx
        if dataset_idx > 0:
            local_idx = idx - self.concat_dataset.cumulative_sizes[dataset_idx - 1]

        dataset = self.datasets[dataset_idx]
        data = dataset[local_idx]

        # Map to meta buffer
        source_episode_id, timestep_index = dataset.timepoints[local_idx].unbind(dim = -1)
        
        # We need relative episode index within the dataset's valid episodes
        relative_episode_idx = torch.searchsorted(dataset.valid_episodes, source_episode_id)
        
        meta_episode_id = self.meta_episode_offsets[dataset_idx] + relative_episode_idx
        
        # Get meta fields (value, advantages, advantage_ids)
        for field in self.meta_buffer.fieldnames:
            meta_data = self.meta_buffer.data[field][meta_episode_id, timestep_index]
            data[field] = tensor(meta_data)
            
        return data
