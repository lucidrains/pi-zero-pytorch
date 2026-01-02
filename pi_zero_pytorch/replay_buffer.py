from __future__ import annotations

import torch
from torch import tensor, from_numpy, stack, arange, broadcast_tensors
from torch.utils.data import Dataset
import torch.nn.functional as F

import bisect
import einx
from einops import rearrange

from memmap_replay_buffer import ReplayBuffer

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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

        valid_mask = episode_lens > 0
        
        if 'invalidated' in experiences.meta_data:
            valid_mask = valid_mask & ~from_numpy(experiences.meta_data['invalidated'])

        # filter by task id

        if exists(task_id):
            is_task_id = from_numpy(experiences.meta_data['task_id']) == task_id
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
            timestep_invalidated = from_numpy(experiences.data['invalidated'][valid_episodes, :max_episode_len])

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
