from shutil import rmtree
from pathlib import Path
import random
from random import choice

import torch
from torch import tensor, randn, randint
from torch.nn import Module

from pi_zero_pytorch.replay_buffer import ReplayBuffer

# functions

def cast_tuple(v):
    return v if isinstance(v, tuple) else (v,)

# mock replay buffer

def create_mock_replay_buffer(
    folder,
    max_episodes,
    max_timesteps,
    num_episodes = None,
    max_task_id = 9,
    image_shape = (256, 256),
    num_images = 2,
    num_text_tokens = 100,
    max_text_len = 32,
    joint_dim = 12,
    dim_action_input = 6,
    trajectory_length = 16,
    cleanup_if_exists = True
):
    if cleanup_if_exists and Path(folder).exists():
        rmtree(folder)

    if not isinstance(folder, Path):
        folder = Path(folder)
    
    folder.mkdir(parents = True, exist_ok = True)

    if num_episodes is None:
        num_episodes = random.randint(1, max_episodes)

    buffer = ReplayBuffer(
        folder,
        max_episodes = max_episodes,
        max_timesteps = max_timesteps,
        meta_fields = dict(
            task_id     = ('int', (), -1),
            fail        = 'bool',
            invalidated = 'bool',
            recap_step  = ('int', (), -1)
        ),
        fields = dict(
            images      = ('float', (3, num_images, *image_shape)),
            text        = ('int', (max_text_len,)),
            internal    = ('float', (joint_dim,)),
            reward      = 'float',
            actions     = ('float', (trajectory_length, dim_action_input)),
            terminated  = 'bool',
            value       = 'float',
            advantages  = 'float',
            returns     = 'float',
            advantage_ids = 'int',
            task_id     = 'int',
            invalidated = 'bool'
        )
    )

    for _ in range(num_episodes):
        episode_len = random.randint(1, max_timesteps)
        task_id = random.randint(0, max_task_id)

        with buffer.one_episode(
            task_id = task_id,
        ):
            for step in range(episode_len):
                
                # generate random data
                
                images = randn(3, num_images, *image_shape)
                text = randint(0, num_text_tokens, (max_text_len,))
                internal = randn(joint_dim)
                reward = randn(())
                actions = randn(trajectory_length, dim_action_input)
                terminated = (step == episode_len - 1)
                value = randn(())
                advantages = randn(())
                returns = randn(())
                advantage_ids = random.randint(0, 1)
                
                buffer.store(
                    images = images,
                    text = text,
                    internal = internal,
                    reward = reward,
                    actions = actions,
                    terminated = terminated,
                    value = value,
                    advantages = advantages,
                    returns = returns,
                    advantage_ids = advantage_ids,
                    task_id = task_id
                )

    return buffer

# mock env

class Env(Module):
    def __init__(
        self,
        image_shape,
        num_images,
        num_text_tokens,
        max_text_len,
        joint_dim,
        can_terminate_after = 2
    ):
        super().__init__()
        self.image_shape = image_shape
        self.num_images = num_images
        self.num_text_tokens = num_text_tokens
        self.max_text_len = max_text_len
        self.joint_dim = joint_dim

        self.can_terminate_after = can_terminate_after
        self.register_buffer('_step', tensor(0))

    def get_random_state(self):
        return (
            randn(3, self.num_images, *self.image_shape),
            randint(0, self.num_text_tokens, (self.max_text_len,)),
            randn(self.joint_dim)
        )

    def reset(
        self,
        seed = None
    ):
        self._step.zero_()
        return self.get_random_state()

    def step(
        self,
        actions,
    ):
        state = self.get_random_state()
        reward = tensor(-1.)

        if self._step > self.can_terminate_after:
            truncated = tensor(choice((True, False)))
            terminated = tensor(choice((True, False)))
        else:
            truncated = terminated = tensor(False)

        self._step.add_(1)

        return state, reward, truncated, terminated
