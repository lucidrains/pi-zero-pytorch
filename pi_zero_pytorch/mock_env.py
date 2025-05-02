from __future__ import annotations
from random import choice

import torch
from torch import tensor, randn, randint
from torch.nn import Module

# functions

def cast_tuple(v):
    return v if isinstance(v, tuple) else (v,)

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
        reward = randint(-100, 100, ()).float()

        if self._step > self.can_terminate_after:
            truncated = tensor(choice((True, False)))
            terminated = tensor(choice((True, False)))
        else:
            truncated = terminated = tensor(False)

        self._step.add_(1)

        return state, reward, truncated, terminated
