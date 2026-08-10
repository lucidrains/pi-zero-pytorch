from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pi_zero_pytorch.pi_zero import SigLIP, BinnedValueLayer, PiZero


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


# helper classes

class SmallValueNetwork(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        min_value: float = -1.0,
        max_value: float = 0.0,
        num_bins: int = 201
    ):
        super().__init__()
        self.image_size = image_size
        self.siglip = SigLIP(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads
        )
        self.to_value = BinnedValueLayer(
            dim=dim,
            min_value=min_value,
            max_value=max_value,
            num_bins=num_bins
        )

    def forward(self, images: torch.Tensor, return_value_and_logits: bool = False):
        if images.ndim == 5:
            if images.shape[2] == 1:
                images = images.squeeze(2)
            elif images.shape[1] == 1:
                images = images.squeeze(1)
            else:
                b, n, c, h, w = images.shape
                images = images.view(b * n, c, h, w)

        if images.ndim == 4 and images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        embeds = self.siglip(images)
        pooled = embeds.mean(dim=1)
        return self.to_value(pooled, return_value_and_logits=return_value_and_logits)


class SmallPiZero(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        dim_action: int = 32,
        dim_action_input: int = 6,
        dim_joint_state: int = 32,
        num_tokens: int = 256,
        depth: int = 2,
        heads: int = 4,
        image_size: int = 32,
        patch_size: int = 4,
        num_advantage_tokens: int = 2
    ):
        super().__init__()
        self.vit = SigLIP(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads
        )
        self.pizero = PiZero(
            dim=dim,
            num_tokens=num_tokens,
            dim_action_input=dim_action_input,
            dim_joint_state=dim_joint_state,
            dim_action=dim_action,
            depth=depth,
            heads=heads,
            vit=self.vit,
            vit_dim=dim,
            num_advantage_tokens=num_advantage_tokens
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        joint_state: torch.Tensor,
        actions: torch.Tensor,
        advantage_ids: torch.Tensor = None,
        **kwargs
    ):
        return self.pizero(
            images=images,
            token_ids=token_ids,
            joint_state=joint_state,
            actions=actions,
            advantage_ids=advantage_ids,
            **kwargs
        )


VALUE_NETWORK_CONFIGS = {
    "mock": {"dim": 8, "depth": 1, "heads": 1, "image_size": 32, "patch_size": 16},
    "small": {"dim": 64, "depth": 2, "heads": 4, "image_size": 224, "patch_size": 14}
}
