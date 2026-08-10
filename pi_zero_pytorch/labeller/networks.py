import torch

from pi_zero_pytorch.pi_zero import BinnedValueLayer, PiZero, SigLIP


class SmallValueNetwork(torch.nn.Module):
    def __init__(
        self,
        image_size = 224,
        patch_size = 14,
        dim = 256,
        depth = 4,
        heads = 8,
        min_value = -1.,
        max_value = 0.,
        num_bins = 201
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        self.siglip = SigLIP(
            image_size = image_size,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            heads = heads
        )

        self.to_value = BinnedValueLayer(
            dim = dim,
            min_value = min_value,
            max_value = max_value,
            num_bins = num_bins
        )

    def forward(self, images, return_value_and_logits = False):
        embeds = self.siglip(images)
        pooled = embeds.mean(dim = 1)
        return self.to_value(pooled, return_value_and_logits = return_value_and_logits)


VALUE_NETWORK_CONFIGS = {
    "mock": {"dim": 8, "depth": 1, "heads": 1, "image_size": 32, "patch_size": 16},
    "small": {"dim": 64, "depth": 2, "heads": 4, "image_size": 224, "patch_size": 14},
    "medium": {"dim": 128, "depth": 4, "heads": 8, "image_size": 224, "patch_size": 14},
    "large": {"dim": 256, "depth": 6, "heads": 8, "image_size": 224, "patch_size": 14}
}


class SmallPiZero(torch.nn.Module):
    def __init__(
        self,
        dim = 32,
        dim_action = 32,
        dim_action_input = 6,
        dim_joint_state = 32,
        num_tokens = 256,
        depth = 2,
        heads = 4,
        image_size = 32,
        patch_size = 4,
        max_text_len = 32,
        num_advantage_tokens = 2
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_text_len = max_text_len

        # minimal vit
        self.vit = SigLIP(
            image_size = image_size,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            heads = heads
        )

        # minimal pizero
        self.pizero = PiZero(
            dim = dim,
            num_tokens = num_tokens,
            dim_action_input = dim_action_input,
            dim_joint_state = dim_joint_state,
            dim_action = dim_action,
            depth = depth,
            heads = heads,
            vit = self.vit,
            vit_dim = dim,
            num_advantage_tokens = num_advantage_tokens
        )

    def forward(self, images, token_ids, joint_state, actions, advantage_ids = None, **kwargs):
        return self.pizero(
            images = images,
            token_ids = token_ids,
            joint_state = joint_state,
            actions = actions,
            advantage_ids = advantage_ids,
            **kwargs
        )


PI_ZERO_CONFIGS = {
    "mock": {"dim": 4, "depth": 1, "heads": 1, "image_size": 32, "patch_size": 16},
    "small": {"dim": 16, "depth": 1, "heads": 2, "image_size": 32, "patch_size": 8}
}
