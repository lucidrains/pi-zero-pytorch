import pytest

import torch
from pi_zero_pytorch import π0

@pytest.mark.parametrize('only_vlm', (True, False))
@pytest.mark.parametrize('num_residual_streams', (1, 4))
def test_pi_zero_with_vit(
    only_vlm: bool,
    num_residual_streams: int
):
    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    v = Extractor(v, return_embeddings_only = True)

    model = π0(
        dim = 512,
        vit = v,
        vit_dim = 1024,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 20_000,
        num_residual_streams = num_residual_streams
    )

    images = torch.randn(2, 3, 2, 256, 256)
    commands = torch.randint(0, 20_000, (2, 1024))

    if only_vlm:
        vlm_logits = model.forward_only_vision_language(images, commands)
        assert vlm_logits.ndim == 3
        return

    joint_state = torch.randn(2, 12)
    actions = torch.randn(2, 32, 6)

    loss, _ = model(images, commands, joint_state, actions)
    loss.backward()

    # after much training

    sampled_actions = model(images, commands, joint_state, trajectory_length = 32) # (1, 32, 6)

    assert sampled_actions.shape == (2, 32, 6)
