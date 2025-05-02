import pytest

import torch
from pi_zero_pytorch import π0
from einops import repeat, rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.mark.parametrize('only_vlm', (True, False))
@pytest.mark.parametrize('num_residual_streams', (1, 4))
def test_pi_zero_with_vit(
    only_vlm: bool,
    num_residual_streams: int,
):
    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 32,
        depth = 1,
        heads = 16,
        dim_head = 16,
        mlp_dim = 64,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    v = Extractor(v, return_embeddings_only = True)

    model = π0(
        dim = 32,
        vit = v,
        vit_dim = 32,
        depth = 1,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 32,
        num_residual_streams = num_residual_streams,
    ).to(device)

    images = torch.randn(2, 3, 2, 256, 256)
    commands = torch.randint(0, 32, (2, 1024))

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

def test_policy_optimization():
    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    from pi_zero_pytorch.pi_zero import (
        Agent,
        EPO,
    )

    from pi_zero_pytorch.mock_env import Env

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 32,
        depth = 1,
        heads = 2,
        dim_head = 8,
        mlp_dim = 16,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    v = Extractor(v, return_embeddings_only = True)

    model = π0(
        dim = 32,
        vit = v,
        vit_dim = 32,
        depth = 1,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 32,
        policy_optimizable = True
    ).to(device)

    images = torch.randn(2, 3, 2, 256, 256)
    commands = torch.randint(0, 32, (2, 1024))

    joint_state = torch.randn(2, 12)
    actions = torch.randn(2, 32, 6)

    loss, _ = model(images, commands, joint_state, actions)
    loss.backward()

    # agent

    agent = Agent(model)

    mock_env = Env((256, 256), 2, 32, 1024, 12)

    epo = EPO(agent, mock_env)

    memories = epo.gather_experience_from_env(steps = 10)

    epo.learn_agent(memories, batch_size = 2)
