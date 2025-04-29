import pytest

import torch
from pi_zero_pytorch import π0

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
        depth = 6,
        heads = 16,
        dim_head = 16,
        mlp_dim = 64,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)

    v = Extractor(v, return_embeddings_only = True)

    model = π0(
        dim = 32,
        vit = v,
        vit_dim = 32,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 20_000,
        num_residual_streams = num_residual_streams,
    ).to(device)

    images = torch.randn(2, 3, 2, 256, 256).to(device)
    commands = torch.randint(0, 20_000, (2, 1024)).to(device)

    if only_vlm:
        vlm_logits = model.forward_only_vision_language(images, commands)
        assert vlm_logits.ndim == 3
        return

    joint_state = torch.randn(2, 12).to(device)
    actions = torch.randn(2, 32, 6).to(device)

    loss, _ = model(images, commands, joint_state, actions)
    loss.backward()

    # after much training

    sampled_actions = model(images, commands, joint_state, trajectory_length = 32) # (1, 32, 6)

    assert sampled_actions.shape == (2, 32, 6)

def optimize_policy():
    from pi_zero_pytorch import (
        Agent,
        EPO,
    )

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 32,
        depth = 6,
        heads = 16,
        dim_head = 16,
        mlp_dim = 64,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)

    v = Extractor(v, return_embeddings_only = True)

    model = π0(
        dim = 32,
        vit = v,
        vit_dim = 32,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 20_000,
        policy_optimizable = True
    ).to(device)

    images = torch.randn(2, 3, 2, 256, 256).to(device)
    commands = torch.randint(0, 20_000, (2, 1024)).to(device)

    joint_state = torch.randn(2, 12).to(device)
    actions = torch.randn(2, 32, 6).to(device)

    loss, _ = model(images, commands, joint_state, actions)
    loss.backward()

    # agent

    agent = Agent(model)

    final_action_to_env, (
        actions,
        timesteps,
        sampled_flows,
        log_probs
    ) = agent.actor(
        images,
        commands,
        joint_state,
        trajectory_length = 32,
        steps = 4,
        return_states_for_replay = True
    )

    # actions go out into the environment, rewards are received, generalized advantage calculated with critic values

    advantages = torch.randn(2).to(device)

    # optimize policy with replay tensors from above

    actor_loss = agent.actor.forward_for_policy_loss(
        images,
        commands,
        joint_state,
        actions,
        times = timesteps,
        flow = sampled_flows,
        old_log_probs = log_probs,
        advantages = advantages,
    )

    actor_loss.backward()
