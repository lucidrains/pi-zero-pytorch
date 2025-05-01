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
    ).to(device)

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

    images = torch.randn(2, 3, 2, 256, 256).to(device)
    commands = torch.randint(0, 32, (2, 1024)).to(device)

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

def test_policy_optimization():
    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    from pi_zero_pytorch.pi_zero import (
        Agent,
        EPO,
        calc_generalized_advantage_estimate
    )

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
    ).to(device)

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

    images = torch.randn(1, 3, 2, 256, 256).to(device)
    commands = torch.randint(0, 32, (1, 1024)).to(device)

    joint_state = torch.randn(1, 12).to(device)
    actions = torch.randn(1, 32, 6).to(device)

    loss, _ = model(images, commands, joint_state, actions)
    loss.backward()

    # agent

    agent = Agent(model)

    steps = 4

    all_input_tensors = []
    all_replay_tensors = []

    for _ in range(steps):
        input_tensors = [images, commands, joint_state]

        final_action_to_env, replay_tensors = agent.actor(
            *input_tensors,
            trajectory_length = 4,
            steps = 2,
            return_states_for_replay = True
        )

        all_input_tensors.append(input_tensors)
        all_replay_tensors.append(replay_tensors)

    (
        actions,
        timesteps,
        sampled_flows,
        log_probs
    ) = map(torch.cat, zip(*all_replay_tensors))

    (
        images,
        commands,
        joint_state,
    ) = map(torch.cat, zip(*all_input_tensors))


    t = actions.shape[1]

    values, _ = agent.critic(
        repeat(images, 'b ... -> (b t) ...', t = t),
        repeat(commands, 'b ... -> (b t) ...', t = t),
        repeat(joint_state, 'b ... -> (b t) ...', t = t),
        actions = rearrange(actions, 'b t ... -> (b t) ...'),
        times = rearrange(timesteps, 'b t ... -> (b t) ...')
    )

    values = rearrange(values, '(b t) -> t b', t = t)

    # actions go out into the environment, rewards are received, generalized advantage calculated with critic values

    rewards = torch.randn(steps).to(device)
    boundaries = torch.randint(0, 2, (steps,)).to(device)

    rewards = repeat(rewards, 'b -> t b', t = t)
    boundaries = repeat(boundaries, 'b -> t b', t = t)

    advantages = calc_generalized_advantage_estimate(rewards, values.detach(), boundaries, use_accelerated = False)

    advantages = rearrange(advantages, 't b -> b t')

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

    critic_loss = agent.critic.forward_for_critic_loss(
        repeat(images, 'b ... -> (t b) ...', t = t),
        repeat(commands, 'b ... -> (t b) ...', t = t),
        repeat(joint_state, 'b ... -> (t b) ...', t= t),
        rearrange(actions, 'b t ... -> (t b) ...'),
        old_values = values,
        advantages = advantages,
    )

    critic_loss.backward()
