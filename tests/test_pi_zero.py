import pytest
param = pytest.mark.parametrize

import torch
from pi_zero_pytorch import π0
from einops import repeat, rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@param('only_vlm', (True, False))
@param('num_residual_streams', (1, 4))
@param('inpaint_with_frozen_actions', (False, True))
@param('action_dit_norm_all_linears', (False, True))
@param('task_status_loss', (False, True))
@param('advantage_condition', (False, True))
@param('model_predict_output', ('flow', 'clean'))
def test_pi_zero_with_vit(
    only_vlm: bool,
    num_residual_streams: int,
    inpaint_with_frozen_actions: bool,
    action_dit_norm_all_linears: bool,
    task_status_loss: bool,
    advantage_condition,
    model_predict_output
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
        num_advantage_tokens = 2 if advantage_condition else 0,
        sample_soft_mask_lens = (2, 1, 29),
        action_dit_norm_all_linears = action_dit_norm_all_linears,
        num_residual_streams = num_residual_streams,
        model_predict_output = model_predict_output
    ).to(device)

    images = torch.randn(2, 3, 2, 256, 256)
    commands = torch.randint(0, 32, (2, 1024))

    if only_vlm:
        vlm_logits = model.forward_only_vision_language(images, commands)
        assert vlm_logits.ndim == 3
        return

    joint_state = torch.randn(2, 12)
    actions = torch.randn(2, 32, 6)

    # for pi0.6

    advantage_ids = None
    if advantage_condition:
        advantage_ids = torch.randint(0, 2, (2,))

    # task status

    task_status = torch.randint(0, 3, (2,)) if task_status_loss else None

    loss, _ = model(images, commands, joint_state, actions, task_status = task_status, advantage_ids = advantage_ids)
    loss.backward()

    # maybe inpaint

    frozen_actions = None
    if inpaint_with_frozen_actions:
        frozen_actions = actions[:, -3:]

    # after much training

    inference_advantage_id = 1 if advantage_condition else None # fixed to always advantage positive

    sampled_actions = model(images, commands, joint_state, trajectory_length = 32, frozen_actions = frozen_actions, advantage_ids = inference_advantage_id, return_frozen_actions_with_sampled = True) # (1, 32, 6)

    assert sampled_actions.shape == (2, 32, 6)

@param('num_latent_genes', (1, 16))
@param('model_predict_output', ('flow', 'clean'))
@param('use_spo', (False, True))
@param('use_asymmetric_spo', (False, True))
def test_flow_policy_optimization(
    num_latent_genes,
    model_predict_output,
    use_spo,
    use_asymmetric_spo
):

    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    from pi_zero_pytorch.pi_zero import (
        Agent,
        EFPO,
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
        model_predict_output = model_predict_output,
        use_spo = use_spo,
        use_asymmetric_spo = use_asymmetric_spo
    ).to(device)

    images = torch.randn(2, 3, 2, 256, 256)
    commands = torch.randint(0, 32, (2, 1024))

    joint_state = torch.randn(2, 12)
    actions = torch.randn(2, 32, 6)

    loss, _ = model(images, commands, joint_state, actions)
    loss.backward()

    # agent

    agent = Agent(
        model,
        num_latent_genes = num_latent_genes
    )

    mock_env = Env((256, 256), 2, 32, 1024, 12)

    epo = EFPO(
        agent,
        cpu = True,
    )

    memories = epo.gather_experience_from_env(mock_env, steps = 4)

    epo.learn_agent(memories, batch_size = 2)

def test_evo_strat():
    from x_evolution import EvoStrategy

    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    from pi_zero_pytorch.pi_zero import (
        Agent
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
    ).to(device)

    # for parallelism
    # $ accelerate config
    # $ accelerate launch <evolve.py>

    model.evolve(
        environment = lambda noised_model: torch.randint(0, int(1e6), ()), # some simulation
        noise_population_size = 4,
        num_generations = 1,
        params_to_optimize = None
    )

def test_soft_mask():
    from pi_zero_pytorch.pi_zero import create_soft_inpaint_mask

    soft_mask = create_soft_inpaint_mask(24, 5, 5)

    assert (soft_mask[:5] == 1.).all() and (soft_mask[-5:] == 0.).all()
    assert ((soft_mask[5:-5] > 0.) & (soft_mask[5:-5] < 1.)).all()

def test_self_contained_rtc_guidance():
    from pi_zero_pytorch import RTCGuidance

    model = π0(
        dim = 512,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 20_000,
        action_dit_norm_all_linears = True
    )

    vision = torch.randn(1, 1024, 512)
    commands = torch.randint(0, 20_000, (1, 1024))
    joint_state = torch.randn(1, 12)
    times = torch.rand(1,)
    actions = torch.randn(1, 32, 6)

    rtc_guidance = RTCGuidance()

    model_forward_with_guidance = rtc_guidance.with_model_and_frozen_actions(
        model,
        frozen_actions = actions,
        soft_mask = (24, 3, 5),
        input_time_arg_name = 'times',
        input_noised_actions_arg_name = 'actions',
        add_guidance_to_flow = True
    )

    flow_with_guidance = model_forward_with_guidance(vision, commands, joint_state, actions, times = times, return_actions_flow = True)

    assert flow_with_guidance.shape == actions.shape

@param('critic_use_discrete_bins', (False, True))
@param('value_clip', (False, True))
def test_value(
    critic_use_discrete_bins,
    value_clip
):

    model = π0(
        dim = 512,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 20_000,
        is_critic = True,
        critic_use_discrete_bins = critic_use_discrete_bins
    )

    vision = torch.randn(1, 1024, 512)
    commands = torch.randint(0, 20_000, (1, 1024))
    joint_state = torch.randn(1, 12)
    times = torch.rand(1,)
    actions = torch.randn(1, 32, 6)

    values, logits = model(vision, commands, joint_state, actions, times = times, return_actions_flow = True)

    assert values.shape == (1,)
    assert logits.shape == (1, 50)

    loss = model.forward_for_critic_loss(vision, commands, joint_state, actions, old_values = values, advantages = values, value_clip = value_clip)

    assert loss.numel() == 1

def test_pi_zero_six():
    from pi_zero_pytorch import π0, PiZeroSix

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
        vit = v,
        vit_dim = 32,
        dim = 512,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 20_000,
        num_advantage_tokens = 2,
        num_tasks = 10
    )

    # you'll want to supply your own environment

    from pi_zero_pytorch.mock_env import Env
    mock_env = Env((256, 256), 2, 32, 1024, 12)

    # pass your agent and environment to PiZeroSix for learning to be orchestrated

    pi_zero_six = PiZeroSix(model)

    # gather experiences from environment

    experience = pi_zero_six.gather_experience_from_env(mock_env, steps = 10, num_episodes = 3, task_id = 2)

    # labeling

    pi_zero_six.set_episode_fail_(experience, episode_id = 1)

    pi_zero_six.calculate_advantages_(experience)

    pi_zero_six.set_advantage_token_id_(experience)

    pi_zero_six.invalidate_(experience, 1)
