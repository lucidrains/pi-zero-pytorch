from shutil import rmtree

import pytest
param = pytest.mark.parametrize

import torch
from pi_zero_pytorch import π0, ReplayBuffer, JoinedReplayDataset
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
        num_tokens = 1024,
        num_advantage_tokens = 2 if advantage_condition else 0,
        sample_soft_mask_lens = (2, 1, 29),
        action_dit_norm_all_linears = action_dit_norm_all_linears,
        num_residual_streams = num_residual_streams,
        model_predict_output = model_predict_output
    ).to(device)

    images = torch.randn(2, 3, 2, 256, 256)
    commands = torch.randint(0, 1024, (2, 32))

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

    from pi_zero_pytorch.mock import Env

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
        num_tokens = 1024,
        model_predict_output = model_predict_output,
        use_spo = use_spo,
        use_asymmetric_spo = use_asymmetric_spo
    ).to(device)

    images = torch.randn(2, 3, 2, 256, 256)
    commands = torch.randint(0, 1024, (2, 32))

    joint_state = torch.randn(2, 12)
    actions = torch.randn(2, 32, 6)

    loss, _ = model(images, commands, joint_state, actions)
    loss.backward()

    # agent

    agent = Agent(
        model,
        num_latent_genes = num_latent_genes
    )

    mock_env = Env((256, 256), 2, 1024, 32, 12)

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

    from pi_zero_pytorch.mock import Env

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
        num_tokens = 1024,
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

@param('manual_training', (False, True))
def test_pi_zero_six(
    manual_training
):
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
        depth = 1,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 20_000,
        num_advantage_tokens = 2,
        num_tasks = 10
    )

    # you'll want to supply your own environment

    from pi_zero_pytorch.mock import Env
    mock_env = Env((256, 256), 2, 1024, 32, 12)

    # pass your agent and environment to PiZeroSix for learning to be orchestrated

    pi_zero_six = PiZeroSix(model, cpu = True)

    # gather experiences from environment

    experience = pi_zero_six.gather_experience_from_env(mock_env, num_episodes = 3, steps = 4)

    # meta buffer
    
    meta_buffer = ReplayBuffer(
        './meta',
        max_episodes = 3,
        max_timesteps = 4,
        fields = dict(
            value = 'float',
            advantages = 'float',
            returns = 'float',
            advantage_ids = 'int'
        ),
        meta_fields = dict(
            task_id = 'int'
        ),
        overwrite = True
    )

    # initialize meta task ids
    meta_buffer.meta_data['task_id'][:] = -1

    # update buffer values and calculate returns in meta buffer
    pi_zero_six.update_buffer_values_(experience, destination_buffer = meta_buffer)
    pi_zero_six.calculate_return_or_advantages_(experience, destination_buffer = meta_buffer)

    # labeling

    pi_zero_six.set_episode_fail_(experience, episode_id = 1)
    pi_zero_six.set_episode_success_(experience, episode_id = 2)
    pi_zero_six.invalidate_(experience, 1)

    pi_zero_six.invalidate_by_value_threshold_(experience, -100., value_buffer = meta_buffer)

    # joined dataset for training

    joined_dataset = JoinedReplayDataset([pi_zero_six.dataset(experience)], meta_buffer)

    if manual_training:
        # now learn from the experience

        for batch in pi_zero_six.dataloader(joined_dataset):
            loss, *_ = model(**batch)
            loss.backward()
    else:
        pi_zero_six.train_value_network(joined_dataset, num_train_steps = 2)

    pi_zero_six.update_buffer_values_(experience, destination_buffer = meta_buffer)
    pi_zero_six.calculate_return_or_advantages_(experience, destination_buffer = meta_buffer)
    pi_zero_six.set_advantage_token_id_(meta_buffer)

    pi_zero_six.train_policy_network(joined_dataset, num_train_steps = 2)

def test_train_time_rtc():
    from pi_zero_pytorch import π0

    model = π0(
        dim = 512,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 20_000,
        train_time_rtc = True,
        train_time_rtc_max_delay = 4
    )

    vision = torch.randn(1, 1024, 512)
    commands = torch.randint(0, 20_000, (1, 1024))
    joint_state = torch.randn(1, 12)
    actions = torch.randn(1, 32, 6)

    loss, _ = model(vision, commands, joint_state, actions)
    loss.backward()

    # after much training

    sampled_actions = model(vision, commands, joint_state, frozen_actions = actions[:, -3:], trajectory_length = 32) # (1, 32, 6)

    assert sampled_actions.shape == (1, 32 - 3, 6)

@pytest.fixture
def pi_zero_six_workspace():
    from pathlib import Path
    workspace = Path('./test/workspace_pretrain')
    replay_folder = Path('./test/replay_buffer_pretrain')

    if workspace.exists():
        rmtree(workspace, ignore_error = True)

    if replay_folder.exists():
        rmtree(replay_folder, ignore_error = True)

    workspace.mkdir(exist_ok = True, parents = True)
    replay_folder.mkdir(exist_ok = True, parents = True)

    yield workspace, replay_folder

    if workspace.exists():
        rmtree(workspace)

    if replay_folder.exists():
        rmtree(replay_folder)

def test_pi_zero_six_recap(pi_zero_six_workspace):
    from pi_zero_pytorch import PiZeroSix
    from pi_zero_pytorch.mock import create_mock_replay_buffer
    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    workspace, replay_folder = pi_zero_six_workspace

    # mocks
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 32,
        depth = 1,
        heads = 4,
        dim_head = 8,
        mlp_dim = 64
    )
    
    v = Extractor(v, return_embeddings_only = True)
    
    model = π0(
        vit = v,
        vit_dim = 32,
        dim = 64, # small dim for speed
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 1024,
        num_advantage_tokens = 2,
        num_tasks = 2
    )
    
    # mock buffer
    
    mock_pretrain = create_mock_replay_buffer(
        folder = replay_folder,
        max_episodes = 5,
        max_timesteps = 10,
        num_episodes = 2,
        max_task_id = 1,
        dim_action_input = 6,
        joint_dim = 12
    )

    # pi_zero_six
    
    pi_zero_six = PiZeroSix(
        model,
        pretrain_data = mock_pretrain,
        cpu = True,
        workspace_folder = str(workspace)
    )
    
    # run pretrain for generalist

    pi_zero_six.pretrain(
        num_train_steps_actor = 1, # minimum steps
        num_train_steps_critic = 1,
        batch_size = 2
    )

    # check assertions
    
    assert (workspace / 'pretrained-actor.pt').exists()
    assert (workspace / 'pretrained-critic.pt').exists()

    # sft stage

    task_id = next(iter(pi_zero_six.task_id_name))
    task_name = pi_zero_six.task_id_name[task_id]

    pi_zero_six.sft(
        task_id,
        num_train_steps_actor = 1, # minimum steps
        num_train_steps_critic = 1,
        batch_size = 2
    )

    # assert the finetuned actor critic from step 2 exists

    assert (workspace / task_name / "0" / "actor.pt").exists()
    assert (workspace / task_name / "0" / "critic.pt").exists()

    # rollout

    from pi_zero_pytorch.mock import Env
    mock_env = Env((256, 256), 2, 1024, 32, 12)

    experience = pi_zero_six.gather_experience_from_env(mock_env, num_episodes = 3, task_id = task_id)

    # make sure experience is saved

    assert (workspace / task_name / "0" / "data.0").exists()

    # make sure can gather consecutive experiences

    experience = pi_zero_six.gather_experience_from_env(mock_env, num_episodes = 3, task_id = task_id)

    assert (workspace / task_name / "0" / "data.1").exists()

    # complete one iteration of recap

    pi_zero_six.recap_finetune(
        task_id,
        num_train_steps_actor = 1, # minimum steps
        num_train_steps_critic = 1,
        batch_size = 2
    )

    assert (workspace / task_name / "1" / "actor.pt").exists()