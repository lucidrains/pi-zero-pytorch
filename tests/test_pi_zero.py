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
@param('model_predict_output', ('flow', 'clean'))
def test_pi_zero_with_vit(
    only_vlm: bool,
    num_residual_streams: int,
    inpaint_with_frozen_actions: bool,
    action_dit_norm_all_linears: bool,
    task_status_loss: bool,
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

    task_status = torch.randint(0, 3, (2,)) if task_status_loss else None

    loss, _ = model(images, commands, joint_state, actions, task_status = task_status)
    loss.backward()

    # maybe inpaint

    frozen_actions = None
    if inpaint_with_frozen_actions:
        frozen_actions = actions[:, -3:]

    # after much training

    sampled_actions = model(images, commands, joint_state, trajectory_length = 32, frozen_actions = frozen_actions, return_frozen_actions_with_sampled = True) # (1, 32, 6)

    assert sampled_actions.shape == (2, 32, 6)

@param('num_latent_genes', (1, 16))
@param('model_predict_output,use_spo', (('flow', False), ('flow', True), ('clean', True)))
def test_policy_optimization(
    num_latent_genes,
    model_predict_output,
    use_spo
):

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
        policy_optimizable = True,
        model_predict_output = model_predict_output,
        use_spo = use_spo
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

    epo = EPO(
        agent,
        mock_env,
        accelerate_kwargs = dict(
            cpu = True
        )
    )

    memories = epo.gather_experience_from_env(steps = 10)

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
        policy_optimizable = True,
    ).to(device)

    # for parallelism
    # $ accelerate config
    # $ accelerate launch <evolve.py>

    model.evolve(
        environment = lambda noised_model: torch.randint(0, int(1e6), ()), # some simulation
        noise_population_size = 4,
        num_generations = 1
    )
