import json
import torch
from pathlib import Path
from torch import nn, tensor, cat
from safetensors import safe_open
from einops import rearrange
from tqdm import tqdm

# helpers

def download_pi0_weights(
    local_dir = 'checkpoints/pi0_base',
    repo_id = 'lerobot/pi0_base'
):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError('Please install huggingface_hub to download weights (pip install huggingface_hub)')

    print(f'Downloading weights from {repo_id} to {local_dir}...')
    
    Path(local_dir).mkdir(parents = True, exist_ok = True)
    
    snapshot_download(
        repo_id = repo_id,
        local_dir = local_dir,
        allow_patterns = ['config.json', 'model.safetensors']
    )
    
    print('Download complete.')

# architecture

def get_pi0_architecture():
    # PaliGemma 2B + Gemma Expert 300M
    return dict(
        paligemma = dict(
            dim = 2048,
            num_query_heads = 8,
            num_kv_heads = 1,
            dim_head = 256,
            mlp_hidden = 16384,
            num_layers = 18
        ),
        gemma_expert = dict(
            dim = 1024,
            num_query_heads = 8,
            num_kv_heads = 1,
            dim_head = 256,
            mlp_hidden = 4096,
            num_layers = 18
        ),
        vocab_size = 257152
    )

def create_pizero_config_for_pi0(pi0_config):
    arch = get_pi0_architecture()
    pg, ge = arch['paligemma'], arch['gemma_expert']

    # SwiGLU MLP: hidden = dim * expand * 2/3
    state_ff_expand = (pg['mlp_hidden'] * 3) / (pg['dim'] * 2)
    action_ff_expand = (ge['mlp_hidden'] * 3) / (ge['dim'] * 2)

    return dict(
        dim = pg['dim'],
        dim_action = ge['dim'],
        num_tokens = arch['vocab_size'],
        dim_action_input = pi0_config['max_action_dim'],
        dim_joint_state = pi0_config['max_state_dim'],
        depth = pg['num_layers'],
        dim_head = pg['dim_head'],
        heads = pg['num_query_heads'],
        kv_heads = pg['num_kv_heads'],
        attn_softclamp_value = 0.,
        final_norm_softclamp_value = 0.,
        norm_eps = 1e-6,
        ff_expand_factor = state_ff_expand,
        action_ff_expand_factor = action_ff_expand,
        dim_time_cond = 1024
    )

def build_converted_state_dict(pi_weights, pz_state):
    arch = get_pi0_architecture()
    pg, ge = arch['paligemma'], arch['gemma_expert']

    # projections

    maps = {
        'action_in_proj.weight': 'to_action_tokens.weight',
        'action_in_proj.bias': 'to_action_tokens.bias',
        'action_out_proj.weight': 'actions_to_pred_flow.weight',
        'state_proj.weight': 'to_joint_state_tokens.weight',
        'state_proj.bias': 'to_joint_state_tokens.bias'
    }

    pi_keys = set(pi_weights.keys())

    for pi_k, pz_k in maps.items():
        if pi_k in pi_keys and pz_k in pz_state:
            pz_state[pz_k].copy_(pi_weights.get_tensor(pi_k))

    # time conditioning

    if 'action_time_mlp_in.weight' in pi_keys:
        pz_state['to_time_cond.1.weight'].copy_(pi_weights.get_tensor('action_time_mlp_in.weight'))
        pz_state['to_time_cond.1.bias'].copy_(pi_weights.get_tensor('action_time_mlp_in.bias'))

    # transformer layers

    for i in tqdm(range(pg['num_layers']), desc = 'converting state layers'):
        pi_p = f'paligemma_with_expert.paligemma.model.language_model.layers.{i}'
        pi_e = f'paligemma_with_expert.gemma_expert.model.layers.{i}'
        pz_p = f'layers.{i}'

        # state path (paligemma)

        pz_state[f'{pz_p}.0.rmsnorm.weight'].copy_(pi_weights.get_tensor(f'{pi_p}.input_layernorm.weight'))
        pz_state[f'{pz_p}.1.norm.weight'].copy_(pi_weights.get_tensor(f'{pi_p}.post_attention_layernorm.weight'))

        q, k, v = [pi_weights.get_tensor(f'{pi_p}.self_attn.{x}_proj.weight') for x in ('q', 'k', 'v')]
        pz_state[f'{pz_p}.0.to_qkv.weight'].copy_(cat([q, k, v], dim = 0))
        pz_state[f'{pz_p}.0.to_out.weight'].copy_(pi_weights.get_tensor(f'{pi_p}.self_attn.o_proj.weight'))

        gate, up = [pi_weights.get_tensor(f'{pi_p}.mlp.{x}_proj.weight') for x in ('gate', 'up')]
        pz_state[f'{pz_p}.1.proj_in.weight'].copy_(cat([up, gate], dim = 0))
        pz_state[f'{pz_p}.1.proj_out.weight'].copy_(pi_weights.get_tensor(f'{pi_p}.mlp.down_proj.weight'))

        # action path (gemma expert)

        aq, ak, av = [pi_weights.get_tensor(f'{pi_e}.self_attn.{x}_proj.weight') for x in ('q', 'k', 'v')]
        pz_state[f'{pz_p}.0.to_actions_qkvg.weight'].copy_(cat([aq, ak, av, torch.zeros_like(aq)], dim = 0))
        pz_state[f'{pz_p}.0.to_actions_out.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.self_attn.o_proj.weight'))

        agate, aup = [pi_weights.get_tensor(f'{pi_e}.mlp.{x}_proj.weight') for x in ('gate', 'up')]
        pz_state[f'{pz_p}.2.proj_in.weight'].copy_(cat([aup, agate], dim = 0))
        pz_state[f'{pz_p}.2.proj_out.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.mlp.down_proj.weight'))

    # final norm and rotary

    pz_state['final_norm.weight'].copy_(pi_weights.get_tensor('paligemma_with_expert.paligemma.model.language_model.norm.weight'))

    # lm head

    pi_head = 'paligemma_with_expert.paligemma.lm_head.weight'
    if pi_head in pi_keys:
        if 'state_to_logits.weight' in pz_state: pz_state['state_to_logits.weight'].copy_(pi_weights.get_tensor(pi_head))
        if 'token_emb.weight' in pz_state: pz_state['token_emb.weight'].copy_(pi_weights.get_tensor(pi_head))

    # vision encoder

    if any(k.startswith('vit.') for k in pz_state.keys()):
        vi_p = 'paligemma_with_expert.paligemma.model.vision_tower.vision_model'

        # patch embedding
        pz_state['vit.to_patch_embed.0.weight'].copy_(pi_weights.get_tensor(f'{vi_p}.embeddings.patch_embedding.weight'))
        pz_state['vit.to_patch_embed.0.bias'].copy_(pi_weights.get_tensor(f'{vi_p}.embeddings.patch_embedding.bias'))

        # position embedding
        pz_state['vit.pos_embed'].copy_(pi_weights.get_tensor(f'{vi_p}.embeddings.position_embedding.weight'))

        # transformer layers
        for i in tqdm(range(27), desc = 'converting vision layers'):
            v_pi = f'{vi_p}.encoder.layers.{i}'
            v_pz = f'vit.layers.{i}'

            # attention
            pz_state[f'{v_pz}.0.norm.weight'].copy_(pi_weights.get_tensor(f'{v_pi}.layer_norm1.weight'))
            pz_state[f'{v_pz}.0.norm.bias'].copy_(pi_weights.get_tensor(f'{v_pi}.layer_norm1.bias'))

            vq, vk, vv = [pi_weights.get_tensor(f'{v_pi}.self_attn.{x}_proj.weight') for x in ('q', 'k', 'v')]
            bq, bk, bv = [pi_weights.get_tensor(f'{v_pi}.self_attn.{x}_proj.bias') for x in ('q', 'k', 'v')]

            pz_state[f'{v_pz}.0.to_qkv.weight'].copy_(cat([vq, vk, vv], dim = 0))
            pz_state[f'{v_pz}.0.to_qkv.bias'].copy_(cat([bq, bk, bv], dim = 0))

            pz_state[f'{v_pz}.0.to_out.weight'].copy_(pi_weights.get_tensor(f'{v_pi}.self_attn.out_proj.weight'))
            pz_state[f'{v_pz}.0.to_out.bias'].copy_(pi_weights.get_tensor(f'{v_pi}.self_attn.out_proj.bias'))

            # feedforward
            pz_state[f'{v_pz}.1.norm.weight'].copy_(pi_weights.get_tensor(f'{v_pi}.layer_norm2.weight'))
            pz_state[f'{v_pz}.1.norm.bias'].copy_(pi_weights.get_tensor(f'{v_pi}.layer_norm2.bias'))
            pz_state[f'{v_pz}.1.proj_in.weight'].copy_(pi_weights.get_tensor(f'{v_pi}.mlp.fc1.weight'))
            pz_state[f'{v_pz}.1.proj_in.bias'].copy_(pi_weights.get_tensor(f'{v_pi}.mlp.fc1.bias'))
            pz_state[f'{v_pz}.1.proj_out.weight'].copy_(pi_weights.get_tensor(f'{v_pi}.mlp.fc2.weight'))
            pz_state[f'{v_pz}.1.proj_out.bias'].copy_(pi_weights.get_tensor(f'{v_pi}.mlp.fc2.bias'))

        # post-layernorm
        pz_state['vit.norm.weight'].copy_(pi_weights.get_tensor(f'{vi_p}.post_layernorm.weight'))
        pz_state['vit.norm.bias'].copy_(pi_weights.get_tensor(f'{vi_p}.post_layernorm.bias'))

        # multimodal projector
        p_pi = 'paligemma_with_expert.paligemma.model.multi_modal_projector.linear'
        pz_state['maybe_to_image_tokens.weight'].copy_(pi_weights.get_tensor(f'{p_pi}.weight'))
        pz_state['maybe_to_image_tokens.bias'].copy_(pi_weights.get_tensor(f'{p_pi}.bias'))

    return pz_state
