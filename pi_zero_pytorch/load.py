import json
import torch
from pathlib import Path
from torch import nn, tensor, cat
from safetensors import safe_open
from einops import rearrange
from tqdm import tqdm

# helpers

def download_pi0_weights(
    local_dir,
    repo_id = None
):
    """
    Download weights from either HuggingFace or Google Cloud Storage.
    
    Args:
        local_dir: Local directory to save weights
        repo_id: If provided, downloads from HuggingFace hub.
                 If None and local_dir starts with 'gs://', downloads from GCS.
                 Otherwise, auto-detects from folder name.
    """
    local_dir = str(local_dir)
    
    # Check if downloading from Google Cloud Storage
    if local_dir.startswith('gs://'):
        try:
            import subprocess
            print(f'Downloading from Google Cloud Storage: {local_dir}')
            
            # Extract the actual local path from the folder name
            gcs_path = local_dir
            local_path = Path('checkpoints') / Path(gcs_path).name
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Use gsutil to download
            subprocess.run(
                ['gsutil', '-m', 'cp', '-r', f'{gcs_path}/*', str(local_path)],
                check=True
            )
            print(f'Downloaded to {local_path}')
            return local_path
            
        except FileNotFoundError:
            raise ImportError('gsutil not found. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Failed to download from GCS: {e}')
    
    # HuggingFace download
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError('Please install huggingface_hub to download weights (pip install huggingface_hub)')

    # Auto-determine repo_id from folder name if not specified
    if repo_id is None:
        folder_name = Path(local_dir).name
        repo_id = f'lerobot/{folder_name}'
    
    print(f'Downloading weights from HuggingFace: {repo_id} to {local_dir}...')
    
    Path(local_dir).mkdir(parents = True, exist_ok = True)
    
    snapshot_download(
        repo_id = repo_id,
        local_dir = local_dir,
        allow_patterns = ['config.json', 'model.safetensors']
    )
    
    print('Download complete.')
    return Path(local_dir)

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
        norm_eps = 1e-6,
        ff_expand_factor = state_ff_expand,
        action_ff_expand_factor = action_ff_expand,
        dim_time_cond = ge['dim'],
        num_action_register_tokens = 0,
        time_mlp_depth = 2,
        time_sinusoidal = True,
        time_min_period = 0.004,
        time_max_period = 4.0,
        time_infused_action_tokens = True,
        layer_time_cond = pi0_config.get('type') == 'pi05' or pi0_config.get('pi05', False),
        token_scale = pg['dim'] ** 0.5,
        activation = 'gelu_pytorch_tanh',
        pi05 = pi0_config.get('type') == 'pi05' or pi0_config.get('pi05', False),
        action_dit_norm_all_linears = False,
    )

def build_converted_state_dict(pi_weights, pz_state):
    arch = get_pi0_architecture()
    pg, ge = arch['paligemma'], arch['gemma_expert']

    # projections

    maps = {
        'action_in_proj.weight': 'to_action_tokens.weight',
        'action_in_proj.bias': 'to_action_tokens.bias',
        'action_out_proj.weight': 'actions_to_pred_flow.weight',
        'action_out_proj.bias': 'actions_to_pred_flow.bias',
        'state_proj.weight': 'to_joint_state_tokens.weight',
        'state_proj.bias': 'to_joint_state_tokens.bias'
    }

    pi_keys = set(pi_weights.keys())

    for pi_k, pz_k in maps.items():
        if pi_k in pi_keys and pz_k in pz_state:
            pz_state[pz_k].copy_(pi_weights.get_tensor(pi_k))

    # time conditioning (AdaRMS for PI0.5)
    if 'time_mlp_in.weight' in pi_keys:
        pz_state['to_time_cond.1.layers.0.0.weight'].copy_(pi_weights.get_tensor('time_mlp_in.weight'))
        pz_state['to_time_cond.1.layers.0.0.bias'].copy_(pi_weights.get_tensor('time_mlp_in.bias'))
    if 'time_mlp_out.weight' in pi_keys:
        pz_state['to_time_cond.1.layers.1.weight'].copy_(pi_weights.get_tensor('time_mlp_out.weight'))
        pz_state['to_time_cond.1.layers.1.bias'].copy_(pi_weights.get_tensor('time_mlp_out.bias'))

    # Action-time fusion (for PI0 style)
    if 'action_time_mlp_in.weight' in pi_keys:
        if 'to_action_time_fuse.layers.0.0.weight' in pz_state:
            pz_state['to_action_time_fuse.layers.0.0.weight'].copy_(pi_weights.get_tensor('action_time_mlp_in.weight'))
            pz_state['to_action_time_fuse.layers.0.0.bias'].copy_(pi_weights.get_tensor('action_time_mlp_in.bias'))
            pz_state['to_action_time_fuse.layers.1.weight'].copy_(pi_weights.get_tensor('action_time_mlp_out.weight'))
            pz_state['to_action_time_fuse.layers.1.bias'].copy_(pi_weights.get_tensor('action_time_mlp_out.bias'))

    # transformer layers

    for i in tqdm(range(pg['num_layers']), desc = 'converting state layers'):
        pi_p = f'paligemma_with_expert.paligemma.model.language_model.layers.{i}'
        pi_e = f'paligemma_with_expert.gemma_expert.model.layers.{i}'
        pz_p = f'layers.{i}'

        # state path (paligemma)

        # GemmaRMSNorm and PiZero's RMSNorm both use: output * (1.0 + weight)
        # So weights are directly compatible, no adjustment needed
        pz_state[f'{pz_p}.0.rmsnorm.weight'].copy_(pi_weights.get_tensor(f'{pi_p}.input_layernorm.weight'))
        pz_state[f'{pz_p}.1.norm.weight'].copy_(pi_weights.get_tensor(f'{pi_p}.post_attention_layernorm.weight'))

        q, k, v = [pi_weights.get_tensor(f'{pi_p}.self_attn.{x}_proj.weight') for x in ('q', 'k', 'v')]
        pz_state[f'{pz_p}.0.to_qkv.weight'].copy_(cat([q, k, v], dim = 0))
        pz_state[f'{pz_p}.0.to_out.weight'].copy_(pi_weights.get_tensor(f'{pi_p}.self_attn.o_proj.weight'))

        gate, up = [pi_weights.get_tensor(f'{pi_p}.mlp.{x}_proj.weight') for x in ('gate', 'up')]
        pz_state[f'{pz_p}.1.proj_in.weight'].copy_(cat([gate, up], dim = 0))
        pz_state[f'{pz_p}.1.proj_out.weight'].copy_(pi_weights.get_tensor(f'{pi_p}.mlp.down_proj.weight'))

        # action path (gemma expert) - cond_layers norms
        
        has_adaptive_norms = f'{pi_e}.input_layernorm.dense.weight' in pi_keys
        
        if has_adaptive_norms:
            # AdaptiveRMSNorm
            # input_layernorm -> cond_layers[i][0]
            pz_state[f'cond_layers.{i}.0.norm.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.input_layernorm.weight'))
            pz_state[f'cond_layers.{i}.0.to_modulation.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.input_layernorm.dense.weight'))
            pz_state[f'cond_layers.{i}.0.to_modulation.bias'].copy_(pi_weights.get_tensor(f'{pi_e}.input_layernorm.dense.bias'))
            # post_attention_layernorm -> cond_layers[i][1] (was index 2 in old 4-element structure)
            pz_state[f'cond_layers.{i}.1.norm.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.post_attention_layernorm.weight'))
            pz_state[f'cond_layers.{i}.1.to_modulation.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.post_attention_layernorm.dense.weight'))
            pz_state[f'cond_layers.{i}.1.to_modulation.bias'].copy_(pi_weights.get_tensor(f'{pi_e}.post_attention_layernorm.dense.bias'))
        else:
            # plain RMSNorm
            # input_layernorm -> cond_layers[i][0]
            pz_state[f'cond_layers.{i}.0.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.input_layernorm.weight'))
            # post_attention_layernorm -> cond_layers[i][1]
            pz_state[f'cond_layers.{i}.1.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.post_attention_layernorm.weight'))

        # action attention weights
        aq, ak, av = [pi_weights.get_tensor(f'{pi_e}.self_attn.{x}_proj.weight') for x in ('q', 'k', 'v')]
        pz_state[f'{pz_p}.0.to_actions_qkv.weight'].copy_(cat([aq, ak, av], dim = 0))
        pz_state[f'{pz_p}.0.to_actions_out.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.self_attn.o_proj.weight'))

        # action MLP weights
        agate, aup = [pi_weights.get_tensor(f'{pi_e}.mlp.{x}_proj.weight') for x in ('gate', 'up')]
        pz_state[f'{pz_p}.2.proj_in.weight'].copy_(cat([agate, aup], dim = 0))
        pz_state[f'{pz_p}.2.proj_out.weight'].copy_(pi_weights.get_tensor(f'{pi_e}.mlp.down_proj.weight'))

    # final norm and rotary

    pz_state['final_norm.weight'].copy_(pi_weights.get_tensor('paligemma_with_expert.paligemma.model.language_model.norm.weight'))
    
    expert_norm_p = 'paligemma_with_expert.gemma_expert.model.norm'
    
    # Check if final_actions_norm is Adaptive (has .norm submodule) or standard RMSNorm
    if 'final_actions_norm.norm.weight' in pz_state:
        # AdaptiveRMSNorm
        pz_state['final_actions_norm.norm.weight'].copy_(pi_weights.get_tensor(f'{expert_norm_p}.weight'))
        
        if f'{expert_norm_p}.dense.weight' in pi_keys:
            pz_state['final_actions_norm.to_modulation.weight'].copy_(pi_weights.get_tensor(f'{expert_norm_p}.dense.weight'))
            pz_state['final_actions_norm.to_modulation.bias'].copy_(pi_weights.get_tensor(f'{expert_norm_p}.dense.bias'))
    else:
        # Standard RMSNorm
        pz_state['final_actions_norm.weight'].copy_(pi_weights.get_tensor(f'{expert_norm_p}.weight'))

    # lm head

    pi_head = 'paligemma_with_expert.paligemma.lm_head.weight'
    if pi_head in pi_keys:
        if 'state_to_logits.weight' in pz_state: pz_state['state_to_logits.weight'].copy_(pi_weights.get_tensor(pi_head))
        if 'token_emb.weight' in pz_state: pz_state['token_emb.weight'].copy_(pi_weights.get_tensor(pi_head))

    # vision encoder
    
    vit_pz_keys = [k for k in pz_state.keys() if k.startswith('vit.layers.') and k.endswith('.0.to_qkv.weight')]
    num_vit_layers = len(vit_pz_keys)

    if num_vit_layers > 0:
        vi_p = 'paligemma_with_expert.paligemma.model.vision_tower.vision_model'

        # patch embedding
        pz_state['vit.to_patch_embed.0.weight'].copy_(pi_weights.get_tensor(f'{vi_p}.embeddings.patch_embedding.weight'))
        pz_state['vit.to_patch_embed.0.bias'].copy_(pi_weights.get_tensor(f'{vi_p}.embeddings.patch_embedding.bias'))

        # position embedding
        pz_state['vit.pos_embed'].copy_(pi_weights.get_tensor(f'{vi_p}.embeddings.position_embedding.weight'))

        # transformer layers
        for i in tqdm(range(num_vit_layers), desc = 'converting vision layers'):
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
