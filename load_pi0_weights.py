import json
import torch
from pathlib import Path
from torch import nn, tensor, cat
from safetensors import safe_open
from collections import OrderedDict
from pi_zero_pytorch import PiZero, SigLIPEncoder

# helpers

def load_pi0_config(path = "checkpoints/pi0_base/config.json"):
    with open(path) as f:
        return json.load(f)

def get_pi0_weights(path = "checkpoints/pi0_base/model.safetensors"):
    weights = OrderedDict()
    with safe_open(path, framework = "pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights

def expand_kv_heads(weights, num_kv, num_q):
    if num_kv == num_q:
        return weights
    dim = weights.shape[1]
    dim_head = weights.shape[0] // num_kv
    weights = weights.view(num_kv, dim_head, dim)
    weights = weights.repeat_interleave(num_q // num_kv, dim = 0)
    return weights.view(-1, dim)

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
        dim_action_input = pi0_config["max_action_dim"],
        dim_joint_state = pi0_config["max_state_dim"],
        depth = pg['num_layers'],
        dim_head = pg['dim_head'],
        heads = pg['num_query_heads'],
        kv_heads = pg['num_kv_heads'],
        ff_expand_factor = state_ff_expand,
        action_ff_expand_factor = action_ff_expand,
        dim_time_cond = 1024
    )


def build_converted_state_dict(pi_weights, pz_state, verbose = True):
    arch = get_pi0_architecture()
    pg, ge = arch['paligemma'], arch['gemma_expert']
    new_state = OrderedDict()
    
    def log(msg):
        if verbose: print(msg)
    
    # projections

    maps = {
        "action_in_proj.weight": "to_action_tokens.weight",
        "action_in_proj.bias": "to_action_tokens.bias",
        "action_out_proj.weight": "actions_to_pred_flow.weight",
        "state_proj.weight": "to_joint_state_tokens.weight",
        "state_proj.bias": "to_joint_state_tokens.bias"
    }
    
    for pi_k, pz_k in maps.items():
        if pi_k in pi_weights and pz_k in pz_state:
            new_state[pz_k] = pi_weights[pi_k]

    # time conditioning

    if "action_time_mlp_in.weight" in pi_weights:
        new_state["to_time_cond.1.weight"] = pi_weights["action_time_mlp_in.weight"]
        new_state["to_time_cond.1.bias"] = pi_weights["action_time_mlp_in.bias"]

    # transformer layers

    for i in range(pg['num_layers']):
        pi_p = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}"
        pi_e = f"paligemma_with_expert.gemma_expert.model.layers.{i}"
        pz_p = f"layers.{i}"

        # state path (paligemma)

        new_state[f"{pz_p}.0.rmsnorm.weight"] = pi_weights[f"{pi_p}.input_layernorm.weight"]
        new_state[f"{pz_p}.1.rmsnorm.weight"] = pi_weights[f"{pi_p}.post_attention_layernorm.weight"]
        
        q, k, v = [pi_weights[f"{pi_p}.self_attn.{x}_proj.weight"] for x in ('q', 'k', 'v')]
        new_state[f"{pz_p}.0.to_qkv.weight"] = cat([q, k, v], dim = 0)
        new_state[f"{pz_p}.0.to_out.weight"] = pi_weights[f"{pi_p}.self_attn.o_proj.weight"]

        gate, up = [pi_weights[f"{pi_p}.mlp.{x}_proj.weight"] for x in ('gate', 'up')]
        new_state[f"{pz_p}.1.proj_in.weight"] = cat([gate, up], dim = 0)
        new_state[f"{pz_p}.1.proj_out.weight"] = pi_weights[f"{pi_p}.mlp.down_proj.weight"]

        # action path (gemma expert)

        aq, ak, av = [pi_weights[f"{pi_e}.self_attn.{x}_proj.weight"] for x in ('q', 'k', 'v')]
        new_state[f"{pz_p}.0.to_actions_qkvg.weight"] = cat([aq, ak, av, torch.zeros_like(aq)], dim = 0)
        new_state[f"{pz_p}.0.to_actions_out.weight"] = pi_weights[f"{pi_e}.self_attn.o_proj.weight"]

        agate, aup = [pi_weights[f"{pi_e}.mlp.{x}_proj.weight"] for x in ('gate', 'up')]
        new_state[f"{pz_p}.2.proj_in.weight"] = cat([agate, aup], dim = 0)
        new_state[f"{pz_p}.2.proj_out.weight"] = pi_weights[f"{pi_e}.mlp.down_proj.weight"]

    # lm head

    pi_head = "paligemma_with_expert.paligemma.lm_head.weight"
    if pi_head in pi_weights:
        if "state_to_logits.weight" in pz_state: new_state["state_to_logits.weight"] = pi_weights[pi_head]
        if "token_emb.weight" in pz_state: new_state["token_emb.weight"] = pi_weights[pi_head]

    # vision encoder

    if any(k.startswith("vit.") for k in pz_state.keys()):
        log("Mapping SigLIP weights...")
        vi_p = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
        
        # patch and pos

        pw = pi_weights[f"{vi_p}.embeddings.patch_embedding.weight"]
        new_state["vit.vit.to_patch_embedding.2.weight"] = pw.permute(0, 2, 3, 1).reshape(pw.shape[0], -1)
        new_state["vit.vit.to_patch_embedding.2.bias"] = pi_weights[f"{vi_p}.embeddings.patch_embedding.bias"]
        
        for n in (1, 3):
            new_state[f"vit.vit.to_patch_embedding.{n}.weight"] = torch.ones_like(pz_state[f"vit.vit.to_patch_embedding.{n}.weight"])
            new_state[f"vit.vit.to_patch_embedding.{n}.bias"] = torch.zeros_like(pz_state[f"vit.vit.to_patch_embedding.{n}.bias"])
        
        pos = pi_weights[f"{vi_p}.embeddings.position_embedding.weight"]
        pz_pos = pz_state["vit.vit.pos_embedding"].clone()
        pz_pos[1:] = pos
        new_state["vit.vit.pos_embedding"] = pz_pos

        # vit layers

        for i in range(27):
            v_pi = f"{vi_p}.encoder.layers.{i}"
            v_pz = f"vit.vit.transformer.layers.{i}"
            
            new_state[f"{v_pz}.0.norm.weight"] = pi_weights[f"{v_pi}.layer_norm1.weight"]
            new_state[f"{v_pz}.0.norm.bias"] = pi_weights[f"{v_pi}.layer_norm1.bias"]
            new_state[f"{v_pz}.1.net.0.weight"] = pi_weights[f"{v_pi}.layer_norm2.weight"]
            new_state[f"{v_pz}.1.net.0.bias"] = pi_weights[f"{v_pi}.layer_norm2.bias"]
            
            vq, vk, vv = [pi_weights[f"{v_pi}.self_attn.{x}_proj.weight"] for x in ('q', 'k', 'v')]
            new_state[f"{v_pz}.0.to_qkv.weight"] = cat([vq, vk, vv], dim = 0)
            
            new_state[f"{v_pz}.0.to_out.0.weight"] = pi_weights[f"{v_pi}.self_attn.out_proj.weight"]
            new_state[f"{v_pz}.0.to_out.0.bias"] = pi_weights[f"{v_pi}.self_attn.out_proj.bias"]
            
            for n, x in ((1, 'fc1'), (4, 'fc2')):
                new_state[f"{v_pz}.1.net.{n}.weight"] = pi_weights[f"{v_pi}.mlp.{x}.weight"]
                new_state[f"{v_pz}.1.net.{n}.bias"] = pi_weights[f"{v_pi}.mlp.{x}.bias"]

        new_state["vit.vit.transformer.norm.weight"] = pi_weights[f"{vi_p}.post_layernorm.weight"]
        new_state["vit.vit.transformer.norm.bias"] = pi_weights[f"{vi_p}.post_layernorm.bias"]

        p_pi = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
        new_state["maybe_to_image_tokens.weight"] = pi_weights[f"{p_pi}.weight"]
        new_state["maybe_to_image_tokens.bias"] = pi_weights[f"{p_pi}.bias"]

    return new_state

def load_pi0_weights_into_pizero(
    checkpoint_path = "checkpoints/pi0_base/model.safetensors",
    config_path = "checkpoints/pi0_base/config.json",
    verbose = True
):
    from pi_zero_pytorch import PiZero
    from vit_pytorch import ViT
    
    config = load_pi0_config(config_path)
    pi_weights = get_pi0_weights(checkpoint_path)
    
    pz_config = create_pizero_config_for_pi0(config)
    
    pz_config['vit'] = SigLIPEncoder()
    pz_config['vit_dim'] = 1152

    model = PiZero(**pz_config)
    pz_state = model.state_dict()
    
    new_state = build_converted_state_dict(pi_weights, pz_state, verbose = verbose)
    model.load_state_dict(new_state, strict = False)
    
    return model, new_state

if __name__ == "__main__":
    checkpoint_path = "checkpoints/pi0_base/model.safetensors"
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        exit()
        
    model, loaded = load_pi0_weights_into_pizero()
    
    total = sum(p.numel() for p in model.parameters())
    loaded_params = sum(model.state_dict()[k].numel() for k in loaded.keys())
    
    print(f"\nTotal parameters: {total:,}")
    print(f"Loaded parameters: {loaded_params:,}")
    print(f"Percentage loaded: {100 * loaded_params / total:.1f}%")
