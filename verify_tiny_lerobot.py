# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "einops",
#   "lerobot",
#   "beartype",
#   "accelerate",
#   "pi-zero-pytorch"
# ]
# ///

import os
import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# mock transformers check before lerobot imports

mock_check = MagicMock()
mock_check.check_whether_transformers_replace_is_installed_correctly.return_value = True
sys.modules['transformers.models.siglip.check'] = mock_check

import lerobot.policies.pi0.modeling_pi0 as modeling_pi0
modeling_pi0.check_whether_transformers_replace_is_installed_correctly = lambda: True

from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch, GemmaConfig, PaliGemmaWithExpertModel
from lerobot.policies.pi0.configuration_pi0 import PI0Config

from transformers import CONFIG_MAPPING
from einops import rearrange

from pi_zero_pytorch.pi_zero import PiZero, SigLIP

# constants

DIM = 64
DEPTH = 1
HEADS = 4
HEAD_DIM = 16
KV_HEADS = 1
MLP_DIM = 256
VOCAB_SIZE = 257152
PATCH_SIZE = 16
IMAGE_SIZE = 224

SIGLIP_DIM = 64
SIGLIP_MLP_DIM = 128
SIGLIP_HEADS = 4

# helpers

def exists(v):
    return v is not None

def copy_weight(dst, src):
    dst.data.copy_(src.data)

def copy_linear(dst, src):
    copy_weight(dst.weight, src.weight)
    if exists(dst.bias) and exists(src.bias):
        copy_weight(dst.bias, src.bias)

# model creation

def create_tiny_lerobot():
    vlm_cfg = GemmaConfig(width = DIM, depth = DEPTH, mlp_dim = MLP_DIM, num_heads = HEADS, num_kv_heads = KV_HEADS, head_dim = HEAD_DIM)
    exp_cfg = GemmaConfig(width = DIM, depth = DEPTH, mlp_dim = MLP_DIM, num_heads = HEADS, num_kv_heads = KV_HEADS, head_dim = HEAD_DIM)
    
    modeling_pi0.get_gemma_config = lambda v: vlm_cfg if 'paligemma' in v else exp_cfg
    
    orig_init = PaliGemmaWithExpertModel.__init__
    
    def patched_init(self, vlm_config, action_expert_config, use_adarms = None, precision = "float32"):
        nn.Module.__init__(self)
        vlm_cfg_hf = CONFIG_MAPPING["paligemma"]()
        vlm_cfg_hf._vocab_size = VOCAB_SIZE
        vlm_cfg_hf.image_token_index = VOCAB_SIZE
        vlm_cfg_hf.text_config.hidden_size = vlm_config.width
        vlm_cfg_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_cfg_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_cfg_hf.text_config.head_dim = vlm_config.head_dim
        vlm_cfg_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_cfg_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_cfg_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_cfg_hf.text_config.torch_dtype = "float32"
        vlm_cfg_hf.text_config.vocab_size = VOCAB_SIZE
        
        vlm_cfg_hf.vision_config.hidden_size = SIGLIP_DIM
        vlm_cfg_hf.vision_config.intermediate_size = SIGLIP_MLP_DIM
        vlm_cfg_hf.vision_config.num_hidden_layers = DEPTH
        vlm_cfg_hf.vision_config.num_attention_heads = SIGLIP_HEADS
        vlm_cfg_hf.vision_config.image_size = IMAGE_SIZE
        vlm_cfg_hf.vision_config.patch_size = PATCH_SIZE
        vlm_cfg_hf.vision_config.projection_dim = DIM
        vlm_cfg_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_cfg_hf.vision_config.torch_dtype = "float32"
        
        exp_cfg_hf = CONFIG_MAPPING["gemma"](
            head_dim = action_expert_config.head_dim,
            hidden_size = action_expert_config.width,
            intermediate_size = action_expert_config.mlp_dim,
            num_attention_heads = action_expert_config.num_heads,
            num_hidden_layers = action_expert_config.depth,
            num_key_value_heads = action_expert_config.num_kv_heads,
            vocab_size = VOCAB_SIZE,
            hidden_activation = "gelu_pytorch_tanh",
            torch_dtype = "float32"
        )
        
        from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
        from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
        
        self.paligemma = PaliGemmaForConditionalGeneration(config = vlm_cfg_hf)
        self.gemma_expert = GemmaForCausalLM(config = exp_cfg_hf)
        self.gemma_expert.model.embed_tokens = None
    
    PaliGemmaWithExpertModel.__init__ = patched_init
    
    config = PI0Config()
    config.max_action_dim = 32
    config.max_state_dim = 32
    config.chunk_size = 50
    config.dtype = 'float32'
    config.device = 'cpu'
    
    model = PI0Pytorch(config)
    PaliGemmaWithExpertModel.__init__ = orig_init
    
    model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear = nn.Linear(SIGLIP_DIM, DIM)
    return model.eval()

def create_tiny_pizero():
    vit = SigLIP(
        image_size = IMAGE_SIZE,
        patch_size = PATCH_SIZE,
        dim = SIGLIP_DIM,
        depth = DEPTH,
        heads = SIGLIP_HEADS,
        mlp_dim = SIGLIP_MLP_DIM,
        activation = 'gelu_tanh'
    )
    
    model = PiZero(
        dim = DIM,
        num_tokens = VOCAB_SIZE,
        dim_action_input = 32,
        dim_joint_state = 32,
        dim_action = DIM,
        depth = DEPTH,
        dim_head = HEAD_DIM,
        heads = HEADS,
        kv_heads = KV_HEADS,
        ff_expand_factor = (MLP_DIM / DIM * 1.5),
        action_ff_expand_factor = (MLP_DIM / DIM * 1.5),
        time_sinusoidal = True,
        time_infused_action_tokens = True,
        layer_time_cond = False,
        vit = vit,
        vit_dim = SIGLIP_DIM,
        dim_time_cond = DIM,
        time_mlp_depth = 2,
        activation = 'gelu_pytorch_tanh',
        action_dit_norm_all_linears = False,
        num_action_register_tokens = 0
    )
    return model.eval()

# weight sync

def sync_weights(ler, pz):
    # siglip
    l_vit = ler.paligemma_with_expert.paligemma.model.vision_tower.vision_model
    p_vit = pz.vit
    
    copy_weight(p_vit.to_patch_embed[0].weight, l_vit.embeddings.patch_embedding.weight)
    copy_weight(p_vit.to_patch_embed[0].bias, l_vit.embeddings.patch_embedding.bias)
    copy_weight(p_vit.pos_embed, l_vit.embeddings.position_embedding.weight)
    
    for l_layer, (p_attn, p_ff) in zip(l_vit.encoder.layers, p_vit.layers):
        copy_weight(p_attn.norm.weight, l_layer.layer_norm1.weight)
        copy_weight(p_attn.norm.bias, l_layer.layer_norm1.bias)
        copy_weight(p_attn.to_qkv.weight, torch.cat([l_layer.self_attn.q_proj.weight, l_layer.self_attn.k_proj.weight, l_layer.self_attn.v_proj.weight], dim = 0))
        copy_weight(p_attn.to_qkv.bias, torch.cat([l_layer.self_attn.q_proj.bias, l_layer.self_attn.k_proj.bias, l_layer.self_attn.v_proj.bias], dim = 0))
        copy_linear(p_attn.to_out, l_layer.self_attn.out_proj)
        copy_weight(p_ff.norm.weight, l_layer.layer_norm2.weight)
        copy_weight(p_ff.norm.bias, l_layer.layer_norm2.bias)
        copy_linear(p_ff.proj_in, l_layer.mlp.fc1)
        copy_linear(p_ff.proj_out, l_layer.mlp.fc2)
    
    copy_weight(p_vit.norm.weight, l_vit.post_layernorm.weight)
    copy_weight(p_vit.norm.bias, l_vit.post_layernorm.bias)

    # common projectors
    l_pg = ler.paligemma_with_expert.paligemma.model
    copy_linear(pz.maybe_to_image_tokens, l_pg.multi_modal_projector.linear)
    copy_weight(pz.token_emb.weight, ler.paligemma_with_expert.paligemma.language_model.embed_tokens.weight)
    copy_linear(pz.to_joint_state_tokens, ler.state_proj)
    copy_linear(pz.to_action_tokens, ler.action_in_proj)
    copy_linear(pz.actions_to_pred_flow, ler.action_out_proj)

    # transformer layers
    p_block = pz.layers[0]
    p_cond = pz.cond_layers[0]
    l_vlm = ler.paligemma_with_expert.paligemma.language_model.layers[0]
    l_exp = ler.paligemma_with_expert.gemma_expert.model.layers[0]

    copy_weight(p_block[0].rmsnorm.weight, l_vlm.input_layernorm.weight)
    copy_weight(p_block[0].to_qkv.weight, torch.cat([l_vlm.self_attn.q_proj.weight, l_vlm.self_attn.k_proj.weight, l_vlm.self_attn.v_proj.weight], dim = 0))
    copy_weight(p_block[0].to_out.weight, l_vlm.self_attn.o_proj.weight)

    copy_weight(p_cond[0].weight, l_exp.input_layernorm.weight)
    copy_weight(p_block[0].to_actions_qkv.weight, torch.cat([l_exp.self_attn.q_proj.weight, l_exp.self_attn.k_proj.weight, l_exp.self_attn.v_proj.weight], dim = 0))
    copy_weight(p_block[0].to_actions_out.weight, l_exp.self_attn.o_proj.weight)

    copy_weight(p_block[1].norm.weight, l_vlm.post_attention_layernorm.weight)
    copy_weight(p_block[1].proj_in.weight, torch.cat([l_vlm.mlp.gate_proj.weight, l_vlm.mlp.up_proj.weight], dim = 0))
    copy_weight(p_block[1].proj_out.weight, l_vlm.mlp.down_proj.weight)

    copy_weight(p_cond[1].weight, l_exp.post_attention_layernorm.weight)
    copy_weight(p_block[2].proj_in.weight, torch.cat([l_exp.mlp.gate_proj.weight, l_exp.mlp.up_proj.weight], dim = 0))
    copy_weight(p_block[2].proj_out.weight, l_exp.mlp.down_proj.weight)

    # final heads
    copy_weight(pz.final_norm.weight, ler.paligemma_with_expert.paligemma.language_model.norm.weight)
    copy_weight(pz.state_to_logits.weight, ler.paligemma_with_expert.paligemma.lm_head.weight)
    copy_weight(pz.final_actions_norm.weight, ler.paligemma_with_expert.gemma_expert.model.norm.weight)
    
    copy_linear(pz.to_action_time_fuse.layers[0][0], ler.action_time_mlp_in)
    copy_linear(pz.to_action_time_fuse.layers[1], ler.action_time_mlp_out)

if __name__ == '__main__':
    torch.manual_seed(42)
    
    # compare
    
    ler = create_tiny_lerobot()
    pz = create_tiny_pizero()
    
    sync_weights(ler, pz)
    
    # inputs
    
    img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    tokens = torch.randint(0, 1000, (1, 48))
    state = torch.randn(1, 32)
    actions = torch.randn(1, 50, 32)
    times = torch.tensor([0.5])
    
    with torch.no_grad():
        # lerobot
        img_masks = [torch.ones(1, dtype = torch.bool)]
        lang_masks = torch.ones(1, 48, dtype = torch.bool)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = ler.embed_prefix([img], img_masks, tokens, lang_masks)
        prefix_att_2d_masks = modeling_pi0.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim = 1) - 1
        prefix_att_2d_masks_4d = ler._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        vlm_out, past_kv = ler.paligemma_with_expert.forward(
            attention_mask = prefix_att_2d_masks_4d,
            position_ids = prefix_position_ids,
            inputs_embeds = [prefix_embs, None],
            use_cache = True,
        )
        
        ler_logits = ler.paligemma_with_expert.paligemma.lm_head(ler.paligemma_with_expert.paligemma.language_model.norm(vlm_out[0]))
        ler_flow = ler.denoise_step(state = state, prefix_pad_masks = prefix_pad_masks, past_key_values = past_kv, x_t = actions, timestep = times)

        # pizero
        pz_logits = pz.forward_only_vision_language(images = img, token_ids = tokens)
        pz_flow = pz(images = img, token_ids = tokens, joint_state = state, actions = actions, times = times, return_actions_flow = True)
    
    # asserts
    
    logits_diff = (pz_logits - ler_logits).abs().max().item()
    flow_diff = (pz_flow - ler_flow).abs().max().item()
    
    print(f'logits max diff: {logits_diff:.2e}')
    print(f'flow max diff: {flow_diff:.2e}')
    
    assert logits_diff < 5e-5, f'logits divergence: {logits_diff}'
    assert flow_diff < 5e-5, f'flow divergence: {flow_diff}'
    
    print('✓ parity success')
