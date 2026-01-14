# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "einops",
#   "lerobot",
#   "beartype",
#   "accelerate",
#   "safetensors",
#   "fire",
#   "pi-zero-pytorch"
# ]
# ///

import os
import gc
import sys
import torch
import subprocess
from pathlib import Path

import fire

# constants

CHECKPOINT = 'checkpoints/pi0_base'
REF_PATH = 'lerobot_reference.pt'
CACHE_PATH = 'vision_cache.pt'

# helpers

def cleanup():
    gc.collect()
    gc.collect()

def exists(v):
    return v is not None

def diff(t1, t2):
    return (t1 - t2).abs().max().item()

# lerobot reference generation script (run in isolation)

LEROBOT_SCRIPT = '''
import sys
import torch
import json
from pathlib import Path
from unittest.mock import MagicMock
from safetensors.torch import load_file

# mock

mock_check = MagicMock()
mock_check.check_whether_transformers_replace_is_installed_correctly.return_value = True
sys.modules["transformers.models.siglip.check"] = mock_check

import lerobot.policies.pi0.modeling_pi0 as modeling_pi0
modeling_pi0.check_whether_transformers_replace_is_installed_correctly = lambda: True

from lerobot.policies.pi0.modeling_pi0 import PI0Policy, make_att_2d_masks
from lerobot.policies.pi0.configuration_pi0 import PI0Config

def generate(ref_path, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    
    with open(checkpoint_dir / 'config.json') as f:
        config_data = json.load(f)
        
    for k in ['type', 'device']: config_data.pop(k, None)
    config = PI0Config(**config_data)
    config.device = 'cpu'
    
    policy = PI0Policy(config)
    state_dict = load_file(checkpoint_dir / 'model.safetensors')
    policy.load_state_dict({'model.' + k: v for k, v in state_dict.items()}, strict = False)
    
    # tie embeddings
    w = state_dict['paligemma_with_expert.paligemma.lm_head.weight']
    lm_model = policy.model.paligemma_with_expert.paligemma.language_model
    (lm_model.model if hasattr(lm_model, 'model') else lm_model).embed_tokens.weight.data.copy_(w)

    model = policy.model.eval()
    
    # inputs
    torch.manual_seed(42)
    B = 1
    inputs = {
        'images': torch.randn(B, 3, 224, 224),
        'text_tokens': torch.randint(0, 257152, (B, 48)),
        'joint_state': torch.randn(B, 32),
        'actions': torch.randn(B, 50, 32),
        'times': torch.rand(B)
    }
    
    acts = {'inputs': inputs}
    hooks = []

    def hook(name):
        def fn(m, i, o):
            res = o[0] if isinstance(o, (tuple, list)) else o
            if hasattr(res, 'last_hidden_state'): res = res.last_hidden_state
            acts[name] = res.detach().clone()
        return fn

    pg = model.paligemma_with_expert.paligemma
    hooks.append(pg.model.language_model.norm.register_forward_hook(hook('vlm_hidden')))
    hooks.append(pg.lm_head.register_forward_hook(hook('vlm_logits')))
    hooks.append(model.action_out_proj.register_forward_hook(hook('flow')))

    with torch.no_grad():
        img_masks = [torch.ones(1, dtype = torch.bool)]
        lang_masks = torch.ones(1, 48, dtype = torch.bool)
        
        embs, pad_mask, att_mask = model.embed_prefix([inputs['images']], img_masks, inputs['text_tokens'], lang_masks)
        att_2d_mask = make_att_2d_masks(pad_mask, att_mask)
        pos_ids = torch.cumsum(pad_mask, dim = 1) - 1
        att_4d_mask = model._prepare_attention_masks_4d(att_2d_mask)
        
        _, past_kv = model.paligemma_with_expert.forward(
            attention_mask = att_4d_mask, 
            position_ids = pos_ids, 
            inputs_embeds = [embs, None], 
            use_cache = True
        )
        
        model.denoise_step(state = inputs['joint_state'], prefix_pad_masks = pad_mask, past_key_values = past_kv, x_t = inputs['actions'], timestep = inputs['times'])
        
    for h in hooks: h.remove()
    
    # Compute vlm_logits from vlm_hidden (lm_head hook doesn't fire in this forward path)
    acts['vlm_logits'] = pg.lm_head(acts['vlm_hidden'])
    
    torch.save(acts, ref_path)

if __name__ == "__main__":
    generate(sys.argv[1], sys.argv[2])
'''

class Verifier:
    def generate_ref(self):
        from pi_zero_pytorch.load import download_pi0_weights
        download_pi0_weights(CHECKPOINT)
        
        script = Path('_lerobot_ref.py')
        script.write_text(LEROBOT_SCRIPT)
        
        try:
            subprocess.run([sys.executable, str(script), REF_PATH, CHECKPOINT], check = True)
        finally:
            script.unlink(missing_ok = True)
        cleanup()

    def verify(self, forced_generate_ref = False):
        if not os.path.exists(REF_PATH) or forced_generate_ref:
            self.generate_ref()

        from pi_zero_pytorch.pi_zero import PiZero
        
        ref = torch.load(REF_PATH, weights_only = False)
        inputs = ref['inputs']
        
        model = PiZero.from_checkpoint(CHECKPOINT).cpu().eval()
        
        visual_tokens = None
        if os.path.exists(CACHE_PATH):
            visual_tokens = torch.load(CACHE_PATH, weights_only = True)
            del model.vit
            model.vit = None
            cleanup()
        
        with torch.no_grad():
            # vlm
            pz_logits = model.forward_only_vision_language(
                images = inputs['images'],
                token_ids = inputs['text_tokens'],
                visual_tokens = visual_tokens
            )
            
            if visual_tokens is None and exists(model.vit):
                # cache vision tokens if first time
                def hook(m, i, o):
                    torch.save(o.detach().clone(), CACHE_PATH)
                h = model.vit.register_forward_hook(hook)
                model.forward_only_vision_language(images = inputs['images'], token_ids = inputs['text_tokens'])
                h.remove()
                visual_tokens = torch.load(CACHE_PATH, weights_only = True)
                del model.vit
                model.vit = None
                cleanup()

            # flow
            pz_flow = model(
                images = inputs['images'],
                token_ids = inputs['text_tokens'],
                joint_state = inputs['joint_state'],
                actions = inputs['actions'],
                times = inputs['times'],
                visual_tokens = visual_tokens,
                return_actions_flow = True
            )
        
        # compare
        
        logits_diff = diff(pz_logits, ref['vlm_logits'])
        flow_diff = diff(pz_flow, ref['flow'])
        
        print(f'vlm logits max diff: {logits_diff:.2e}')
        print(f'action flow max diff: {flow_diff:.2e}')
        
        assert logits_diff < 2e-3, f'vlm divergence: {logits_diff}'
        assert flow_diff < 1e-5, f'flow divergence: {flow_diff}'
        
        print('✓ parity success')

if __name__ == '__main__':
    fire.Fire(Verifier)
