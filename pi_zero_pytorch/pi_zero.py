from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import bisect
from copy import deepcopy
from random import random, randrange
from shutil import rmtree
from pathlib import Path

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Callable, Literal

from inspect import signature
from contextlib import contextmanager
from functools import partial, wraps
from collections import namedtuple
from itertools import count

import numpy as np
from numpy import ndarray

import torch
import torch.nn.functional as F
from torch import pi, nn, arange, cat, stack, tensor, Tensor, broadcast_tensors, is_tensor, full, from_numpy
from torch.nn import Module, ModuleList
from torch.distributions.beta import Beta

from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Dataset

from torchdiffeq import odeint

from ema_pytorch import EMA

from adam_atan2_pytorch import AdamAtan2

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, einsum, pack, unpack

from hl_gauss_pytorch import HLGaussLayer

from assoc_scan import AssocScan

from evolutionary_policy_optimization import LatentGenePool

from x_evolution import EvoStrategy

from x_mlps_pytorch import MLP

import tqdm

from memmap_replay_buffer import ReplayBuffer

from accelerate import Accelerator

from pydantic import BaseModel, Field, model_validator

from torch_einops_utils import (
    maybe,
    pad_right_ndim,
    pad_at_dim,
    pad_left_at_dim,
    pack_with_inverse,
    pad_sequence,
    and_masks,
    lens_to_mask
)

# ein notation

# b - batch
# n - sequence
# na - seq of actions
# nt - seq of text tokens
# nv - seq of visual tokens
# ns - seq of additional internal state tokens
# nm - seq of memory tokens
# nfa - seq of frozen actions
# d - dimension
# da - action dimension
# djs - joint state dimension
# c - image channels
# h - image height
# w - image width
# f - image frames
# s - residual streams (hyper connections paper)
# e - episodes
# t - time steps

# token layout for transformer
# vision and language tokens are autoregressive causal mask, actions, interal states + joint bidirectional amongst own tokens, but still autoregressive with respect to other tokens

# [state token groups] [action token groups] -> [autoregressive masking] [bidirectional]
# [external state] [visual tokens] [language tokens] [maybe reward / condition token] [action registers] [joint state + internal state] [actions]

# for an attempt to introduce recurrence, all tokens above can be flanked by read and write memory tokens
# [read memory tokens] [...] [write memory tokens]

# constants

LinearNoBias = partial(nn.Linear, bias = False)

# flex attention related
# https://pytorch.org/blog/flexattention/

flex_attention = None

if torch.cuda.is_available():
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention = torch.compile(flex_attention)

def create_pizero_attn_mask(
    prefix_causal_length,
    mask: Tensor, # bool[b n]
    prefix_bidirectional_length = 0,
    prefix_bidirectional_start = 0,
    discretized_action_length = 0
):
    # the pi-zero attention is a triangular causal mask, but bidirectional attention for the actions at the very right hand side

    def mask_fn(batch_index, head_index, query_index, key_index):
        key_mask = mask[batch_index, key_index]   # variable length states
        causal_mask = query_index >= key_index    # causal

        is_prefix = (
            key_index >= prefix_bidirectional_start and
            key_index < (prefix_bidirectional_start + prefix_bidirectional_length)
        )

        bidirectional_action_mask = (             # bidirectional action mask
            key_index >= prefix_causal_length and
            query_index >= prefix_causal_length
        )

        return key_mask & (is_prefix | (causal_mask & ~action_ignore_discretized_tokens) | bidirectional_action_mask)

    return mask_fn

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def sample(prob):
    return random() < prob

def xnor(x, y):
    return not (x ^ y)

def save_args_kwargs(fn):
    @wraps(fn)
    def decorated(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        return fn(self, *args, **kwargs)

    return decorated

def tree_map_tensor(t, fn):
    return tree_map(lambda el: fn(el) if is_tensor(el) else el, t)

def to_device(t, device):
    return tree_map_tensor(t, lambda el: el.to(device))

def move_input_tensors_to_device(fn):

    @wraps(fn)
    def decorated_fn(self, *args, **kwargs):
        args, kwargs = to_device((args, kwargs), self.device)
        return fn(self, *args, **kwargs)

    return decorated_fn

def temp_batch_dim(fn):

    @wraps(fn)
    def inner(*args, **kwargs):
        args, kwargs = tree_map(lambda t: rearrange(t, '... -> 1 ...') if is_tensor(t) else t, (args, kwargs))

        out = fn(*args, **kwargs)

        out = tree_map(lambda t: rearrange(t, '1 ... -> ...') if is_tensor(t) else t, out)
        return out

    return inner

def cycle(it):
    while True:
        for batch in it:
            yield batch

# tensor helpers

def is_tensor_empty(t):
    return t.numel() == 0

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def l2norm(t, dim = -1, eps = 1e-6):
    return F.normalize(t, dim = dim, eps = eps)

def straight_through(src, tgt):
    return src + (tgt - src).detach()

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def tree_flatten_with_inverse(input):
    out, tree_spec = tree_flatten(input)

    def inverse(output):
        return tree_unflatten(output, tree_spec)

    return out, inverse

def project(x, y):
    x, inverse = pack_with_inverse(x, 'b *')
    y, _ = pack_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = l2norm(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

# flow related

def log_snr_to_alpha_sigma(log_snr):
    alpha = log_snr.sigmoid().sqrt()
    sigma = (-log_snr).sigmoid().sqrt()
    return alpha, sigma

# rotary embedding

class RotaryEmbedding(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t, device = None):
        if isinstance(t, int):
            t = torch.arange(t, device = device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# gemma rms norm

class RMSNorm(Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_float = x.float()
        output = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

# siglip encoder

class SimpleAttention(Module):
    def __init__(self, dim, heads = 16, dim_head = 72, norm_eps = 1e-6, norm_klass = nn.LayerNorm):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = norm_klass(dim, eps = norm_eps)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        attn = sim.softmax(dim = -1, dtype = torch.float32).to(sim.dtype)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        dim_inner = None,
        expand_factor = 4.,
        glu = False,
        bias = False,
        rmsnorm = False,
        norm_klass = None,
        norm_eps = 1e-6,
        norm_all = False,
        activation = None
    ):
        super().__init__()
        activation = default(activation, 'silu' if glu else 'gelu_tanh')

        if isinstance(activation, str):
            if activation == 'silu':
                activation = nn.SiLU()
            elif activation in ('gelu', 'gelu_pytorch_tanh', 'gelu_new', 'gelu_tanh', 'gelu_fast'):
                activation = nn.GELU(approximate = 'tanh' if activation != 'gelu' else 'none')
            else:
                 raise ValueError(f'unknown activation {activation}')

        self.activation = activation
        dim_inner = default(dim_inner, int(dim * expand_factor * (2 / 3 if glu else 1)))

        norm_klass = default(norm_klass, (RMSNorm if rmsnorm else nn.LayerNorm))
        self.norm = norm_klass(dim, eps = norm_eps) if norm_klass != nn.Identity else nn.Identity()

        self.glu = glu

        self.proj_in = nn.Linear(dim, dim_inner * (2 if glu else 1), bias = bias)
        self.proj_out = nn.Linear(dim_inner, dim, bias = bias)

        self.post_proj_in_norm = RMSNorm(dim_inner, eps = norm_eps) if norm_all else nn.Identity()
        self.post_proj_out_norm = RMSNorm(dim, eps = norm_eps) if norm_all else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = self.proj_in(x)

        if self.glu:
            gate, x = x.chunk(2, dim = -1)
            x = self.activation(gate) * x
        else:
            x = self.activation(x)

        x = self.post_proj_in_norm(x)
        out = self.proj_out(x)
        return self.post_proj_out_norm(out)

class SigLIP(Module):
    def __init__(
        self,
        image_size = 224,
        patch_size = 14,
        dim = 1152,
        depth = 27,
        heads = 16,
        mlp_dim = 4304,
        norm_eps = 1e-6,
        norm_klass = nn.LayerNorm,
        activation = 'gelu_tanh'
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        dim_head = dim // heads

        self.to_patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

        self.pos_embed = nn.Parameter(torch.randn(num_patches, dim))

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                SimpleAttention(dim, heads, dim_head, norm_eps = norm_eps, norm_klass = norm_klass),
                FeedForward(dim = dim, dim_inner = mlp_dim, bias = True, norm_eps = norm_eps, norm_klass = norm_klass, activation = activation)
            ]))

        self.norm = norm_klass(dim, eps = norm_eps)

    def forward(self, x):
        x = self.to_patch_embed(x)
        num_patches = x.shape[1]

        x = x + self.pos_embed[:num_patches]

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        out = self.norm(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

# classes

def default_sample_times(
    shape,
    s = 0.999,
    alpha = 1.5,
    beta = 1,
    device = None
):
    """ they propose to sample times from Beta distribution - last part of appendix part B """

    alpha = full(shape, alpha, device = device)
    beta = full(shape, beta, device = device)
    sampled = Beta(alpha, beta).sample()
    return (1. - sampled) * s


# inpainting softmask, for real-time action chunking
# https://arxiv.org/abs/2506.07339 - eq (5)

def create_soft_inpaint_mask(
    chunk_len,
    condition_len,  # 'd' in the equation, action that is frozen
    generate_len,   # 's' in the equation, action being generated
    device = None
):
    transition_len = chunk_len - condition_len - generate_len
    assert transition_len >= 0, 'invalid lens, chunk length must be greater than the sum of the condition and generation lengths'

    # use same notation as paper

    H, s, d = chunk_len, generate_len, condition_len

    i = torch.arange(chunk_len, device = device)

    # transition exponential decay equation from frozen to generate section

    c = (H - s - i) / (H - s - d + 1)
    mask = c * (c.exp() - 1) / (math.exp(1) - 1)

    # inplace for 1 and 0 on left and right

    mask[:condition_len] = 1.
    mask[(H - s):] = 0.

    return mask

class SoftMaskInpainter(Module):
    def __init__(
        self,
        condition_len,
        transition_len,
        generate_len
    ):
        super().__init__()
        assert condition_len > 0 and generate_len > 0 and transition_len >= 0

        self.trajectory_length = condition_len + transition_len + generate_len

        soft_mask = create_soft_inpaint_mask(self.trajectory_length, condition_len, generate_len)
        soft_mask = rearrange(soft_mask, 'na -> 1 na 1')

        self.register_buffer('soft_mask', soft_mask, persistent = False)

    def pad_frozen(
        self,
        frozen_actions: Tensor # float[b nfa d]
    ):
        traj_len = self.trajectory_length
        frozen_len = frozen_actions.shape[1]

        if frozen_len >= traj_len:
            return frozen_actions[:, -traj_len:]

        return pad_at_dim(frozen_actions, (0, traj_len - frozen_len), dim = 1)

    def forward(
        self,
        frozen_actions: Tensor, # float[b nfa d]
        new_actions: Tensor     # float[b na d]
    ):
        frozen_actions = self.pad_frozen(frozen_actions)

        if self.soft_mask.device != new_actions.device:
            self.soft_mask = self.soft_mask.to(new_actions.device)

        return new_actions.lerp(frozen_actions, self.soft_mask)

# action guidance related

class RTCGuidance(Module):
    def __init__(
        self,
        guidance_weight_beta = 5.
    ):
        super().__init__()
        self.register_buffer('beta', tensor(guidance_weight_beta), persistent = False)

    def with_model_and_frozen_actions(
        self,
        model: Module,
        frozen_actions: Tensor, # float[b na d]
        soft_mask: tuple[int, int, int] | Tensor, # float[na] | float[1 na 1]
        input_time_arg_name = 'times',
        input_noised_actions_arg_name = 'actions',
        add_guidance_to_flow = True,
        flow_fn_name = 'forward',
        eps = 1e-4
    ):

        flow_fn = getattr(model, flow_fn_name, None)
        assert exists(flow_fn)

        sig = signature(flow_fn)

        if isinstance(soft_mask, tuple):
            soft_mask = SoftMaskInpainter(*soft_mask).soft_mask

        param_names = set(sig.parameters.keys())
        assert input_time_arg_name in param_names and input_noised_actions_arg_name in param_names

        def fn(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs).arguments

            # extract the time and noise actions from the flow function being invoked on pi-zero

            times = bound_args[input_time_arg_name]
            noise_actions = bound_args[input_noised_actions_arg_name]

            # make sure the input into the flow function has requires_grad turned on

            noise_actions.requires_grad_()

            # the actual forward

            output = flow_fn(*args, **kwargs)

            # assume predicted flow is first tensor

            (pred_flow, *rest), inverse_flatten = tree_flatten_with_inverse(output)

            # invoke the proposal

            guidance = self.forward(noise_actions, pred_flow, frozen_actions, times, soft_mask, eps)

            if add_guidance_to_flow:
                pred_flow = pred_flow + guidance

            # constitute output

            output = inverse_flatten((pred_flow, *rest))

            if add_guidance_to_flow:
                return output

            return output, guidance

        return fn

    def forward(
        self,
        noise_actions: Tensor,  # float[b na d]
        pred_actions: Tensor,   # float[b na d]
        frozen_actions: Tensor, # float[b na d]
        times: Tensor,          # float[] | float[b]
        soft_mask: Tensor,      # float[na] | float[1 na 1]
        eps = 1e-4
    ):

        assert noise_actions.requires_grad, 'the input noised actions must have had grad enabled'

        # handle variables

        beta = self.beta

        if soft_mask.ndim == 1:
            soft_mask = rearrange(soft_mask, 'nfa -> 1 nfa 1')

        # take care of the weight, which decays over time, clamped at beta at time = 0

        r_tau_squared = (1 - times).square() / (times.square() + (1 - times).square())

        guidance_weight = min(beta, (1 - times) / (times * r_tau_squared).clamp_min(eps))

        if guidance_weight.ndim == 1:
            guidance_weight = rearrange(guidance_weight, 'b -> b 1 1')

        # now carry out equation 2 for vjp

        error = soft_mask * (frozen_actions - pred_actions)
        vjp_scalar = (error.detach() * pred_actions).sum()

        gradient = torch.autograd.grad(vjp_scalar, noise_actions)[0]

        return gradient * guidance_weight

# the layer that projects from the embedding to a prediction of binned values (from -1. to 0.) in pi0.6
# alternative is the HL gauss layer proposed from deepmind

class BinnedValueLayer(Module):
    def __init__(
        self,
        dim,
        min_value = -1.,
        max_value = 0.,
        num_bins = 201,
        hard_discrete_targets = True
    ):
        super().__init__()
        self.num_bins = num_bins

        # params

        self.to_pred = LinearNoBias(dim, num_bins)

        self.hard_discrete_targets = hard_discrete_targets

        # bins

        bins = torch.linspace(min_value, max_value, num_bins)
        self.register_buffer('bins', bins, persistent = False)

    def value_to_prob(
        self,
        value: Tensor, # float[...]
        temperature = 0.1,
        hard = False
    ) -> Tensor: # float[... bins]
        requires_grad = value.requires_grad

        distance = einx.subtract('..., bins -> ... bins', value, self.bins)

        prob = (-distance / temperature).softmax(dim = -1)

        if not hard:
            return prob

        # straight through one hot if hard

        one_hot = F.one_hot(prob.argmax(dim = -1), num_classes = self.num_bins).float()

        if not requires_grad:
            return one_hot

        return one_hot + prob - prob.detach()

    def loss_fn(
        self,
        pred,
        target,
        reduction = 'none',
        hard_discretized_target = True
    ):
        if pred.ndim == 1:
            pred = log(self.value_to_prob(pred))

        if target.ndim == 1:
            target = self.value_to_prob(target, hard = self.hard_discrete_targets)

        return F.cross_entropy(pred, target, reduction = reduction)

    def forward(
        self,
        embed,
        return_value_and_logits = False
    ):

        logits = self.to_pred(embed)
        prob = logits.softmax(dim = -1)

        values = reduce(prob * self.bins, '... bins -> ...', 'sum')

        if not return_value_and_logits:
            return values

        return values, logits

# attention

class JointAttention(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_action = None,       # separate dimension for action tokens (input/output)
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        dropout = 0.,
        accept_memories = False,
        actions_norm_all = False,
        learned_value_action_residual_mix = False,
        norm_eps = 1e-6,
        rotary_emb: RotaryEmbedding | None = None,
        attn_value_gating = False  # whether to apply gating to attention values
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        kv_heads = default(kv_heads, heads)
        self.kv_heads = kv_heads

        assert divisible_by(heads, kv_heads)
        self.query_groups = heads // kv_heads

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        self.q_kv_split = (dim_inner, dim_kv_inner, dim_kv_inner)
        self.actions_qkvg_split = (dim_inner, dim_kv_inner, dim_kv_inner, dim_inner)
        self.mem_qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner, dim_inner, dim_kv_inner, dim_kv_inner)

        # action input/output dimension defaults to state dimension for backwards compatibility
        dim_action = default(dim_action, dim)
        self.dim_action = dim_action

        self.rotary_emb = rotary_emb

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.rmsnorm = RMSNorm(dim, eps = norm_eps)

        # state parameters

        self.to_qkv = nn.Linear(dim, (heads + 2 * kv_heads) * dim_head, bias = False)
        self.to_out = LinearNoBias(dim_inner, dim)

        # maybe memory parameters

        self.accept_memories = accept_memories

        self.mem_rmsnorm = RMSNorm(dim, eps = norm_eps) if accept_memories else None
        self.to_mem_qkv = LinearNoBias(dim, (heads + 2 * kv_heads) * dim_head) if accept_memories else None
        self.to_mem_out = LinearNoBias(dim_inner, dim) if accept_memories else None

        # action parameters - use dim_action for input/output but same heads
        # split QKV from gate projection for cleaner control
        self.attn_value_gating = attn_value_gating
        
        self.to_actions_qkv = LinearNoBias(dim_action, (heads + 2 * kv_heads) * dim_head)
        self.actions_qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)
        
        # separate gate projection, only created if gating is enabled
        if attn_value_gating:
            self.to_actions_gate = LinearNoBias(dim_action, dim_inner)
            nn.init.zeros_(self.to_actions_gate.weight)  # zero-init so (1 + gate) = 1 initially
        else:
            self.to_actions_gate = None

        self.to_actions_out = LinearNoBias(dim_inner, dim_action)

        # norms for all action linears
        # from Bytedance's GR-3

        self.actions_norm_all = actions_norm_all

        if actions_norm_all:
            self.actions_q_norm = RMSNorm(dim_head, eps = norm_eps)
            self.actions_k_norm = RMSNorm(dim_head, eps = norm_eps)
            self.actions_v_norm = RMSNorm(dim_head, eps = norm_eps)
            self.actions_out_norm = RMSNorm(dim_action, eps = norm_eps)

        # identity probes for surgery

    def forward_actions_with_cached_state(
        self,
        actions,
        cached_state_keys_values: tuple[Tensor, Tensor],
        memories: tuple[Tensor, Tensor] | None = None,
        rotary_emb = None,
        mask: Tensor | None = None, # bool[b n]
        return_keys_values = False,
        flex_attn_fn: Callable | None = None,
        knowledge_insulate = False,
        discretized_action_length = 0
    ):

        aq, ak, av = self.to_actions_qkv(actions).split(self.actions_qkv_split, dim = -1)
        
        # optional gating
        if self.attn_value_gating:
            ag = self.to_actions_gate(actions)
            ag = self.split_heads(ag)
        else:
            ag = None

        aq, ak, av = tuple(self.split_heads(t) for t in (aq, ak, av))

        if self.actions_norm_all:
            aq, ak, av = tuple(norm(t) for norm, t in zip((self.actions_q_norm, self.actions_k_norm, self.actions_v_norm), (aq, ak, av)))

        q = aq
        mk, mv = cached_state_keys_values

        # able to stop gradients from actions to state - (knowledge insulation blogpost https://www.physicalintelligence.company/research/knowledge_insulation)

        if knowledge_insulate:
            mk, mv = tuple(t.detach() for t in (mk, mv))

        # handle read, write memories

        assert not (self.accept_memories ^ exists(memories))

        if exists(memories):
            _, write_memories = memories
            write_memories = self.mem_rmsnorm(write_memories)

        # concat cache key / values with action key / values

        if exists(rotary_emb):
            action_len = aq.shape[-2]
            rotary_emb_actions = rotary_emb[..., -action_len:, :]

            q = apply_rotary_pos_emb(rotary_emb_actions, q)
            ak = apply_rotary_pos_emb(rotary_emb_actions, ak)

        # concat cache key / values with action key / values

        k, v = tuple(cat(tensors, dim = -2) for tensors in zip((mk, mv), (ak, av)))

        # attention

        if exists(flex_attn_fn):
            out = flex_attn_fn(q, k, v)
        else:
            q = q * self.scale

            if self.query_groups > 1:
                k = repeat(k, 'b h n d -> b (h g) n d', g = self.query_groups)
                v = repeat(v, 'b h n d -> b (h g) n d', g = self.query_groups)

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')


            if exists(mask):
                sim = einx.where('b j, b h i j, -> b h i j', mask, sim, max_neg_value(sim))

            if discretized_action_length > 0:
                m_len = mk.shape[-2]
                discretized_action_mask = torch.zeros(sim.shape[-1], dtype = torch.bool, device = actions.device)
                discretized_action_mask[(m_len - discretized_action_length):m_len] = True

                sim = sim.masked_fill(discretized_action_mask, max_neg_value(sim))

            attn = sim.softmax(dim = -1, dtype = torch.float32).to(sim.dtype)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # optional gate - use (1 + gates) so zero-init produces identity

        if self.attn_value_gating and exists(ag):
            out = out * (1. + ag)

        # merge attention heads

        out = self.merge_heads(out)

        actions_out = self.to_actions_out(out)

        if self.actions_norm_all:
            actions_out = self.actions_out_norm(actions_out)

        if not return_keys_values:
            return actions_out

        return actions_out, (mk, mv, ak, av)

    def forward_only_vision_language(
        self,
        state: Tensor, # float[b n d]
        num_visual_tokens = 0,
        is_prefix = False,
        rotary_emb = None
    ) -> Tensor: # float[b n d]

        state = self.rmsnorm(state)

        device = state.device

        q, k, v = self.to_qkv(state).split(self.q_kv_split, dim = -1)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        # attention

        q = q * self.scale

        if self.query_groups > 1:
            k = repeat(k, 'b h n d -> b (h g) n d', g = self.query_groups)
            v = repeat(v, 'b h n d -> b (h g) n d', g = self.query_groups)

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        causal_mask = torch.ones(sim.shape[-2:], dtype = torch.bool, device = device).triu(1)

        if is_prefix:
            causal_mask.fill_(False)
        elif num_visual_tokens > 0:
            causal_mask[:, :num_visual_tokens] = False
            causal_mask[:num_visual_tokens, :num_visual_tokens] = False

        sim = sim.masked_fill(causal_mask, max_neg_value(sim))

        attn = sim.softmax(dim = -1, dtype = torch.float32).to(sim.dtype)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge attention heads

        out = self.merge_heads(out)

        return self.to_out(out)

    def forward(
        self,
        multimodal_seq,
        actions,
        multimodal_prefix_bidirectional_length = 0,
        multimodal_prefix_bidirectional_start = 0,
        rotary_emb = None,
        memories: tuple[Tensor, Tensor] | None = None,
        mask: Tensor | None = None,    # bool[b n]
        ar_mask: Tensor | None = None, # bool[b n]
        return_keys_values = False,
        flex_attn_fn: Callable | None = None,
        knowledge_insulate = False,
        discretized_action_length = 0
    ):
        
        multimodal_len = multimodal_seq.shape[-2] if exists(multimodal_seq) else 0
        device = multimodal_seq.device if exists(multimodal_seq) else actions.device

        if exists(multimodal_seq):
            multimodal_seq = self.rmsnorm(multimodal_seq)

            # separate projections for multimodal seq vs actions

            mq, mk, mv = self.to_qkv(multimodal_seq).split(self.q_kv_split, dim = -1)
        else:
            mq = mk = mv = None

        # Actions are already RMS-normalized by attn_ada_rmsnorm before being passed here
        # (see PiZero.forward line ~2630: action_tokens, gate = attn_ada_rmsnorm(...))
        aq, ak, av = self.to_actions_qkv(actions).split(self.actions_qkv_split, dim = -1)
        
        # optional gating
        if self.attn_value_gating:
            ag = self.to_actions_gate(actions)
        else:
            ag = None

        if exists(multimodal_seq):
            mq, mk, mv, aq, ak, av = tuple(self.split_heads(t) for t in (mq, mk, mv, aq, ak, av))
        else:
            aq, ak, av = tuple(self.split_heads(t) for t in (aq, ak, av))
        
        if exists(ag):
            ag = self.split_heads(ag)

        if self.actions_norm_all:
            aq, ak, av = tuple(norm(t) for norm, t in zip((self.actions_q_norm, self.actions_k_norm, self.actions_v_norm), (aq, ak, av)))

        # able to stop gradients from actions to state - (knowledge insulation blogpost https://www.physicalintelligence.company/research/knowledge_insulation)

        if knowledge_insulate:
            mk, mv = tuple(t.detach() for t in (mk, mv))

        # value residual

        if exists(multimodal_seq):
            q, k, v = tuple(cat(tensors, dim = -2) for tensors in zip((mq, mk, mv), (aq, ak, av)))
        else:
            q, k, v = aq, ak, av

        # handle read, write memories

        has_memories = exists(memories) and any([m.numel() > 0 for m in memories])

        assert not (self.accept_memories ^ has_memories)

        if has_memories:
            memories, unpack_memories = pack_with_inverse(memories, 'b * d')
            memories = self.mem_rmsnorm(memories)
            mqkv_res = self.to_mem_qkv(memories).split(self.mem_qkv_split, dim = -1)
            mqr, mkr, mvr, mqw, mkw, mvw = mqkv_res

            mqr, mkr, mvr, mqw, mkw, mvw = tuple(self.split_heads(t) for t in (mqr, mkr, mvr, mqw, mkw, mvw))

            k = cat((mkr, k, mkw), dim = -2)
            v = cat((mvr, v, mvw), dim = -2)
            q, attn_output_unpack_memories = pack_with_inverse((mqr, q, mqw), 'b h * d')

        # rotary embedding


        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)
            
            # Extract post-RoPE keys for caching (used for KV caching in inference)
            if exists(multimodal_seq):
                mk = k[:, :, :multimodal_len, :]  # state portion post-RoPE
                ak = k[:, :, multimodal_len:, :]  # action portion post-RoPE


        if exists(flex_attn_fn):
            out = flex_attn_fn(q, k, v)

        else:
            # attention

            if self.query_groups > 1:
                k = repeat(k, 'b h n d -> b (h g) n d', g = self.query_groups)
                v = repeat(v, 'b h n d -> b (h g) n d', g = self.query_groups)

            q = q * self.scale

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            if exists(ar_mask):
                cumsum = ar_mask.cumsum(dim = -1)

                # att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
                # where True means can attend. PiZero's causal_mask is inverted (True = masked out).
                # So: causal_mask = NOT(cumsum_k <= cumsum_q) = cumsum_k > cumsum_q
                causal_mask = cumsum.unsqueeze(-2) > cumsum.unsqueeze(-1)  # [b, Q, K]
                causal_mask = causal_mask.unsqueeze(1) # [b, 1, Q, K]

            elif multimodal_prefix_bidirectional_length > 0:
                start = multimodal_prefix_bidirectional_start
                end = start + multimodal_prefix_bidirectional_length
                causal_mask = torch.ones(sim.shape[-2:], dtype = torch.bool, device = device).triu(1)
                causal_mask[start:end, start:end] = False
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # [1, 1, i, j]
            else:
                causal_mask = torch.ones(sim.shape[-2:], dtype = torch.bool, device = device).triu(1)
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # [1, 1, i, j]

            if exists(mask):
                causal_mask = causal_mask | (~mask.view(actions.shape[0], 1, 1, -1))

            if not exists(ar_mask) and discretized_action_length > 0:
                causal_mask[..., seq_len:, (seq_len - discretized_action_length):seq_len] = True

            sim = sim.masked_fill(causal_mask, max_neg_value(sim))

            attn = sim.softmax(dim = -1, dtype = torch.float32).to(sim.dtype)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # optional gating of values - use (1 + gates) so zero-init produces identity

        if self.attn_value_gating and exists(ag):
            gates = pad_at_dim(1. + ag, (out.shape[-2] - ag.shape[-2], 0), value = 1., dim = -2)
            out = out * gates

        # split out memories

        if self.accept_memories:
            mem_read_out, out, mem_write_out = attn_output_unpack_memories(out)

        # merge attention heads

        out = self.merge_heads(out)

        # separate projections for multimodal seq vs actions
        
        mout, aout = out[:, :multimodal_len], out[:, multimodal_len:]

        mout, aout = self.to_out(mout), self.to_actions_out(aout)

        if self.actions_norm_all:
            aout = self.actions_out_norm(aout)

        # need to facilitate returning full merged output to match their residual path
        output = (mout, aout)

        if self.accept_memories:
            mem_out, unpack_memories = pack_with_inverse((mem_read_out, mem_write_out), 'b h * d')
            mem_out = self.merge_heads(mem_out)
            mem_out = self.to_mem_out(mem_out)

            output = (*output, unpack_memories(mem_out, 'b * d'))

        if not return_keys_values:
            return output

        return output, (mk, mv, ak, av)

# actions need time conditioning
# ada-ln zero from DiT - here we will improvise with adaptive rmsnorm

class SinusoidalEmbed(Module):
    def __init__(self, dim, min_period = 0.004, max_period = 4.0):
        super().__init__()
        half_dim = dim // 2
        fraction = torch.linspace(0.0, 1.0, half_dim, dtype = torch.float64)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = (2 * math.pi / period).float()
        self.register_buffer('scaling_factor', scaling_factor)

    def forward(self, times):
        fourier_input = einx.multiply('..., d -> ... d', times, self.scaling_factor.to(times.dtype))
        return cat((fourier_input.sin(), fourier_input.cos()), dim = -1)

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond,
        eps = 1e-6
    ):
        super().__init__()
        self.eps = eps
        self.to_modulation = nn.Linear(dim_cond, dim * 3)
        
        # For weight sync compatibility - stores the base weight
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.zeros(dim))


    def forward(self, x, cond, return_gate = False):
        x_float = x.float()
        raw_normed = (pow(x_float, 2).mean(-1, keepdim = True) + self.eps).rsqrt() * x_float

        if not exists(cond):
            w = self.norm.weight.float()
            out = raw_normed * (1. + w)
            return (out, None) if return_gate else out

        if cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        modulation = self.to_modulation(cond)
        scale, shift, gate = modulation.chunk(3, dim = -1)

        # With conditioning - just use modulation's scale, no extra weight
        out = raw_normed * (1. + scale) + shift


        if return_gate:
            return out.to(x.dtype), gate

        return out.to(x.dtype)

class AdaptiveLayerscale(Module):
    def __init__(
        self,
        dim,
        dim_cond,
        adaln_zero_bias_init_value = -2.
    ):
        super().__init__()
        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = adaln_zero_gamma_linear

    def forward(self, actions, cond):
        if not exists(cond):
            return actions

        if cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        gamma = self.to_adaln_zero_gamma(cond)
        return actions * gamma.sigmoid()

# main class

class PiZero(Module):
    @beartype
    @save_args_kwargs
    def __init__(
        self,
        dim,
        num_tokens,
        dim_action_input,
        dim_joint_state,
        dim_action = None,           # separate dimension for action tokens (for PI weight loading)
        dim_time_cond = None,
        time_mlp_depth = 1,
        time_sinusoidal = False,
        time_min_period = 0.004,
        time_max_period = 4.,
        time_infused_action_tokens = False,
        layer_time_cond = True,
        depth = 12,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        use_flex_attn = False,
        ff_expand_factor = 4.,
        action_ff_expand_factor = None,  # separate expand factor for action feedforward
        norm_eps = 1e-6,
        vit: Module | None = None,
        vit_dim = None,
        external_state_encoders: Module | list[Module] | None = None,
        dim_internal_state: int | None = None,
        num_action_register_tokens = 4,
        pi05 = False,
        token_scale: float | None = None,  # Override token scaling (None = auto sqrt(dim), 1.0 = no scaling)
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        activation: str = 'silu',
        lm_pad_id = -1,
        lm_loss_weight = 1.,
        model_predict_output: Literal['flow', 'clean'] = 'flow', # dreamer4 as well as https://arxiv.org/abs/2511.13720 - something is going around, make sure it is not missed.
        max_timesteps = 16,
        flow_loss_weight = 1.,
        sample_times_fn = default_sample_times,
        rtc_guidance = None,                 # use the guidance proposed in https://arxiv.org/abs/2506.07339, which in turn is inspired by training-free guidance paper, which in turn from pseudo-inverse paper.
        train_time_rtc = False,
        train_time_rtc_max_delay = None,
        sample_guidance_beta = 5.,           # corresponds to the beta max guidance term in the RTC paper
        sample_soft_mask_lens: tuple[int, int, int] | None = None, # (condition, transition, generation) lengths - default to frozen action seq dimension 'nfa' above for condition len with 0 transition, but can be overridden
        reward_tokens_dropout_prob = 0.,
        num_advantage_tokens = 0,
        advantage_tokens_dropout_prob = 0.,
        num_recurrent_memory_tokens = 0,
        num_residual_streams = 1,
        dim_latent = None,
        action_dit_norm_all_linears = True,  # Cheang et al. https://arxiv.org/abs/2507.15493v1 - in GR-3, Bytedance shares the finding that aggressive normalization of the action diffusion transformer (one after each linear), stabilizes training and greatly improves results
        predict_task_status_head = False,    # Cheang et al. https://arxiv.org/abs/2507.15493v1 - an important detail in the paper where they add a prediction head for task status; they generate negative pairs of language - action samples and force the network to predict "invalid" label. this made the robot follow the language significantly better.
        num_tasks = None,
        num_task_status = 3,
        task_status_is_invalid = 2,          # the index for which the task status is invalid - `-1` in paper, but we'll do 2 here
        task_status_loss_weight = 1.,
        use_spo = True,                      # Xie et al. https://arxiv.org/abs/2401.16025 - validated by PI to learn while PPO is unstable and does not - will start adopting SPO for all future on-policy work, until some paper points out deficiencies
        use_asymmetric_spo = True,           # FPO++ paper proposes combining ppo and spo
        is_critic = False,                   # whether this model is used as the critic, with the histogram classification loss from Imani et al. https://arxiv.org/html/2402.13425v1
        critic_use_discrete_bins = True,     # use the categorical discrete binning of the rewards (which is normalized to between -1. and 0.) for pi0.6
        critic_value_kwargs: dict = dict(
            min_value = -10.,
            max_value = 10.,
            num_bins = 50
        ),
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        predict_discretized_action_aux_loss = False,
        predict_discrete_action_loss_weight = 0.1,
        discrete_action_pad_id = -1
    ):
        super().__init__()
        dim_time_cond = default(dim_time_cond, dim * 2)
        dim_action = default(dim_action, dim)
        action_ff_expand_factor = default(action_ff_expand_factor, ff_expand_factor)

        self.dim = dim
        self.token_scale = default(token_scale, dim ** 0.5)

        self.dim_action = dim_action

        # flex attention related

        assert not (use_flex_attn and not exists(flex_attention)), 'flex attention cannot be used'
        self.use_flex_attn = use_flex_attn
        self.pi05 = pi05

        # vit

        self.vit = vit

        self.maybe_to_image_tokens = nn.Linear(vit_dim, dim) if exists(vit_dim) else nn.Identity()

        self.attn_kwargs = {**attn_kwargs, 'kv_heads': kv_heads, 'norm_eps': norm_eps}

        # embedding

        self.token_emb = nn.Embedding(num_tokens, dim)

        # joint state tokens
        # projects to dim_action (aligned with PI weights)

        self.to_joint_state_tokens = nn.Linear(dim_joint_state, dim_action)

        self.dim_internal_state = default(dim_internal_state, dim_action)
        self.to_internal_state_tokens = nn.Linear(self.dim_internal_state, dim_action) if exists(dim_internal_state) else nn.Identity()

        # additional external states

        external_state_encoders = default(external_state_encoders, [])
        self.external_state_encoders = ModuleList(external_state_encoders)

        # actions - use dim_action for action token dimension

        self.dim_action_input = dim_action_input

        self.action_register_tokens = nn.Parameter(torch.zeros(num_action_register_tokens, dim_action))
        nn.init.normal_(self.action_register_tokens, std = 0.02)

        self.to_action_tokens = nn.Linear(dim_action_input, dim_action)

        # time conditioning
        
        self.time_infused_action_tokens = time_infused_action_tokens
        self.layer_time_cond = layer_time_cond

        if time_sinusoidal:
            # If time-infused, sinusoidal dim should match dim_action (usually 1024)
            # cats [action_tokens (1024), sinusoidal (1024)] -> 2048
            # BUT the to_time_cond MLP might expect a different input dim (e.g. 2048 in pretrained)
            
            mlp_in_dim = dim
            time_embed = SinusoidalEmbed(mlp_in_dim, min_period = time_min_period, max_period = time_max_period)

            if time_infused_action_tokens:
                 self.infusion_time_embed = SinusoidalEmbed(dim_action, min_period = time_min_period, max_period = time_max_period)
        else:
            time_embed = SinusoidalEmbed(dim)

        # we can reuse to_time_cond for infusion or layer conditioning
        # for PI0, it's used for infusion
        
        in_dim = mlp_in_dim if time_sinusoidal else dim
        out_dim = dim_action if not layer_time_cond else dim_time_cond
        hidden_dims = (dim_time_cond,) * (time_mlp_depth - 1) if time_mlp_depth > 1 else ()

        self.to_time_cond = nn.Sequential(
            time_embed,
            MLP(in_dim, *hidden_dims, out_dim, activation = nn.SiLU())
        )

        # action-time fusion (pi0)

        self.to_action_time_fuse = nn.Identity()

        if time_infused_action_tokens:
            self.to_action_time_fuse = MLP(dim_action * 2, dim_action, dim_action, activation = nn.SiLU())

        # latent variable / gene conditioning

        can_accept_latent = exists(dim_latent)
        self.can_accept_latent = can_accept_latent

        if can_accept_latent:
            self.to_latent_cond = MLP(dim_latent, dim_time_cond * 2, dim_time_cond, activation = nn.SiLU())

            last_linear = self.to_latent_cond.layers[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

        # positional embedding

        self.rotary_emb = RotaryEmbedding(dim_head)

        # recurrent memory parameters and logic

        self.has_recurrent_memories = num_recurrent_memory_tokens > 0

        self.memory_tokens = nn.Parameter(torch.zeros(num_recurrent_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std = 0.02)

        self.final_norm_write_memories = RMSNorm(dim, eps = norm_eps) if self.has_recurrent_memories else None

        # attention and feedforward

        layers = []
        cond_layers = []

        for i in range(depth):
            is_first_block = i == 0

            layers.append(ModuleList([
                JointAttention(dim = dim, dim_action = dim_action, dim_head = dim_head, heads = heads, actions_norm_all = action_dit_norm_all_linears, accept_memories = self.has_recurrent_memories, learned_value_action_residual_mix = not is_first_block, **self.attn_kwargs),
                FeedForward(dim = dim, expand_factor = ff_expand_factor, glu = True, rmsnorm = True, norm_eps = norm_eps, activation = activation, **ff_kwargs),
                FeedForward(dim = dim_action, expand_factor = action_ff_expand_factor, glu = True, norm_klass = nn.Identity, norm_all = action_dit_norm_all_linears, norm_eps = norm_eps, activation = activation, **ff_kwargs),
                FeedForward(dim = dim, expand_factor = ff_expand_factor, glu = True, rmsnorm = True, norm_eps = norm_eps, **ff_kwargs) if self.has_recurrent_memories else None
            ]))

            # pi05: adaptiveRMSNorm with time conditioning
            # pi0: plain RMSNorm without time conditioning

            if layer_time_cond:
                cond_layers.append(ModuleList([
                    AdaptiveRMSNorm(dim_action, dim_time_cond),
                    AdaptiveRMSNorm(dim_action, dim_time_cond),
                ]))
            else:
                cond_layers.append(ModuleList([
                    RMSNorm(dim_action, eps = norm_eps),
                    RMSNorm(dim_action, eps = norm_eps),
                ]))

        self.layers = ModuleList(layers)
        self.cond_layers = ModuleList(cond_layers)


        self.final_norm = RMSNorm(dim, eps = norm_eps)

        if not layer_time_cond:
            self.final_actions_norm = RMSNorm(dim_action, eps = norm_eps)
        else:
            self.final_actions_norm = AdaptiveRMSNorm(dim_action, dim_time_cond, eps = norm_eps)

        # unembedding

        self.state_to_logits = LinearNoBias(dim, num_tokens)

        # explicit task conditioning without prompt

        self.num_tasks = num_tasks
        self.has_task_cond = exists(num_tasks) and num_tasks > 0

        if self.has_task_cond:
            self.task_emb = nn.Embedding(num_tasks, dim)
            
            # Add a Null Task Token
            self.null_task_token = nn.Parameter(torch.randn(dim))

        # to task status prediction

        self.to_task_status = LinearNoBias(dim, num_task_status)
        self.task_status_is_invalid = task_status_is_invalid
        self.task_status_loss_weight = task_status_loss_weight

        # actor related

        self.actions_to_pred_flow = None
        self.loss_fn = None

        if not is_critic:
            self.actions_to_pred_flow = nn.Linear(dim_action, dim_action_input)
            self.loss_fn = nn.MSELoss(reduction = 'none')

        # use simple policy optimization

        self.use_spo = use_spo

        self.use_asymmetric_spo = use_asymmetric_spo

        # whether the model outputs x0 or flow

        self.model_predict_output = model_predict_output
        self.max_timesteps = max_timesteps

        # critic related

        self.is_critic = is_critic

        self.critic_use_discrete_bins = critic_use_discrete_bins

        if critic_use_discrete_bins:
            self.to_critic_value = BinnedValueLayer(
                dim,
                **critic_value_kwargs
            )
        else:
            self.to_critic_value = HLGaussLayer(
                dim,
                hl_gauss_loss = critic_value_kwargs
            )

        # the language token id padding id, for fine-tuning as well as taking care of the masking on top of causal mask

        self.lm_pad_id = lm_pad_id

        # flow related


        # reward classifier free guidance

        self.reward_tokens_dropout_prob = reward_tokens_dropout_prob

        self.null_reward_token = nn.Parameter(torch.randn(dim))
        
        # advantage tokens and cfg related

        self.num_advantage_tokens = num_advantage_tokens
        self.can_advantage_token_cond = num_advantage_tokens > 0

        if self.can_advantage_token_cond:
            self.advantage_embed = nn.Embedding(num_advantage_tokens, dim)
            self.null_advantage_token = nn.Parameter(torch.randn(dim))

        self.advantage_tokens_dropout_prob = 0.

        # time sampling related

        self.sample_times_fn = default(sample_times_fn, torch.rand)

        # soft mask for sampling with inpainting of frozen actions

        self.soft_mask_inpainter = SoftMaskInpainter(*sample_soft_mask_lens) if exists(sample_soft_mask_lens) else None

        # guidance as proposed in their RTC paper

        rtc_guidance = default(rtc_guidance, model_predict_output == 'flow')
        self.rtc_guidance = RTCGuidance(sample_guidance_beta) if rtc_guidance else None

        # train time RTC related - https://arxiv.org/abs/2512.05964

        self.train_time_rtc = train_time_rtc

        assert not train_time_rtc or exists(train_time_rtc_max_delay)
        self.train_time_rtc_max_delay = train_time_rtc_max_delay

        # predicting discretized action auxiliary loss

        self.predict_discretized_action_aux_loss = predict_discretized_action_aux_loss

        if predict_discretized_action_aux_loss:
            # https://arxiv.org/abs/2501.09747

            from transformers import AutoProcessor
            self.discretized_action_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

            self.discretized_action_vocab_size = self.discretized_action_tokenizer.vocab_size
            self.discrete_action_embeds = nn.Embedding(self.discretized_action_vocab_size + 1, dim)

            self.to_discrete_action_pred = nn.Sequential(
                RMSNorm(dim),
                LinearNoBias(dim, self.discretized_action_vocab_size)
            )

            self.discrete_action_pad_id = discrete_action_pad_id

        # loss related

        self.lm_loss_weight = lm_loss_weight
        self.predict_discrete_action_loss_weight = predict_discrete_action_loss_weight

        self.flow_loss_weight = flow_loss_weight

        # sampling related

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)


        # tensor typing

        self._nm = num_recurrent_memory_tokens

    @staticmethod
    def from_checkpoint(
        folder: str | Path = 'checkpoints/pi0_base',
        **kwargs
    ):
        import json
        from safetensors import safe_open
        from pi_zero_pytorch.load import (
            create_pizero_config_for_pi0,
            build_converted_state_dict,
            download_pi0_weights
        )

        folder = Path(folder)

        if not folder.exists():
            download_pi0_weights(folder)

        config_path = folder / 'config.json'
        weights_path = folder / 'model.safetensors'
        pizero_weights_path = folder / 'pizero.pt'

        assert config_path.exists(), f'config file not found at "{str(config_path)}"'

        with open(config_path) as f:
            config = json.load(f)

        pz_config = create_pizero_config_for_pi0(config)

        pz_config.update(kwargs)

        if 'vit' not in pz_config:
            pz_config['vit'] = SigLIP(norm_eps = pz_config['norm_eps'])
            pz_config['vit_dim'] = 1152

        model = PiZero(**pz_config)

        if not pizero_weights_path.exists():
            print('Converting weights to pizero.pt...')
            with safe_open(weights_path, framework = 'pt') as pi_weights:
                build_converted_state_dict(pi_weights, model.state_dict())
                torch.save(model.state_dict(), pizero_weights_path)

            return model

        new_state = torch.load(pizero_weights_path, weights_only = True, mmap = True)
        model.load_state_dict(new_state, strict = False)

        return model

    def action_params_for_evolution(self):
        action_params = set()

        add_module = lambda m: action_params.update(set(m.parameters()))

        add_module(self.to_action_tokens)

        for (
            (attn, state_ff, actions_ff, memories_ff),
            (attn_ada_rmsnorm, ff_ada_rmsnorm),
        ) in zip(self.layers, self.cond_layers):

            add_module(attn.to_actions_out)
            add_module(attn.to_actions_qkv)
            if attn.attn_value_gating:
                add_module(attn.to_actions_gate)
            add_module(actions_ff)

        add_module(self.actions_to_pred_flow)
        return action_params

    @property
    def can_cfg(self):
        return self.reward_tokens_dropout_prob > 0.

    @property
    def device(self):
        return next(self.parameters()).device

    def create_ema(
        self,
        beta = 0.99,
        **ema_kwargs
    ) -> EMA:

        ema_pi_zero = EMA(
            self,
            beta = beta,
            include_online_model = False,
            forward_method_names = (
                'sample_actions',
            ),
            **ema_kwargs
        )

        return ema_pi_zero

    def create_actor(self, **kwargs) -> PiZero:
        assert not self.is_critic, 'base model must not be a critic'

        orig_args, orig_kwargs = self._init_args_kwargs
        actor = PiZero(*orig_args, **orig_kwargs, **kwargs)

        # load all possible shared parameters except for output head to logits (for histogram loss)

        state_dict = self.state_dict()
        actor.load_state_dict(state_dict, strict = False)

        return actor.to(self.device)

    def create_critic(self, **kwargs) -> PiZero:
        assert not self.is_critic, 'base model must be policy optimizable as well as not a critic already'

        assert 'is_critic' not in kwargs
        kwargs.update(is_critic = True)

        orig_args, orig_kwargs = self._init_args_kwargs
        critic = PiZero(*orig_args, **orig_kwargs, **kwargs)

        # load all possible shared parameters except for output head to logits (for histogram loss)

        state_dict = self.state_dict()
        critic.load_state_dict(state_dict, strict = False)

        return critic.to(self.device)

    def sample_actions(
        self,
        images,
        token_ids,
        joint_states,
        trajectory_length: int,
        latents: Tensor | None = None,               # float[d] | float[b d]
        reward_tokens: Tensor | None = None,         # float[b d]
        advantage_ids: Tensor | int | None = None,   # int[b] | int
        internal_state_tokens: Tensor | None = None, # float[b ns d]
        frozen_actions: Tensor | None = None,        # float[b nfa da]
        soft_mask_lens: tuple[int, int, int] | None = None, # overriding the softmax inpainter at init
        return_frozen_actions_with_sampled = False,
        return_original_noise = False,
        steps = 18,
        show_pbar = True,
        cond_scale = 0.,
        temperature = 1.,
        remove_parallel_component = True,
        keep_parallel_frac = 0.,
        cache_kv = True,
        return_states_for_replay = False,
        critic: Module | None = None,
        actions_last_step: Tensor | None = None, # float[b na da]
    ):
        assert not self.is_critic

        batch_size = token_ids.shape[0]

        was_training = self.training
        self.eval()

        if exists(critic) and not exists(actions_last_step):
            # default to zeros for first step or if not passed
            actions_last_step = torch.zeros((batch_size, trajectory_length, self.dim_action_input), device = self.device)

        pbar = tqdm.tqdm(desc = 'sampling action trajectory', disable = not show_pbar, total = steps)

        # accumulate for flow policy optimization

        critic_values = []

        # validate frozen actions for real-time action chunking, if any

        inpaint_actions = exists(frozen_actions)
        use_rtc_guidance = False

        if inpaint_actions:
            soft_mask_inpainter = self.soft_mask_inpainter

            frozen_action_input_len = frozen_actions.shape[1]

            if not exists(soft_mask_inpainter):
                soft_mask_lens = default(soft_mask_lens, (frozen_action_input_len, 0, trajectory_length - frozen_action_input_len))
                soft_mask_inpainter = SoftMaskInpainter(*soft_mask_lens)

            assert soft_mask_inpainter.trajectory_length == trajectory_length
            frozen_actions_for_inpaint = soft_mask_inpainter.pad_frozen(frozen_actions)

            use_rtc_guidance = exists(self.rtc_guidance)

        # ode step function

        cached_state_kv = None
        null_cached_state_kv = None
        input_args = None
        input_kwargs = None

        def ode_fn(timestep, denoised_actions):
            nonlocal cached_state_kv
            nonlocal null_cached_state_kv
            nonlocal input_args
            nonlocal input_kwargs

            # take care of inpainting if needed

            if inpaint_actions:

                if self.train_time_rtc:
                    time_mask = arange(trajectory_length, device = self.device) < frozen_action_input_len
                    timestep = einx.where('na,,', time_mask, 1., timestep)
                    timestep = repeat(timestep, 'na -> b na', b = batch_size)
                else:
                    denoised_actions = soft_mask_inpainter(frozen_actions_for_inpaint, denoised_actions)

                    if exists(self.rtc_guidance):
                        # the denoised actions must have grad enabled to calculate the vjp

                        denoised_actions.requires_grad_()

            input_args = (
                images,
                token_ids,
                joint_states,
                denoised_actions
            )


            input_kwargs = dict(
                times = timestep,
                latents = latents,
                reward_tokens = reward_tokens,
                advantage_ids = advantage_ids,
                internal_state_tokens = internal_state_tokens,
                cached_state_keys_values = (cached_state_kv, null_cached_state_kv),
                cond_scale = cond_scale,
                remove_parallel_component = remove_parallel_component,
                keep_parallel_frac = keep_parallel_frac
            )

            output, (new_cached_state_kv, new_null_cached_state_kv) = self.forward_with_reward_cfg(*input_args, **input_kwargs)

            flow = output

            # in the follow up improved real time chunking guidance paper, they propose modifying the flow using a technique from some previous inpainting research
            # it involves calculating the jacobian of the error between the frozen action and predicted action (of the frozen section), iiuc

            guidance_term = 0.

            if inpaint_actions and exists(self.rtc_guidance):
                assert self.model_predict_output == 'flow' # handle this eventually

                padded_timestep = pad_right_ndim(timestep, flow.ndim - 1)

                pred_actions = denoised_actions + flow * (1. - padded_timestep)

                if not self.train_time_rtc:
                    guidance_term = self.rtc_guidance(denoised_actions, pred_actions, frozen_actions_for_inpaint, timestep, soft_mask_inpainter.soft_mask)

            # handle probabilistic

            if cache_kv:
                cached_state_kv = new_cached_state_kv
                null_cached_state_kv = new_null_cached_state_kv

            pbar.update(1)

            return flow + guidance_term

        # maybe wrap ode_fn with no grad if not needed, but will be needed for RTC Guidance

        if not use_rtc_guidance:
            ode_fn = torch.no_grad()(ode_fn)

        # start with random gaussian noise - y0

        noise = torch.randn((batch_size, trajectory_length, self.dim_action_input), device = self.device)

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = self.odeint_fn(ode_fn, noise, times)

        sampled_actions = trajectory[-1]

        # final inpaint if needed

        if inpaint_actions:
            sampled_actions = soft_mask_inpainter(frozen_actions_for_inpaint, sampled_actions)

            if not return_frozen_actions_with_sampled:
                sampled_actions = sampled_actions[:, frozen_action_input_len:]

        self.train(was_training)

        pbar.close()

        if return_original_noise:
            out = (sampled_actions, noise) # for diffusion steering paper from Wagenmaker et al.
        else:
            out = sampled_actions

        if not exists(critic):
            return out

        # return critic value predictions if passed in

        del input_kwargs['cached_state_keys_values']
        input_kwargs['times'].zero_().add_(1.)

        # the critic expects the state and the actions from the last step
        # which was passed into sample_actions and captured in the closure

        critic_input_args = (images, token_ids, joint_states, actions_last_step)

        values, _ = critic(*critic_input_args, **input_kwargs)

        return out, values

    @torch.no_grad()
    def forward_with_reward_cfg(
        self,
        *args,
        reward_tokens: Tensor | None = None, # float[b d]
        cached_state_keys_values = (None, None),
        cond_scale = 0.,
        remove_parallel_component = False,
        keep_parallel_frac = 0.,
        **kwargs
    ):

        with_reward_cache, without_reward_cache = cached_state_keys_values

        forward_kwargs = dict(
            return_state_keys_values = True,
            return_actions_flow = True,
        )

        action_flow_with_reward, with_reward_cache_kv = self.forward(
            *args,
            reward_tokens = reward_tokens,
            cached_state_keys_values = with_reward_cache,
            **forward_kwargs,
            **kwargs
        )

        if not exists(reward_tokens) or cond_scale == 0.:
            return action_flow_with_reward, (with_reward_cache_kv, None)

        assert self.can_cfg, 'you need to train with reward token dropout'

        action_flow_without_reward, without_reward_cache_kv = self.forward(
            *args,
            cached_state_keys_values = without_reward_cache,
            **forward_kwargs,
            **kwargs
        )

        update = action_flow_with_reward - action_flow_without_reward

        if remove_parallel_component:
            # from https://arxiv.org/abs/2410.02416

            update_parallel, update_orthog = project(update, action_flow_with_reward)
            update = update_orthog + update_parallel * keep_parallel_frac

        flow_with_reward_cfg = action_flow_with_reward + cond_scale * update

        return flow_with_reward_cfg, (with_reward_cache_kv, without_reward_cache_kv)

    @move_input_tensors_to_device
    def forward_only_vision_language(
        self,
        images: Tensor,         # float[b nv d] | float[b c h w] | float[b c f h w] - vision
        token_ids: Tensor,      # int[b nt] - language
        visual_tokens: Tensor | None = None # float[b nv d]
    ) -> Tensor: # float[b n d]

        device = token_ids.device

        language_tokens = self.token_emb(token_ids)

        # vision

        if exists(visual_tokens):
            pass
        elif exists(self.vit):
            assert images.ndim in {4, 5}
            is_multiple_images = images.ndim == 5

            if is_multiple_images:
                images = rearrange(images, 'b c f h w -> b f c h w')
                images, inverse_pack_image_frames = pack_with_inverse([images], '* c h w')

            with torch.no_grad():
                self.vit.eval()
                visual_tokens = self.vit(images)

            if is_multiple_images:
                visual_tokens, = inverse_pack_image_frames(visual_tokens, '* n d')
                visual_tokens = rearrange(visual_tokens, 'b f n d -> b (f n) d')

        else:
            assert images.ndim in {3, 4}, 'images must be already encoded as (batch, seq, feature dimension) or (batch, frames, seq, feature dimension)'
            
            if images.ndim == 4:
                images = rearrange(images, 'b f n d -> b (f n) d')

            visual_tokens = images

        visual_tokens = self.maybe_to_image_tokens(visual_tokens)
        visual_tokens = visual_tokens / self.token_scale

        # Language tokens scaled by sqrt(dim) to match reference embed_prefix
        language_tokens = language_tokens * self.token_scale

        # concat visual rep with language

        state_tokens, ps = pack([
            visual_tokens,
            language_tokens,
        ], 'b * d')

        # rotary embeddings
        seq_len = state_tokens.shape[-2]
        seq = torch.arange(seq_len, device = device)
        rotary_emb = self.rotary_emb(seq)

        # transformer
        state_tokens = state_tokens * self.token_scale

        num_visual_tokens = visual_tokens.shape[-2]

        for attn, ff, _, _ in self.layers:

            state_attn_out = attn.forward_only_vision_language(
                state_tokens,
                num_visual_tokens = num_visual_tokens,  # Only visual tokens are bidirectional
                is_prefix = True,   # Only bidirectional attend to visual tokens (prefix-LM attention)
                rotary_emb = rotary_emb
            )

            state_tokens = state_tokens + state_attn_out

            state_tokens = ff(state_tokens) + state_tokens

        state_tokens = self.final_norm(state_tokens)

        embed = state_tokens

        logits = self.state_to_logits(embed)

        return logits

    def evolve(
        self,
        return_evo_strat = False,
        **kwargs,
    ):

        evo_strat = EvoStrategy(
            self,
            **{'params_to_optimize': self.action_params_for_evolution(), **kwargs}
        )

        if return_evo_strat:
            return evo_strat

        evo_strat()

    @move_input_tensors_to_device
    def forward_for_policy_loss(
        self,
        images,
        commands,
        joint_state,
        actions,
        old_actor: PiZero,
        advantages: Tensor, # float[b t]
        clip_eps = 0.2,
        norm_eps = 1e-5,
        num_monte_carlo = 2,
        loss_clamp_value = 5.,
        fpo_loss_fn = F.huber_loss,
        **kwargs,
    ):
        batch = actions.shape[0]

        assert not self.is_critic
        assert 'return_actions_flow' not in kwargs

        actor_inputs = dict(
            images = images,
            token_ids = commands,
            joint_state = joint_state,
            actions = actions,
            return_actions_flow = True,
            **kwargs
        )

        # flow matching policy optimization - McAllister et al.
        # https://arxiv.org/abs/2507.21053

        # generate random noise and timesteps - in paper they noted even num monte carlo of 1 did well - lets do 2

        actor_inputs, advantages = tree_map_tensor((actor_inputs, advantages), lambda t: repeat(t, 'b ... -> (b n_mc) ...', n_mc = num_monte_carlo))

        repeated_batch = batch * num_monte_carlo

        repeated_actions = actor_inputs['actions']

        noise = torch.randn_like(repeated_actions)
        times = torch.rand((repeated_batch,), device = self.device)

        actor_inputs.update(noise = noise, times = times)

        target_flow = repeated_actions - noise

        # random times and noises and do flow loss calculation manually for more control

        pred_flow = self.forward(**actor_inputs)

        with torch.no_grad():
            old_actor.eval()
            old_pred_flow = old_actor(**actor_inputs)

        # asymmetric spo - ppo for positive advantages, spo for negative
        # proposed by fpo++ paper for legged robots
        # https://openreview.net/forum?id=BA6n0nmagi

        flow_loss = fpo_loss_fn(pred_flow, target_flow, reduction = 'none')
        old_flow_loss = fpo_loss_fn(old_pred_flow, target_flow, reduction = 'none')

        loss_diff = (flow_loss - old_flow_loss.detach())

        loss_diff_clamped = loss_diff.clamp_max(loss_clamp_value).detach()

        loss_diff = straight_through(loss_diff, loss_diff_clamped)

        # ppo, spo, or both (asymmetric spo)

        ratio = loss_diff.exp()

        advantages = F.layer_norm(advantages, advantages.shape, eps = norm_eps)

        advantages = rearrange(advantages, 'b -> b 1 1')

        calc_spo = lambda: ratio * advantages - advantages.abs() * (ratio - 1.).square() / (2 * clip_eps)

        calc_ppo = lambda: torch.min(ratio * advantages, ratio.clamp(1. - clip_eps, 1. + clip_eps) * advantages)

        if self.use_asymmetric_spo:
            policy_loss = torch.where(advantages >= 0., calc_ppo(), calc_spo())
        elif self.use_spo:
            policy_loss = calc_spo()
        else:
            policy_loss = calc_ppo()

        # sum across actions, then average

        policy_loss = policy_loss.sum(dim = -1)

        return -policy_loss.mean()

    @move_input_tensors_to_device
    def forward_for_critic_loss(
        self,
        *args,
        old_values: Tensor, # float[b]
        advantages: Tensor, # float[b]
        value_clip = True,
        clip_eps = 0.4,
    ):
        assert self.is_critic

        eps = clip_eps
        loss_fn = self.to_critic_value.loss_fn

        forward_kwargs = dict(
            times = torch.ones_like(old_values)
        )

        critic_value, critic_logits = self.forward(*args, **forward_kwargs)

        # derive returns

        advantages = rearrange(advantages, '... -> (...)')
        old_values = rearrange(old_values, '... -> (...)')

        returns = old_values + advantages

        loss = loss_fn(critic_logits, returns, reduction = 'none')

        if not value_clip:
            return loss.mean()

        # maybe value clipping

        clipped_value = old_values + (critic_value - old_values).clamp(-eps, eps)

        clipped_loss = loss_fn(clipped_value, returns, reduction = 'none')

        return torch.max(clipped_loss, loss).mean()

    @move_input_tensors_to_device
    def forward(
        self,
        images: Tensor,                                          # float[b nv d] | float[b c h w] | float[b c f h w] - vision
        token_ids: Tensor,                                       # int[b nt] - language
        joint_state: Tensor,                                     # float[b djs] - joint state
        actions: Tensor | None = None,                           # float[b na da] - action
        times: Tensor | None = None,                             # float[b]
        noise: Tensor | None = None,                             # float[b na da]
        latents: Tensor | None = None,                           # float[d] | float[b d]
        reward_tokens: Tensor | None = None,                     # float[b d]
        internal_state_tokens: Tensor | None = None,             # float[b ns d]
        external_states: tuple[Tensor, ...] | None = None,       # tuple[float[b ...]]
        record_and_return_memory_tokens = False,
        past_recurrent_memory_tokens: Tensor | None = None,      # float[b {self._nm} d]
        task_id: Tensor | int | None = None,                      # int[b]
        task_status: Tensor | None = None,                       # int[b]
        advantage_ids: Tensor | int | None = None,               # int[b]
        return_actions_flow = False,
        return_state_keys_values = False,
        cached_state_keys_values: list[tuple[Tensor, Tensor]] | None = None,
        return_language_loss = True,
        return_action_flow_loss = True,
        knowledge_insulate = False,
        visual_tokens: Tensor | None = None,                     # float[b nv d]
        **kwargs
    ):
        inferencing = exists(cached_state_keys_values)
        assert not (inferencing and not return_actions_flow), 'must be generating action trajectory if receiving cached state key values'

        if not exists(actions) and not self.is_critic:
            return self.sample_actions(images, token_ids, joint_state, **kwargs)

        batch, orig_actions, device = token_ids.shape[0], actions, token_ids.device

        # noising the action for flow matching

        if not exists(times):
            times = self.sample_times_fn((batch,), device = device)

            if self.model_predict_output == 'clean':
                times *= (1. - self.max_timesteps ** -1)

        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        # handle latent genes

        if exists(latents) and latents.ndim == 1:
            latents = repeat(latents, 'd -> b d', b = batch)

        # if not returning the actions predicted flow, assume training and noise the actions for loss

        action_prefix_mask = None

        if not return_actions_flow and not self.is_critic:
            noise = default(noise, torch.randn_like(actions))


            flow = actions - noise
            padded_times = rearrange(times, 'b -> b 1 1')

            actions = noise.lerp(actions, padded_times)

            # if doing training time rtc, train with a random fixed prefix and set times to 1.
            # actually not as simple as the paper makes it seem, as time conditioning is expanded a dimension

            if self.train_time_rtc:
                action_len = actions.shape[-2]

                rand_prefix_len = torch.randint(0, self.train_time_rtc_max_delay, (batch,), device = device)
                action_prefix_mask = lens_to_mask(rand_prefix_len, action_len)

                actions = einx.where('b na, b na d, b na d', action_prefix_mask, orig_actions, actions)
                times = einx.where('b na, , b', action_prefix_mask, 1., times)

        # take care of model output maybe needing a transformation from x0 to flow

        def model_output_clean_to_flow(clean, eps = 1e-2):
            padded_times = rearrange(times, 'b -> b 1 1')

            return (clean - actions) / (1. - padded_times).clamp_min(eps)

        model_output_to_flow = identity if self.model_predict_output == 'flow' else model_output_clean_to_flow

        # actions

        time_cond = self.to_time_cond(times) if self.layer_time_cond else None
        action_tokens = self.to_action_tokens(actions)
        empty_token_action_dim = action_tokens.new_empty((batch, 0, self.dim_action))
        
        if self.time_infused_action_tokens:
            sinusoidal = self.infusion_time_embed(times)
            sinusoidal = sinusoidal[:, None, :].expand_as(action_tokens)
            
            action_tokens = torch.cat([action_tokens, sinusoidal], dim = -1)
            action_tokens = self.to_action_time_fuse(action_tokens)

        # handle maybe latents

        if exists(latents):
            assert self.can_accept_latent

            latent_cond = self.to_latent_cond(latents)

            time_cond = time_cond * (latent_cond + 1.)

        # register tokens

        action_register_tokens = repeat(self.action_register_tokens, '... -> b ...', b = batch)

        # take care of maybe recurrent memory tokens

        assert self.has_recurrent_memories or not exists(past_recurrent_memory_tokens), 'you are asking for memories to be read, but `num_recurrent_memory_tokens` is 0'
        assert self.has_recurrent_memories or not record_and_return_memory_tokens, 'you are asking for memories to be written, but `num_recurrent_memory_tokens` is 0'

        if not exists(past_recurrent_memory_tokens):
            past_recurrent_memory_tokens = actions.new_empty((batch, 0, self.dim))

        if self.has_recurrent_memories:
            write_memory_tokens = repeat(self.memory_tokens, 'nm d -> b nm d', b = batch)
        else:
            write_memory_tokens = actions.new_empty((batch, 0, self.dim))

        # joint state + additional internal states

        joint_state_tokens = empty_token_action_dim

        if not self.pi05:
            joint_state_tokens = self.to_joint_state_tokens(joint_state)
            joint_state_tokens = joint_state_tokens.unsqueeze(1)

        # additional internal state tokens

        if not exists(internal_state_tokens):
            internal_state_tokens = joint_state_tokens.new_empty((batch, 0, self.dim_internal_state))

        internal_state_tokens = self.to_internal_state_tokens(internal_state_tokens)

        # handle memory tokens, both read and write as a tuple of two tensors

        memory_tokens = (past_recurrent_memory_tokens, write_memory_tokens)

        # mem_length = past_recurrent_memory_tokens.shape[-2] + write_memory_tokens.shape[-2]

        # pack into [action registers] [internal + joint states] [actions]

        action_tokens, inverse_pack_action_registers = pack_with_inverse([
            action_register_tokens,
            joint_state_tokens,
            internal_state_tokens,
            action_tokens
        ], 'b * d')

        action_with_registers_length = action_tokens.shape[-2]

        # take care of padding time conditioning if doing training rtc

        if exists(time_cond) and time_cond.ndim == 3:
            orig_action_len = orig_actions.shape[-2]
            time_cond = pad_at_dim(time_cond, (action_with_registers_length - orig_action_len, 0), dim = -2)

        state_tokens = None
        discretized_action_length = 0
        discrete_action_ids = None

        if exists(actions) and self.predict_discretized_action_aux_loss:
            discrete_action_ids = self.discretized_action_tokenizer(actions)
            discrete_action_ids = pad_sequence([tensor(ids) for ids in discrete_action_ids], value = -1)
            discretized_action_length = discrete_action_ids.shape[-1]

        if not inferencing:
            # language

            labels = token_ids[:, 1:]

            language_tokens = self.token_emb(token_ids)

            # vision

            if exists(visual_tokens):
                pass
            elif exists(self.vit):
                assert images.ndim in {4, 5}
                is_multiple_images = images.ndim == 5

                if is_multiple_images:
                    images = rearrange(images, 'b c f h w -> b f c h w')
                    images, inverse_pack_image_frames = pack_with_inverse([images], '* c h w')

                with torch.no_grad():
                    self.vit.eval()
                    visual_tokens = self.vit(images)

                if is_multiple_images:
                    visual_tokens, = inverse_pack_image_frames(visual_tokens, '* n d')
                    visual_tokens = rearrange(visual_tokens, 'b f n d -> b (f n) d')

            else:
                assert images.ndim in {3, 4}, 'images must be already encoded as (batch, seq, feature dimension) or (batch, frames, seq, feature dimension)'

                if images.ndim == 4:
                    images = rearrange(images, 'b f n d -> b (f n) d')

                visual_tokens = images

            visual_tokens = self.maybe_to_image_tokens(visual_tokens)

            visual_tokens = visual_tokens / self.token_scale

            # empty

            empty_token = visual_tokens.new_empty((batch, 0, self.dim))

            # maybe reward tokens

            if not exists(reward_tokens):
                reward_tokens = repeat(self.null_reward_token, 'd -> b 1 d', b = batch)

            # maybe advantage tokens

            if self.can_advantage_token_cond:
                if exists(advantage_ids):
                    if isinstance(advantage_ids, int):
                        advantage_ids = full((batch,), advantage_ids, device = self.device)

                    advantage_tokens = self.advantage_embed(advantage_ids)
                else:
                    advantage_tokens = repeat(self.null_advantage_token, 'd -> b 1 d', b = batch)
            else:
                advantage_tokens = empty_token

            # maybe dropout reward or advantage tokens for cfg

            if self.training and sample(self.reward_tokens_dropout_prob):
                reward_tokens = repeat(self.null_reward_token, 'd -> b 1 d', b = batch)

            if self.training and sample(self.advantage_tokens_dropout_prob):
                advantage_tokens = repeat(self.null_advantage_token, 'd -> b 1 d', b = batch)

            # maybe task token

            if self.has_task_cond:
                if exists(task_id):
                    if isinstance(task_id, int):
                        task_id = full((batch,), task_id, device = self.device)
                    task_token = self.task_emb(task_id)
                else:
                    task_token = repeat(self.null_task_token, 'd -> b 1 d', b = batch)

            else:
                task_token = empty_token

            # handle maybe discretized action prediction

            discrete_action_tokens = empty_token

            if exists(discrete_action_ids):
                discrete_action_ids_with_start = pad_left_at_dim(discrete_action_ids + 1, 1)
                discrete_action_ids_in_pack, target_discrete_action_ids = discrete_action_ids_with_start[:, :-1], discrete_action_ids_with_start[:, 1:]

                discrete_action_tokens = self.discrete_action_embeds(discrete_action_ids_in_pack)

            # additional external states

            if exists(external_states):
                external_state_tokens = [encode(external_state) for encode, external_state in zip(self.external_state_encoders, external_states)]
                external_state_tokens, _ = pack(external_state_tokens, 'b * d')

            else:
                external_state_tokens = visual_tokens.new_empty((batch, 0, self.dim))

            scale = self.token_scale

            language_tokens = language_tokens * scale
            reward_tokens = reward_tokens * scale
            advantage_tokens = advantage_tokens * scale
            task_token = task_token * scale
            external_state_tokens = external_state_tokens * scale
            discrete_action_tokens = discrete_action_tokens * scale

            # concat visual rep with language

            state_tokens, inverse_packed_states = pack_with_inverse([
                external_state_tokens,
                visual_tokens,
                language_tokens,
                reward_tokens,
                advantage_tokens,
                task_token,
                discrete_action_tokens
            ], 'b * d')

        # take care of masking for variable lengthed states, starting with the language tokens

        # which then leads to proper rotary embeddings

        command_length = token_ids.shape[-1]

        language_mask = token_ids != self.lm_pad_id

        if inferencing:
            state_length = cached_state_keys_values[0][0].shape[-2]
        else:
            state_length = state_tokens.shape[-2]

        mask = F.pad(language_mask, (state_length - command_length, action_with_registers_length), value = True)
        mask = F.pad(mask, (past_recurrent_memory_tokens.shape[-2], write_memory_tokens.shape[-2]), value = True)

        # construct ar_mask for segment based bidirectional attention

        ar_mask = torch.zeros_like(mask, dtype = torch.int)

        if not inferencing:
            # prefix segment (0)
            prefix_len = state_length
            
            # registers + state + internal_state
            
            num_registers = self.action_register_tokens.shape[-2]
            num_joint_state = joint_state_tokens.shape[-2]
            num_internal_state = internal_state_tokens.shape[-2]

            # prefix ends at state_length
            
            if not self.pi05:
                # suffix_att_masks = [1, 1, 0, 0, ...] meaning:
                # - joint_state (pos 0): cumsum=1, can only attend to itself within suffix
                # - first_action (pos 1): cumsum=2, starts a new segment
                # - rest of actions: cumsum=2, all bidirectional within segment
                
                # Segment 1: joint_state
                if (prefix_len + num_registers) < ar_mask.shape[-1]:
                    ar_mask[:, prefix_len + num_registers] = 1
                
                # Segment 2: first action token
                first_action_pos = prefix_len + num_registers + num_joint_state + num_internal_state
                if first_action_pos < ar_mask.shape[-1]:
                    ar_mask[:, first_action_pos] = 1
            else:
                # PI0.5: Segment 1 starts at actions
                action_tokens_start = prefix_len + num_registers
                if action_tokens_start < ar_mask.shape[-1]:
                    ar_mask[:, action_tokens_start] = 1

        # rotary embeddings
        seq = (mask.float().cumsum(dim = -1) - 1).clamp_min(0)
        rotary_emb = self.rotary_emb(seq)

        rotary_emb = rearrange(rotary_emb, 'b n d -> b 1 n d')

        # multimodal prefix bidirectional attention

        multimodal_prefix_bidirectional_length = 0
        multimodal_prefix_bidirectional_start = 0

        if not inferencing:
            multimodal_prefix_bidirectional_length = state_tokens.shape[-2]
            multimodal_prefix_bidirectional_start = 0

        # prepare maybe flex attention

        flex_attn_fn = None

        if not inferencing and self.use_flex_attn and state_tokens.is_cuda:

            prefix_length = state_tokens.shape[-2]
            seq_len = prefix_length + action_tokens.shape[-2]

            block_mask = create_block_mask(
                create_pizero_attn_mask(
                    prefix_length,
                    mask = mask,
                    prefix_bidirectional_length = multimodal_prefix_bidirectional_length,
                    prefix_bidirectional_start = multimodal_prefix_bidirectional_start,
                    discretized_action_length = discretized_action_length
                ),
                Q_LEN = seq_len,
                KV_LEN = seq_len,
                device = state_tokens.device,
                _compile = True,
            )

            score_mod_fn = None

            flex_attn_fn = partial(
                flex_attention,
                block_mask = block_mask,
                score_mod = score_mod_fn
            )

        # state keys and values for caching during inference

        cached_state_key_values_iter = iter(default(cached_state_keys_values, []))

        if exists(state_tokens):
            state_tokens = state_tokens * self.token_scale

        action_tokens = action_tokens * (self.dim_action ** 0.5)

        if not inferencing:

            next_state_cached_keys_values = []

            for (
                (attn, state_ff, actions_ff, memories_ff),
                (attn_ada_rmsnorm, ff_ada_rmsnorm),
            ) in zip(self.layers, self.cond_layers):
                # joint attention

                action_residual = action_tokens

                if self.layer_time_cond:
                    action_tokens, gate = attn_ada_rmsnorm(action_tokens, time_cond, return_gate = True)
                else:
                    action_tokens = attn_ada_rmsnorm(action_tokens)
                    gate = None

                (state_attn_out, actions_attn_out, *maybe_mem_out), (state_keys, state_values, action_keys, action_values) = attn(
                    state_tokens,
                    action_tokens,
                    multimodal_prefix_bidirectional_length = multimodal_prefix_bidirectional_length,
                    multimodal_prefix_bidirectional_start = multimodal_prefix_bidirectional_start,
                    rotary_emb = rotary_emb,
                    flex_attn_fn = flex_attn_fn,
                    mask = mask,
                    ar_mask = ar_mask,
                    return_keys_values = True,
                    knowledge_insulate = knowledge_insulate,
                    memories = memory_tokens,
                    discretized_action_length = discretized_action_length
                )

                next_state_cached_keys_values.append((state_keys, state_values))

                action_attn_out = actions_attn_out

                state_tokens = state_tokens + state_attn_out

                if exists(gate):
                    action_attn_out = action_attn_out * gate

                action_tokens = action_residual + action_attn_out

                if self.has_recurrent_memories:
                    (read_mem_attn_out, write_mem_attn_out), = maybe_mem_out
                    read_mem, write_mem = memory_tokens

                    memory_tokens = (read_mem + read_mem_attn_out, write_mem + write_mem_attn_out)

                # state feedforward

                state_tokens_out = state_ff(state_tokens)

                state_tokens = state_tokens + state_tokens_out

                # action feedforward

                action_ff_residual = action_tokens

                if self.layer_time_cond:
                    action_tokens, gate = ff_ada_rmsnorm(action_tokens, time_cond, return_gate = True)
                else:
                    action_tokens = ff_ada_rmsnorm(action_tokens)
                    gate = None

                action_tokens_out = actions_ff(action_tokens)

                action_tokens_out = action_tokens_out
                # action_tokens_out = ff_ada_layerscale(action_tokens_out, time_cond)

                if exists(gate):
                    action_tokens_out = action_tokens_out * gate

                action_tokens = action_ff_residual + action_tokens_out

                # maybe memory feedforward

                if self.has_recurrent_memories:
                    memory_tokens, unpack_memory = pack_with_inverse(memory_tokens, 'b * d')

                    memory_tokens = memories_ff(memory_tokens) + memory_tokens

                    memory_tokens = unpack_memory(memory_tokens)

        else:

            assert exists(cached_state_keys_values) and len(cached_state_keys_values) > 0

            next_state_cached_keys_values = cached_state_keys_values

            for (
                (attn, state_ff, actions_ff, memories_ff),
                (attn_ada_rmsnorm, ff_ada_rmsnorm),
            ) in zip(self.layers, self.cond_layers):
                # actions attention

                action_residual = action_tokens

                if self.layer_time_cond:
                    action_tokens, gate = attn_ada_rmsnorm(action_tokens, time_cond, return_gate = True)
                else:
                    action_tokens = attn_ada_rmsnorm(action_tokens)
                    gate = None

                actions_attn_out, (state_keys, state_values, action_keys, action_values) = attn.forward_actions_with_cached_state(
                    action_tokens,
                    cached_state_keys_values = next(cached_state_key_values_iter),
                    rotary_emb = rotary_emb,
                    mask = mask,
                    return_keys_values = True,
                    discretized_action_length = discretized_action_length
                )

                if exists(gate):
                    actions_attn_out = actions_attn_out * gate

                action_tokens = action_residual + actions_attn_out

                # actions feed forward

                action_ff_residual = action_tokens

                if self.layer_time_cond:
                    action_tokens, gate = ff_ada_rmsnorm(action_tokens, time_cond, return_gate = True)
                else:
                    action_tokens = ff_ada_rmsnorm(action_tokens)
                    gate = None

                action_out = actions_ff(action_tokens)

                # action_out = ff_ada_layerscale(action_out, time_cond)  # Removed: no layerscale

                if exists(gate):
                    action_out = action_out * gate

                action_tokens = action_ff_residual + action_out

                # maybe memory feed forward

                if self.has_recurrent_memories:
                    memory_tokens, unpack_memory = pack_with_inverse(memory_tokens, 'b * d')

                    memory_tokens = memories_ff(memory_tokens) + memory_tokens

                    memory_tokens = unpack_memory(memory_tokens)


        if not inferencing:
            # unpack and unembed to predictions

            state_tokens = self.final_norm(state_tokens)

            _, visual_tokens, tokens, *_, maybe_discrete_action_tokens = inverse_packed_states(state_tokens, 'b * d')

        *_, action_tokens = inverse_pack_action_registers(action_tokens)


        # memories

        read_memories, written_memory_tokens = memory_tokens

        # writeable memories norm

        if self.has_recurrent_memories:
            written_memory_tokens = self.final_norm_write_memories(written_memory_tokens)

        # final actions norm

        if self.layer_time_cond:
            curr_time_cond = time_cond
            if curr_time_cond.ndim == 3:
                curr_time_cond = curr_time_cond[:, -action_tokens.shape[-2]:]

            action_embeds, _ = self.final_actions_norm(action_tokens, curr_time_cond, return_gate = True)
        else:
            action_embeds = self.final_actions_norm(action_tokens)


        # pool the action embeds and project if critic loss

        if self.is_critic:
            action_embeds = reduce(action_embeds, 'b n d -> b d', 'mean')

            return self.to_critic_value(action_embeds, return_value_and_logits = True)

        # validate loss being returned

        assert return_language_loss or return_action_flow_loss or exists(task_status)

        # task status cross entropy loss

        if exists(task_status):
            assert exists(self.to_task_status), '`predict_task_status_head` must be set to True on `PiZero`'

            pooled_action_embeds = reduce(action_embeds, 'b n d -> b d', 'mean')
            pred_task_status = self.to_task_status(pooled_action_embeds)

            pred_task_status_loss = F.cross_entropy(pred_task_status, task_status)

        # flow loss for actions tokens

        model_output = self.actions_to_pred_flow(action_embeds)

        pred_actions_flow = model_output_to_flow(model_output)

        if return_actions_flow:

            if not return_state_keys_values and not record_and_return_memory_tokens:
                return pred_actions_flow

            if not return_state_keys_values:
                return pred_actions_flow, written_memory_tokens

            return pred_actions_flow, next_state_cached_keys_values

        flow_loss = self.zero

        if return_action_flow_loss:
            flow_loss = self.loss_fn(pred_actions_flow, flow)

            flow_loss = reduce(flow_loss, 'b ... d -> b ...', 'mean')

            # maybe mask out
            # for 1. the loss for invalid task labels from GR-3 paper for improved language following
            # for 2. the train time rtc

            is_not_invalid_mask = None
            if exists(task_status):
                is_not_invalid_mask = task_status != self.task_status_is_invalid

            is_not_action_prefix_mask = maybe(torch.logical_not)(action_prefix_mask)

            mask = and_masks([is_not_invalid_mask, is_not_action_prefix_mask])

            # mask out

            if exists(mask):
                flow_loss = flow_loss[mask]

            # average

            flow_loss = flow_loss.mean()

        # maybe discrete action embed loss

        discrete_action_ar_loss = self.zero

        if not is_tensor_empty(maybe_discrete_action_tokens):

            pred_discrete_action_logits = self.to_discrete_action_pred(maybe_discrete_action_tokens)

            discrete_action_ar_loss = F.cross_entropy(
                rearrange(pred_discrete_action_logits, 'b n l -> b l n'),
                target_discrete_action_ids,
                ignore_index = self.discrete_action_pad_id
            )

        # language cross entropy loss

        language_loss = self.zero

        if return_language_loss:
            language_logits = self.state_to_logits(tokens)

            language_loss = F.cross_entropy(
                rearrange(language_logits[:, :-1], 'b n l -> b l n'),
                labels.long(),
                ignore_index = self.lm_pad_id
            )

        # loss breakdown

        loss_breakdown = (language_loss, discrete_action_ar_loss, flow_loss)

        # total loss and return breakdown

        total_loss = (
            language_loss * self.lm_loss_weight +
            flow_loss * self.flow_loss_weight +
            discrete_action_ar_loss * self.predict_discrete_action_loss_weight
        )

        # add the task status loss if needed

        if exists(task_status):
            loss_breakdown = (*loss_breakdown, pred_task_status_loss)

            total_loss = (
                total_loss +
                pred_task_status_loss * self.task_status_loss_weight
            )

        # returning

        if not record_and_return_memory_tokens:
            return total_loss, loss_breakdown

        return total_loss, loss_breakdown, written_memory_tokens

# generalized advantage estimate

GAEReturn = namedtuple('GAEReturn', ('advantages', 'returns'))

@torch.no_grad()
def calc_generalized_advantage_estimate(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    padded_values = F.pad(values, (0, 1), value = 0.)

    values, values_next = padded_values[..., :-1], padded_values[..., 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    advantages = scan(gates, delta)

    returns = advantages + values

    return GAEReturn(advantages, returns)

# agent

class Agent(Module):
    def __init__(
        self,
        model: PiZero,
        optim_klass = AdamAtan2,
        num_latent_genes = 1,
        actor_lr = 3e-4,
        critic_lr = 3e-4,
        actor_weight_decay = 1e-3,
        critic_weight_decay = 1e-3,
        max_grad_norm = 0.5,
        actor_fpo_loss_fn = F.huber_loss,
        critic_use_discrete_bins = False,
        actor_optim_kwargs: dict = dict(),
        critic_optim_kwargs: dict = dict(),
        latent_gene_pool_kwargs: dict = dict(
            frac_tournaments = 0.5
        )
    ):
        super().__init__()

        # Wang et al. 2025

        assert num_latent_genes >= 1
        evolutionary_learning = num_latent_genes > 1

        dim_latent = model.dim if evolutionary_learning else None

        self.latent_gene_pool = LatentGenePool(dim_latent = dim_latent, num_latents = num_latent_genes, **latent_gene_pool_kwargs) if evolutionary_learning else None
        self.has_gene_pool = evolutionary_learning

        # init actor critic, taking into account model may not have probabilistic flow to start off with, and determine whether it needs to be reinstantiated for latent conditioning

        actor = model

        if evolutionary_learning:
            actor = model.create_actor(dim_latent = dim_latent)

        self.actor = actor
        self.critic = actor.create_critic(critic_use_discrete_bins = critic_use_discrete_bins)

        # fpo related

        self.actor_fpo_loss_fn = actor_fpo_loss_fn

        # gradient clipping

        self.max_grad_norm = max_grad_norm

        # optimizers

        self.actor_optim = optim_klass(self.actor.parameters(), lr = actor_lr, weight_decay = actor_weight_decay, **actor_optim_kwargs)
        self.critic_optim = optim_klass(self.critic.parameters(), lr = critic_lr, weight_decay = critic_weight_decay, **critic_optim_kwargs)

    def take_genetic_algorithm_step_(self, fitnesses):
        if not self.has_gene_pool:
            return

        self.latent_gene_pool.genetic_algorithm_step(fitnesses)

    def forward(
        self,
        memories
    ):
        raise NotImplementedError

# online

class EFPO(Module):
    def __init__(
        self,
        agent_or_model: Agent | PiZero,
        cpu = False,
        accelerate_kwargs: dict = dict(),
        replay_buffer_folder: str | Path = './efpo_replay_buffer',
        max_replay_buffer_episodes = 1000,
        max_replay_buffer_steps = 1000,
        action_penalty_thresholds: list[float] | Tensor | None = None,
        action_penalty_weight = 1e-1
    ):
        super().__init__()
        self.accelerate = Accelerator(cpu = cpu, **accelerate_kwargs)

        if isinstance(agent_or_model, PiZero):
            agent = Agent(agent_or_model)
        else:
            agent = agent_or_model

        self.agent = agent

        (
            agent.actor,
            agent.critic,
            agent.actor_optim,
            agent.critic_optim
        ) = self.accelerate.prepare(
            agent.actor,
            agent.critic,
            agent.actor_optim,
            agent.critic_optim
        )

        self.register_buffer('step', tensor(0))

        # replay buffer

        if not isinstance(replay_buffer_folder, Path):
            replay_buffer_folder = Path(replay_buffer_folder)

        self.replay_buffer_folder = replay_buffer_folder
        self.replay_buffer_folder.mkdir(parents = True, exist_ok = True)

        self.max_replay_buffer_episodes = max_replay_buffer_episodes
        self.max_replay_buffer_steps = max_replay_buffer_steps

        # penalize actions out of bounds - helps with FPO convergence in personal experiments

        if isinstance(action_penalty_thresholds, list):
            action_penalty_thresholds = tensor(action_penalty_thresholds)

        self.register_buffer('action_penalty_thresholds', action_penalty_thresholds)

        self.action_penalty_weight = action_penalty_weight

    @property
    def unwrapped_actor(self):
        return self.accelerate.unwrap_model(self.agent.actor)

    @property
    def unwrapped_critic(self):
        return self.accelerate.unwrap_model(self.agent.critic)

    def log(self, **data_kwargs):
        return self.accelerate.log(data_kwargs, step = self.step.item())

    @torch.no_grad()
    def gather_experience_from_env(
        self,
        env,
        steps,
        num_episodes = 1,
        trajectory_length = 16,
        flow_sampling_steps = 4,
        task_id = None,
        **sampling_kwargs
    ):
        self.agent.eval()

        actor = self.unwrapped_actor
        critic = self.unwrapped_critic

        # Use ReplayBuffer for storing experiences

        replay_buffer = ReplayBuffer(
            self.replay_buffer_folder,
            max_episodes = num_episodes,
            max_timesteps = steps,
            meta_fields = dict(
                task_id = 'int'
            ),
            fields = dict(
                images = ('float', (3, env.num_images, *env.image_shape)),
                text = ('int', (env.max_text_len,)),
                internal = ('float', (env.joint_dim,)),
                reward = 'float',
                terminated = 'bool',
                actions = ('float', (trajectory_length, actor.dim_action_input)),
                actions_last_step = ('float', (trajectory_length, actor.dim_action_input)),
                value = 'float',
                advantage = 'float',
                gae_return = 'float'
            ),
            overwrite = True
        )

        for _ in range(num_episodes):

            states = env.reset()

            actions_last_step = torch.zeros((trajectory_length, actor.dim_action_input), device = torch.device('cpu'))

            one_episode_kwargs = dict(task_id = task_id)
            one_episode_kwargs = {k: v for k, v in one_episode_kwargs.items() if exists(v)}

            with replay_buffer.one_episode(**one_episode_kwargs):

                for _ in range(steps):

                    sampled_actions, values = temp_batch_dim(actor)(
                        *states,
                        trajectory_length = trajectory_length,
                        steps = flow_sampling_steps,
                        critic = critic,
                        actions_last_step = actions_last_step,
                        **sampling_kwargs
                    )

                    next_states, reward, truncated, terminated = env.step(sampled_actions)

                    # maybe penalize actions

                    if exists(self.action_penalty_thresholds):
                        penalty = (sampled_actions.abs() - self.action_penalty_thresholds).relu().square().sum(dim = -1).mean()
                        reward = reward - penalty * self.action_penalty_weight

                    images, text, internal = states

                    replay_buffer.store(
                        images = images,
                        text = text,
                        internal = internal,
                        reward = reward,
                        terminated = terminated,
                        actions = sampled_actions,
                        actions_last_step = actions_last_step,
                        value = values
                    )

                    states = next_states
                    actions_last_step = sampled_actions.cpu()

        # compute GAE per-episode and store in buffer
        # this must be done at the trajectory level, not on shuffled batches

        for episode_idx in range(replay_buffer.num_episodes):
            episode_len = replay_buffer.episode_lens[episode_idx]

            if episode_len == 0:
                continue

            rewards = from_numpy(replay_buffer.data['reward'][episode_idx, :episode_len])
            values = from_numpy(replay_buffer.data['value'][episode_idx, :episode_len])
            terminated = from_numpy(replay_buffer.data['terminated'][episode_idx, :episode_len])

            # masks: 0 where episode terminates, 1 otherwise (for bootstrapping)
            masks = (~terminated).float()

            advantages, returns = calc_generalized_advantage_estimate(
                rewards, values, masks, use_accelerated = False
            )

            replay_buffer.data['advantage'][episode_idx, :episode_len] = advantages.numpy()
            replay_buffer.data['gae_return'][episode_idx, :episode_len] = returns.numpy()

        self.accelerate.wait_for_everyone()

        return replay_buffer

    def learn_agent(
        self,
        memories: ReplayBuffer,
        fitnesses = None,
        epochs = 2,
        batch_size = 16
    ):
        actor_fpo_loss_fn = self.agent.actor_fpo_loss_fn

        self.agent.train()

        # dataset and dataloader

        dataloader = memories.dataloader(
            batch_size = batch_size,
            timestep_level = True,
            shuffle = True,
            fieldname_map = dict(
                text = 'token_ids',
                internal = 'joint_state'
            )
        )

        # copy of old actor

        old_actor = deepcopy(self.unwrapped_actor)

        # training loop

        for _ in range(epochs):
            for batch in dataloader:

                images, commands, joint_state, actions, actions_last_step, old_values, advantages = [batch[k] for k in ('images', 'token_ids', 'joint_state', 'actions', 'actions_last_step', 'value', 'advantage')]

                # advantages are pre-computed per-episode before training

                actor_loss = self.agent.actor.forward_for_policy_loss(
                    images,
                    commands,
                    joint_state,
                    actions,
                    old_actor = old_actor,
                    advantages = advantages,
                    fpo_loss_fn = actor_fpo_loss_fn
                )

                actor_loss.backward()

                self.log(actor_loss = actor_loss.item())

                self.accelerate.clip_grad_norm_(self.agent.actor.parameters(), self.agent.max_grad_norm)

                self.agent.actor_optim.step()
                self.agent.actor_optim.zero_grad()

                critic_loss = self.agent.critic.forward_for_critic_loss(
                    images,
                    commands,
                    joint_state,
                    actions_last_step,
                    old_values = old_values,
                    advantages = advantages,
                )

                critic_loss.backward()

                self.log(critic_loss = critic_loss.item())

                self.accelerate.clip_grad_norm_(self.agent.critic.parameters(), self.agent.max_grad_norm)

                self.agent.critic_optim.step()
                self.agent.critic_optim.zero_grad()

            if exists(fitnesses):
                self.log(fitnesses = fitnesses)

                self.agent.take_genetic_algorithm_step_(fitnesses)

        self.step.add_(1)

# offline

# moved to replay_buffer.py

# 0.6 related hyperparameters

class TrainConfig(BaseModel):
    advantage_lookahead: int = Field(..., description = "-1 for full episode.")
    positive_data_fraction: float = Field(..., ge = 0.0, le = 1.0)
    percentile_cutoff: int = Field(..., ge = 0, le = 100)

class TaskConfig(BaseModel):
    max_episode_length: int
    pretrain: TrainConfig
    finetune: TrainConfig

# replay datasets

# ReplayDataset removed in favor of built-in memmap_replay_buffer.ReplayDatasetTimestep

class JoinedReplayDataset(Dataset):
    def __init__(
        self,
        datasets: list[Dataset],
        meta_buffer: ReplayBuffer
    ):
        super().__init__()
        self.datasets = datasets
        self.meta_buffer = meta_buffer

        meta_episode_offset = 0
        self.meta_episode_offsets = []

        for dataset in datasets:
            if hasattr(dataset, 'return_indices'):
                dataset.return_indices = True

            self.meta_episode_offsets.append(meta_episode_offset)
            meta_episode_offset += len(dataset.valid_episodes)

        self.concat_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.concat_dataset.cumulative_sizes, idx)
        
        local_idx = idx
        if dataset_idx > 0:
            local_idx = idx - self.concat_dataset.cumulative_sizes[dataset_idx - 1]

        dataset = self.datasets[dataset_idx]
        data = dataset[local_idx]

        # Map to meta buffer
        episode_id, timestep_index = data.get('_indices').unbind(dim = -1)
        
        # We need relative episode index within the dataset's valid episodes
        relative_episode_idx = torch.searchsorted(dataset.valid_episodes, episode_id)
        
        meta_episode_id = self.meta_episode_offsets[dataset_idx] + relative_episode_idx
        
        # Get meta fields (value, advantages, advantage_ids)
        for field in self.meta_buffer.fieldnames:
            meta_data = self.meta_buffer.data[field][meta_episode_id.item(), timestep_index.item()]
            data[field] = tensor(meta_data)

        return data

# recap config

class RecapConfig(BaseModel):
    tasks: dict[str, TaskConfig]
    task_fail_penalty: float

DEFAULT_RECAP_CONFIG = dict(
    tasks = dict(
        laundry_tshirt_shorts = dict(
            max_episode_length = 200,
            pretrain = dict(
                advantage_lookahead = -1,
                positive_data_fraction = 0.30,
                percentile_cutoff = 70
            ),
            finetune = dict(
                advantage_lookahead = 50,
                positive_data_fraction = 0.10,
                percentile_cutoff = 90
            )
        ),
        laundry_diverse = dict(
            max_episode_length = 500,
            pretrain = dict(
                advantage_lookahead = -1,
                positive_data_fraction = 0.30,
                percentile_cutoff = 70
            ),
            finetune = dict(
                advantage_lookahead = 50,
                positive_data_fraction = 0.40,
                percentile_cutoff = 60
            )
        ),
        cafe_espresso = dict(
            max_episode_length = 200,
            pretrain = dict(
                advantage_lookahead = -1,
                positive_data_fraction = 0.30,
                percentile_cutoff = 70
            ),
            finetune = dict(
                advantage_lookahead = 50,
                positive_data_fraction = 0.40,
                percentile_cutoff = 60
            )
        ),
        box_assembly = dict(
            max_episode_length = 600,
            pretrain = dict(
                advantage_lookahead = -1,
                positive_data_fraction = 0.30,
                percentile_cutoff = 70
            ),
            finetune = dict(
                advantage_lookahead = 50,
                positive_data_fraction = 0.40,
                percentile_cutoff = 60
            )
        ),
        laundry_failure_removal_ablation = dict(
            max_episode_length = 200,
            pretrain = dict(
                advantage_lookahead = -1,
                positive_data_fraction = 0.30,
                percentile_cutoff = 70
            ),
            finetune = dict(
                advantage_lookahead = 50,
                positive_data_fraction = 0.40,
                percentile_cutoff = 60
            )
        )
    ),
    task_fail_penalty = -10  # they use a big negative constant for failures, when labeling the experiences - value network is bounded from -1. to 0 anyways so it works out
)

class PiZeroSix(Module):
    def __init__(
        self,
        agent_or_model: PiZero | Agent,
        pretrain_data: ReplayBuffer | None = None,
        config: dict | RecapConfig = DEFAULT_RECAP_CONFIG,
        cpu = False,
        workspace_folder = './workspace',
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        # accelerate

        self.accelerate = Accelerator(cpu = cpu, **accelerate_kwargs)

        # config

        if isinstance(config, dict):
            config = RecapConfig(**config)

        self.config = config

        # pretrain data

        self.pretrain_data = pretrain_data

        # task ids for now are implicitly the order of the keys

        self.task_strings = list(config.tasks.keys())

        # agent

        if isinstance(agent_or_model, PiZero):
            agent = Agent(agent_or_model, critic_use_discrete_bins = True)
        else:
            agent = agent_or_model

        self.agent = agent

        assert agent.actor.can_advantage_token_cond, '`num_advantage_tokens` must be set to greater than 1 to employ the Pi0.6 learning from experience scheme'

        assert agent.critic.critic_use_discrete_bins, 'they use discretized values'

        # wrapping

        agent.actor, agent.critic = self.accelerate.prepare(agent.actor, agent.critic)

        # positive advantage id

        self.positive_advantage_id = agent.actor.num_advantage_tokens - 1 # assume the last advantage token is the highest positive, if there is gradation

        # labeling

        self.register_buffer('task_fail_penalty', tensor(config.task_fail_penalty))

        # a folder that keeps track of the pretrained model, and for the fine tuning states of all the tasks (which iteration it is on, with 0th being the SFT stage where advantage is fixed to positive)

        self.workspace_folder = Path(workspace_folder)
        self.workspace_folder.mkdir(exist_ok = True, parents = True)

        self.pretrained_actor_path = self.workspace_folder / 'pretrained-actor.pt'
        self.pretrained_critic_path = self.workspace_folder / 'pretrained-critic.pt'

        if exists(self.pretrain_data):
            num_episodes = pretrain_data.num_episodes
            task_ids = np.unique(pretrain_data.meta_data['task_id'][:num_episodes]).tolist()
        else:
            task_ids = list(range(len(self.task_strings)))

        # very simply, each specialized task will have a subfolder within the workspace folder
        # this will contain folders enumerated from 0 to K times, where K is the improvement iteration for the RECAP algorithm in pi0.6

        assert all([0 <= task_id < len(self.task_strings) for task_id in task_ids]), 'invalid task_id discovered in replay buffer'

        self.task_id_name = {task_id: self.task_strings[task_id] for task_id in task_ids}
        self.task_name_id = {task_name: task_id for task_id, task_name in self.task_id_name.items()}

        # create all folders if not exist

        self.task_workspaces = dict()

        for task_id, task_name in self.task_id_name.items():
            task_workspace_folder = self.workspace_folder / task_name
            task_workspace_folder.mkdir(exist_ok = True)

            self.task_workspaces[task_id] = task_workspace_folder

    def print(self, *args, **kwargs):
        return self.accelerate.print(*args, **kwargs)

    @property
    def unwrapped_actor(self):
        return self.accelerate.unwrap_model(self.agent.actor)

    @property
    def unwrapped_critic(self):
        return self.accelerate.unwrap_model(self.agent.critic)

    @property
    def is_main(self):
        return self.accelerate.is_main_process

    def save_pretrained(self, overwrite = True):
        actor, critic = self.unwrapped_actor, self.unwrapped_critic

        if not self.is_main:
            return

        assert overwrite or not self.pretrained_actor_path.exists()
        assert overwrite or not self.pretrained_critic_path.exists()

        torch.save(actor.state_dict(), str(self.pretrained_actor_path))
        torch.save(critic.state_dict(), str(self.pretrained_critic_path))

    def load_pretrained(self):
        actor, critic = self.unwrapped_actor, self.unwrapped_critic

        assert self.pretrained_actor_path.exists(), 'pretrained actor does not exist in the workspace'
        assert self.pretrained_critic_path.exists(), 'pretrained critic does not exist in the workspace'

        actor_weights = torch.load(str(self.pretrained_actor_path), weights_only = True)
        critic_weights = torch.load(str(self.pretrained_critic_path), weights_only = True)

        actor.load_state_dict(actor_weights)
        critic.load_state_dict(critic_weights)

    def save(
        self,
        folder: str | Path,
        overwrite = True
    ):
        actor, critic = self.unwrapped_actor, self.unwrapped_critic

        if isinstance(folder, str):
            folder = Path(folder)

        folder.mkdir(exist_ok = True, parents = True)

        if not self.is_main:
            return

        actor_path = folder / 'actor.pt'
        critic_path = folder / 'critic.pt'

        assert overwrite or not actor_path.exists()
        assert overwrite or not critic_path.exists()

        torch.save(actor.state_dict(), str(actor_path))
        torch.save(critic.state_dict(), str(critic_path))

    def load(
        self,
        folder: str | Path,
        overwrite = True
    ):
        actor, critic = self.unwrapped_actor, self.unwrapped_critic

        if isinstance(folder, str):
            folder = Path(folder)

        folder.mkdir(exist_ok = True, parents = True)

        actor_path = folder / 'actor.pt'
        critic_path = folder / 'critic.pt'

        assert overwrite or not actor_path.exists()
        assert overwrite or not critic_path.exists()

        actor_state_dict = torch.load(str(actor_path), weights_only = True)
        critic_state_dict = torch.load(str(critic_path), weights_only = True)

        actor.load_state_dict(actor_state_dict)
        critic.load_state_dict(critic_state_dict)

    def get_last_task_finetune_folder(self, task_id):
        task_workspace = self.task_workspaces[task_id]

        finetune_ids = [int(folder.name) for folder in task_workspace.glob('*/') if folder.name.isdigit()]

        assert len(finetune_ids) > 0, 'you need to run `.sft` first to generate the finetuned specialist (pretrain data filtered by task) before initiating rollouts with environment for recap algorithm'

        finetune_id = max(finetune_ids)

        return task_workspace / str(finetune_id)

    def pretrain(
        self,
        num_train_steps_actor = 100,
        num_train_steps_critic = 100,
        batch_size = 4
    ):
        assert exists(self.pretrain_data)

        pretrain_data = self.pretrain_data
        
        # Create temporary meta-buffer for pretraining
        
        values_and_advantages = ReplayBuffer(
            self.workspace_folder / 'pretrain_values_and_advantages',
            max_episodes = pretrain_data.num_episodes,
            max_timesteps = pretrain_data.max_timesteps,
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
        
        values_and_advantages.meta_data['task_id'][:] = pretrain_data.meta_data['task_id'][:pretrain_data.num_episodes]
        values_and_advantages.episode_lens[:] = pretrain_data.episode_lens[:pretrain_data.num_episodes]
        values_and_advantages.num_episodes = pretrain_data.num_episodes

        dataset = self.dataset(pretrain_data, return_indices = True)
        joined_dataset = JoinedReplayDataset([dataset], values_and_advantages)

        self.calculate_return_or_advantages_(dataset, type = 'returns', mode = 'pretrain', destination_buffer = values_and_advantages)

        self.train_value_network(joined_dataset, num_train_steps = num_train_steps_critic, batch_size = batch_size)

        self.update_buffer_values_(dataset, batch_size = batch_size, destination_buffer = values_and_advantages)

        self.calculate_return_or_advantages_(dataset, type = 'advantages', mode = 'pretrain', destination_buffer = values_and_advantages)

        self.set_advantage_token_id_(values_and_advantages, mode = 'pretrain')

        self.train_policy_network(joined_dataset, num_train_steps = num_train_steps_actor, batch_size = batch_size)

        self.save_pretrained()

    def sft(
        self,
        task_id_or_name: int | str,
        num_train_steps_actor = 100,
        num_train_steps_critic = 100,
        batch_size = 4,
        recalculate_advantages_with_finetuned_critic = False
    ):
        assert exists(self.pretrain_data)

        if isinstance(task_id_or_name, int):
            task_id = task_id_or_name
            assert task_id in self.task_id_name
            task_name = self.task_id_name[task_id]
        else:
            task_name = task_id_or_name
            assert task_name in self.task_name_id
            task_id = self.task_name_id[task_name]

        task_workspace = self.task_workspaces[task_id]

        # makes sure it does not already exist

        sft_workspace = task_workspace / "0"

        assert not sft_workspace.exists()

        # starts from pretrained

        self.load_pretrained()

        # sft only for the task

        pretrain_data = self.pretrain_data
        dataset = self.dataset(pretrain_data, task_id = task_id)
        
        # Create temporary meta-buffer for SFT
        
        num_episodes = len(dataset.valid_episodes)
        
        values_and_advantages = ReplayBuffer(
            task_workspace.parent / f'.{task_id}_sft_values_and_advantages',
            max_episodes = num_episodes,
            max_timesteps = pretrain_data.max_timesteps,
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
        
        values_and_advantages.meta_data['task_id'][:] = pretrain_data.meta_data['task_id'][dataset.valid_episodes]
        values_and_advantages.episode_lens[:] = pretrain_data.episode_lens[dataset.valid_episodes]
        values_and_advantages.num_episodes = num_episodes

        joined_dataset = JoinedReplayDataset([dataset], values_and_advantages)

        # Initial returns (likely still using pre-trained critic values stored in meta-buffer if we were 1:1, but here we calculate fresh)
        self.calculate_return_or_advantages_(dataset, type = 'returns', mode = 'pretrain', destination_buffer = values_and_advantages)

        self.train_value_network(joined_dataset, task_id = task_id, num_train_steps = num_train_steps_critic, batch_size = batch_size)

        if recalculate_advantages_with_finetuned_critic:
            self.update_buffer_values_(dataset, batch_size = batch_size, task_id = task_id, destination_buffer = values_and_advantages)
            self.calculate_return_or_advantages_(dataset, type = 'advantages', mode = 'finetune', destination_buffer = values_and_advantages)
            self.set_advantage_token_id_(values_and_advantages, mode = 'finetune')

        self.train_policy_network(joined_dataset, advantage_id = self.positive_advantage_id, task_id = task_id, num_train_steps = num_train_steps_actor, batch_size = batch_size)

        self.save(sft_workspace)

    def recap_finetune(
        self,
        task_id_or_name: int | str,
        num_train_steps_actor = 100,
        num_train_steps_critic = 100,
        batch_size = 4
    ):
        assert exists(self.pretrain_data)

        if isinstance(task_id_or_name, int):
            task_id = task_id_or_name
            assert task_id in self.task_id_name
            task_name = self.task_id_name[task_id]
        else:
            task_name = task_id_or_name
            assert task_name in self.task_name_id
            task_id = self.task_name_id[task_name]

        task_workspace = self.task_workspaces[task_id]

        pretrain_data = self.pretrain_data

        all_data_folders = [*task_workspace.glob('*/data.*/')]
        assert len(all_data_folders) > 0, f'no experiences generated yet through rollouts with environment'

        all_rollout_data = [ReplayBuffer.from_folder(data_dir) for data_dir in all_data_folders]

        all_datasets = [
            self.dataset(pretrain_data, task_id = task_id, return_indices = True),
            *[self.dataset(data, return_indices = True) for data in all_rollout_data]
        ]

        num_meta_episodes = sum(len(d.valid_episodes) for d in all_datasets)
        max_timesteps = max([d.replay_buffer.max_timesteps for d in all_datasets])

        # 2. Create/Prepare values_and_advantages buffer
        meta_buffer_path = task_workspace / 'values_and_advantages'
        
        values_and_advantages = ReplayBuffer(
            meta_buffer_path,
            max_episodes = num_meta_episodes,
            max_timesteps = max_timesteps,
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

        values_and_advantages.num_episodes = num_meta_episodes

        # Populate meta data in values_and_advantages
        offset = 0
        for d in all_datasets:
            num_episodes = len(d.valid_episodes)
            values_and_advantages.meta_data['task_id'][offset:offset+num_episodes] = d.replay_buffer.meta_data['task_id'][d.valid_episodes]
            values_and_advantages.episode_lens[offset:offset+num_episodes] = d.replay_buffer.episode_lens[d.valid_episodes]
            offset += num_episodes

        # 3. Step 1: Calculate initial returns based on current critic
        offset = 0
        for d in all_datasets:
            self.update_buffer_values_(d, destination_buffer = values_and_advantages, destination_episode_offset = offset)
            self.calculate_return_or_advantages_(d, type = 'returns', mode = 'finetune', destination_buffer = values_and_advantages, destination_episode_offset = offset)
            offset += len(d.valid_episodes)

        # 4. Step 2: Train critic using JoinedReplayDataset
        joined_dataset = JoinedReplayDataset(all_datasets, values_and_advantages)
        self.train_value_network(joined_dataset, num_train_steps = num_train_steps_critic, batch_size = batch_size)

        # 5. Step 3: Update values and advantages with NEW critic
        offset = 0
        for d in all_datasets:
            self.update_buffer_values_(d, destination_buffer = values_and_advantages, destination_episode_offset = offset)
            self.calculate_return_or_advantages_(d, type = 'advantages', mode = 'finetune', destination_buffer = values_and_advantages, destination_episode_offset = offset)
            offset += len(d.valid_episodes)

        # 6. Step 4: Calculate the percentile cutoff and set advantage token ids
        self.set_advantage_token_id_(values_and_advantages, mode = 'finetune')

        # 7. Step 5: Train actor
        self.train_policy_network(joined_dataset, num_train_steps = num_train_steps_actor, batch_size = batch_size)

        # 8. Save next version
        finetune_ids = [int(folder.name) for folder in task_workspace.glob('*/') if folder.name.isdigit()]
        finetune_id = max(finetune_ids)
        next_recap_iter_id = finetune_id + 1

        self.save(task_workspace / str(next_recap_iter_id))

    def dataset(
        self,
        experiences: ReplayBuffer,
        task_id: int | None = None, 
        fields: list[str] | None = None,
        return_indices = False,
        fieldname_map: dict[str, str] | None = None
    ):
        filter_meta = dict(task_id = task_id) if exists(task_id) else None

        if exists(fields):
            fields = tuple(fields)

        return experiences.dataset(
            timestep_level = True,
            fields = fields,
            filter_meta = filter_meta,
            return_indices = return_indices,
            include_metadata = False,
            fieldname_map = default(fieldname_map, dict(
                text = 'token_ids',
                internal = 'joint_state',
            ))
        )

    def dataloader(
        self,
        experiences: ReplayBuffer | Dataset,
        batch_size = 8,
        task_id: int | None = None,
        fields: list[str] | None = None,
        return_indices = False,
        fieldname_map: dict[str, str] | None = None,
        **dl_kwargs

    ) -> DataLoader:

        if isinstance(experiences, ReplayBuffer):
            filter_meta = dict(task_id = task_id) if exists(task_id) else None

            if exists(fields):
                fields = tuple(fields)

            dataloader = experiences.dataloader(
                batch_size = batch_size,
                timestep_level = True,
                fields = fields,
                filter_meta = filter_meta,
                return_indices = return_indices,
                fieldname_map = fieldname_map,
                **dl_kwargs
            )

            return dataloader

        dataset = experiences

        if return_indices and hasattr(dataset, 'return_indices'):
            dataset.return_indices = return_indices

        assert len(dataset) > 0, 'no experiences to learn from'

        dataloader = DataLoader(dataset, batch_size = batch_size, **dl_kwargs)

        return dataloader

    def train_value_network(
        self,
        experience: ReplayBuffer,
        num_train_steps: int,
        optim_klass = AdamAtan2,
        task_id: int | None = None,
        batch_size = 8,
        lr = 3e-4,
        weight_decay = 1e-2,
        max_grad_norm = 0.5,
        dl_kwargs: dict = dict(),
        optim_kwargs: dict = dict()
    ):

        optim = optim_klass(self.unwrapped_critic.parameters(), lr = lr, weight_decay = weight_decay, **optim_kwargs)

        dataloader = self.dataloader(
            experience,
            batch_size = batch_size,
            task_id = task_id,
            fields = ['images', 'text', 'internal', 'actions_last_step'], # Critic is conditioned on actions_last_step
            fieldname_map = dict(
                text = 'token_ids',
                internal = 'joint_state',
                actions_last_step = 'actions'
            ),
            **dl_kwargs
        )

        model = self.agent.critic
        critic_loss_fn = model.to_critic_value.loss_fn

        dataloader, optim = self.accelerate.prepare(dataloader, optim)

        dl_iter = cycle(dataloader)

        for _ in range(num_train_steps):

            batch_dict = next(dl_iter)

            batch_dict.pop('task_id', None)
            returns = batch_dict.pop('returns')

            pred_value, logits = model(task_id = task_id, **batch_dict)

            cross_entropy_loss = critic_loss_fn(logits, returns, reduction = 'mean')

            self.accelerate.backward(cross_entropy_loss)

            self.accelerate.clip_grad_norm_(model.parameters(), max_grad_norm)

            self.print(f'value loss: {cross_entropy_loss.item():.3f}')

            optim.step()
            optim.zero_grad()

        self.print('value network training complete')

    def train_policy_network(
        self,
        experience: ReplayBuffer | Dataset,
        num_train_steps: int,
        optim_klass = AdamAtan2,
        task_id: int | None = None,
        advantage_id: int | None = None, # for step 2 (SFT stage) - (or the 0th iteration of the finetuning loop) they fix the advantage token to be always positive
        batch_size = 8,
        lr = 3e-4,
        weight_decay = 1e-2,
        max_grad_norm = 0.5,
        dl_kwargs: dict = dict(),
        optim_kwargs: dict = dict()
    ):
        optim = optim_klass(self.unwrapped_actor.parameters(), lr = lr, weight_decay = weight_decay, **optim_kwargs)

        fields = default(dl_kwargs.get('fields'), [
            'images',
            'text',
            'internal',
            'actions',
            'actions_last_step'
        ])

        if exists(advantage_id):
            fields.append('advantage_ids')

        dataloader = self.dataloader(
            experience,
            batch_size = batch_size,
            task_id = task_id,
            fields = fields,
            **dl_kwargs
        )

        model = self.agent.actor

        dataloader, optim = self.accelerate.prepare(dataloader, optim)

        dl_iter = cycle(dataloader)

        for _ in range(num_train_steps):

            batch_dict = next(dl_iter)

            batch_dict.pop('task_id', None)

            if exists(advantage_id):
                batch_dict['advantage_ids'] = advantage_id

            loss, *_ = model(task_id = task_id, **batch_dict)

            self.accelerate.backward(loss)

            self.accelerate.clip_grad_norm_(model.parameters(), max_grad_norm)

            optim.step()
            optim.zero_grad()

        self.print('policy network training complete')

    @beartype
    @torch.no_grad()
    def update_buffer_values_(
        self,
        experiences: ReplayBuffer | Dataset,
        batch_size = 16,
        task_id: int | None = None,
        destination_buffer: ReplayBuffer | None = None,
        destination_episode_offset: int = 0
    ):
        self.agent.eval()
        critic = self.agent.critic
        device = critic.device

        dataloader = self.dataloader(
            experiences,
            batch_size = batch_size,
            task_id = task_id,
            fields = ['images', 'text', 'internal', 'actions_last_step'],
            fieldname_map = dict(
                text = 'token_ids',
                internal = 'joint_state',
                actions_last_step = 'actions'
            ),
            return_indices = True,
            shuffle = False
        )

        dataloader = self.accelerate.prepare(dataloader)

        dest_buffer = default(destination_buffer, experiences if isinstance(experiences, ReplayBuffer) else experiences.replay_buffer)

        for batch in dataloader:
            indices = batch.pop('_indices')
            batch_size = indices.shape[0]

            batch = tree_map_tensor(batch, lambda t: t.to(device))

            batch['times'] = torch.ones((batch_size,), device = device)

            batch.pop('task_id', None)
            values, _ = critic(task_id = task_id, **batch)

            for i, (episode_id, timestep_index) in enumerate(indices):

                if exists(destination_buffer):
                    if hasattr(experiences, 'replay_buffer'):
                        relative_episode_idx = torch.searchsorted(experiences.valid_episodes, episode_id.to(experiences.valid_episodes.device))
                        dest_episode_id = destination_episode_offset + relative_episode_idx
                    else:
                        dest_episode_id = destination_episode_offset + episode_id
                else:
                    dest_episode_id = episode_id

                dest_buffer.data['value'][dest_episode_id.item(), timestep_index.item()] = values[i].item()

        dest_buffer.flush()

    @beartype
    def invalidate_(
        self,
        experiences: ReplayBuffer,
        episode_id: int
    ):
        experiences.store_meta_datapoint(
            episode_id,
            name = 'invalidated',
            datapoint = True
        )

    @beartype
    def invalidate_by_value_threshold_(
        self,
        experiences: ReplayBuffer,
        threshold: float,
        task_id = None,
        value_field = 'value',
        value_buffer: ReplayBuffer | None = None,
        value_episode_offset: int = 0
    ):
        assert 'invalidated' in experiences.data

        read_buffer = default(value_buffer, experiences)
        read_episode_offset = value_episode_offset if exists(value_buffer) else 0

        # we need to map the episodes from the source buffer to the value buffer if an offset is provided
        # for now, assume they are 1:1 if no offset, or offset is applied to all

        values = read_buffer.data[value_field]

        if read_episode_offset > 0:
            num_episodes = experiences.num_episodes
            values = values[read_episode_offset:read_episode_offset + num_episodes]

        should_invalidate = values <= threshold

        if exists(task_id):
            should_invalidate = should_invalidate & experiences.data['task_id'] == task_id

        experiences.data['invalidated'][:] = should_invalidate

    @beartype
    def calculate_return_or_advantages_(
        self,
        experiences: ReplayBuffer | Dataset,
        type: Literal['returns', 'advantages', 'returns_and_advantages'] = 'returns_and_advantages',
        gamma = 1.,
        lam = 0.95,
        mode: Literal['pretrain', 'finetune'] = 'finetune',
        use_accelerated = None,
        destination_buffer: ReplayBuffer | None = None,
        destination_episode_offset: int = 0
    ):

        is_dataset = hasattr(experiences, 'replay_buffer')
        buffer = experiences.replay_buffer if is_dataset else experiences
        dest_buffer = default(destination_buffer, buffer)

        episode_ids = experiences.valid_episodes if is_dataset else range(buffer.num_episodes)

        for i, episode_id in enumerate(episode_ids):

            episode_len = buffer.episode_lens[episode_id].item()

            dest_episode_id = destination_episode_offset + i if exists(destination_buffer) else episode_id


            values = dest_buffer.data['value'][dest_episode_id, :episode_len]
            rewards = buffer.data['reward'][episode_id, :episode_len]
            terminated = buffer.data['terminated'][episode_id, :episode_len]

            # extra insurance once moved to batched

            terminated[episode_len - 1] = True

            # todo - continue to reduce complexity for numpy to torch and back and move to lib

            rewards, values, terminated = map(from_numpy, (rewards, values, terminated))

            # get lookahead for task

            lookahead = -1

            if exists(self.config):
                task_id = buffer.meta_data['task_id'][episode_id]
                task_str = self.task_strings[task_id]
                task_config = self.config.tasks[task_str]
                lookahead = getattr(task_config, mode).advantage_lookahead

            # calculate advantage depending on lookahead

            has_lookahead = lookahead > 0

            advantages, _ = calc_generalized_advantage_estimate(
                rewards = rewards,
                values = values,
                masks = ~terminated,
                gamma = gamma,
                lam = 1. if has_lookahead else lam,
                use_accelerated = use_accelerated
            )

            # if lookahead is greater than 0, then we need to subtract the discounted future advantage

            if has_lookahead:

                lookahead = min(lookahead, advantages.shape[-1])

                gamma_nth_step = gamma ** lookahead

                future_advantages = F.pad(advantages, (0, lookahead), value = 0.)[lookahead:]

                advantages = advantages - gamma_nth_step * future_advantages

            # maybe store advantages

            if type in {'advantages', 'returns_and_advantages'}:
                dest_buffer.data['advantages'][dest_episode_id, :episode_len] = advantages.cpu().numpy()

            # maybe store the returns

            if type in {'returns', 'returns_and_advantages'}:
                dest_buffer.data['returns'][dest_episode_id, :episode_len] = (advantages + values).cpu().numpy()

        dest_buffer.flush()

    @beartype
    def set_advantage_token_id_(
        self,
        experiences: ReplayBuffer | Dataset,
        num_advantages_sample = 10_000,
        mode: Literal['pretrain', 'finetune'] = 'finetune',
        destination_buffer: ReplayBuffer | None = None,
        destination_episode_offset: int = 0
    ):
        is_dataset = hasattr(experiences, 'replay_buffer')
        buffer = experiences.replay_buffer if is_dataset else experiences
        dest_buffer = default(destination_buffer, buffer)

        if is_dataset:
            num_episodes = len(experiences.valid_episodes)
            all_task_ids = from_numpy(buffer.meta_data['task_id'][experiences.valid_episodes])
            episode_lens = from_numpy(buffer.episode_lens[experiences.valid_episodes])
            episode_ids = experiences.valid_episodes
        else:
            num_episodes = buffer.num_episodes
            all_task_ids = from_numpy(buffer.meta_data['task_id'][:num_episodes])
            episode_lens = from_numpy(buffer.episode_lens[:num_episodes])
            episode_ids = arange(num_episodes)

        max_timesteps = buffer.max_timesteps

        for task_id in all_task_ids.unique().tolist():

            task_str = self.task_strings[task_id]
            task_config = self.config.tasks[task_str]

            threshold = 1. - getattr(task_config, mode).positive_data_fraction

            # sample and get the percentile cutoff for whether to set advantage to "positive" label

            task_mask = all_task_ids == task_id

            task_episode_indices = torch.where(task_mask)[0]
            task_episode_ids = episode_ids[task_mask]

            task_episode_ids = rearrange(task_episode_ids, 'e -> e 1')

            timesteps = arange(max_timesteps)

            indices_mask = einx.less('j, i -> i j', timesteps, episode_lens[task_mask])

            timesteps_broadcast = rearrange(timesteps, 't -> 1 t')

            task_episode_ids_broadcast, timesteps_broadcast = torch.broadcast_tensors(task_episode_ids, timesteps_broadcast)

            indices = stack((task_episode_ids_broadcast, timesteps_broadcast), dim = -1)
            indices = indices[indices_mask]

            # determine destination indices
            if exists(destination_buffer):
                dest_task_episode_ids = destination_episode_offset + task_episode_indices
                dest_task_episode_ids = rearrange(dest_task_episode_ids, 'e -> e 1')
                dest_task_episode_ids_broadcast, _ = torch.broadcast_tensors(dest_task_episode_ids, rearrange(timesteps, 't -> 1 t'))
                dest_indices = stack((dest_task_episode_ids_broadcast, timesteps_broadcast), dim = -1)
                dest_indices = dest_indices[indices_mask]
            else:
                dest_indices = indices

            # maybe sample from all advantages
            total_samples = indices.shape[0]

            if total_samples == 0:
                continue

            if total_samples > num_advantages_sample:
                randperm_indices = torch.randperm(total_samples)[:num_advantages_sample]
                sampled_source_indices = indices[randperm_indices]
                sampled_dest_indices = dest_indices[randperm_indices]
            else:
                sampled_source_indices = indices
                sampled_dest_indices = dest_indices

            # if destination buffer is used, advantages might be stored there
            read_buffer = dest_buffer if exists(destination_buffer) else buffer
            read_indices = sampled_dest_indices if exists(destination_buffer) else sampled_source_indices

            advantages = einx.get_at('[e t], b [2] -> b', read_buffer.data['advantages'], read_indices)

            # determine the advantage at designated percentile per task
            advantage_cutoff = torch.quantile(advantages, threshold)

            # calculate labels for all task indices using this cutoff
            all_advantages = einx.get_at('[e t], b [2] -> b', read_buffer.data['advantages'], dest_indices)
            advantage_token_ids = (all_advantages >= advantage_cutoff).int()

            # set it back for all indices
            einx.set_at('[e t], b [2], b', dest_buffer.data['advantage_ids'], dest_indices, advantage_token_ids)

        dest_buffer.flush()

    @beartype
    def set_episode_fail_(
        self,
        experiences: ReplayBuffer,
        episode_id,
        timestep = None
    ):

        if not exists(timestep):
            max_len = experiences.episode_lens[episode_id]
            timestep = int(max_len - 1)
        else:
            experiences.episode_lens[episode_id] = timestep

        reward = experiences.store_datapoint(
            episode_id,
            timestep,
            name = 'reward',
            datapoint = self.task_fail_penalty
        )

        experiences.store_meta_datapoint(
            episode_id,
            name = 'fail',
            datapoint = True
        )

        return experiences

    @beartype
    def set_episode_success_(
        self,
        experiences: ReplayBuffer,
        episode_id,
        timestep = None
    ):

        if not exists(timestep):
            max_len = experiences.episode_lens[episode_id]
            timestep = int(max_len - 1)
        else:
            experiences.episode_lens[episode_id] = timestep

        reward = experiences.store_datapoint(
            episode_id,
            timestep,
            name = 'reward',
            datapoint = tensor(0)
        )

        experiences.store_meta_datapoint(
            episode_id,
            name = 'fail',
            datapoint = False
        )

        return experiences

    @torch.no_grad()
    def gather_experience_from_env(
        self,
        env,
        num_episodes = 1,
        steps = None,
        trajectory_length = 16,
        flow_sampling_steps = 4,
        max_timesteps = 64,
        cond_scale = 1.,
        experience_store_path = None,
        task_id = -1,
        normalize_reward_by_steps = None,
        recap_step = 0, # starts at 0, in which case the logic will be the SFT step, before the proper binary advantage conditioning
        **sampling_kwargs

    ) -> ReplayBuffer:

        has_task = task_id >= 0
        normalize_reward_by_steps = default(normalize_reward_by_steps, has_task)

        assert cond_scale > 0., f'classifier free guidance scaling must be enabled for proposed pi0.6'

        assert exists(steps) ^ has_task, 'either `steps` or `task_id` must be defined, but not both - each task has its own specified max length for normalizing the rewards'

        self.agent.eval()

        actor = self.unwrapped_actor
        critic = self.unwrapped_critic

        # get the max length of each task from the config
        # also determine the last finetuned actor / critic and load

        if has_task:
            task_str = self.task_strings[task_id]
            task_config = self.config.tasks[task_str]

            task_finetune_folder = self.get_last_task_finetune_folder(task_id)

            self.load(task_finetune_folder)

        # default some task specific config
        # (1) the max episode length per task for normalizing the rewards

        if not exists(steps):
            assert has_task
            steps = task_config.max_episode_length

        # (2) the task workspace path, where data is stored with folder names data.0, data.1, etc. for consecutive rollouts

        if has_task:
            assert not exists(experience_store_path)

            past_data = [int(str(file).split('.')[-1]) for file in task_finetune_folder.glob("data.*/")]

            last_rollout_id = max(past_data) if len(past_data) > 0 else -1
            next_rollout_id = last_rollout_id + 1

            experience_store_path = task_finetune_folder / f"data.{next_rollout_id}"
            experience_store_path.mkdir()
        else:
            experience_store_path = './experiences'

        # create the buffer for storing the data

        experience_buffer = ReplayBuffer(
            experience_store_path,
            max_episodes = num_episodes,
            max_timesteps = steps,
            overwrite = True,
            meta_fields = dict(
                task_id     = ('int', (), -1),
                fail        = 'bool',
                invalidated = 'bool',
                recap_step  = ('int', (), -1) # -1 stands for base pretraining dataset
            ),
            fields = dict(
                images      = ('float', (3, env.num_images, *env.image_shape)),
                text        = ('int', (env.max_text_len,)),
                internal    = ('float', (env.joint_dim,)),
                reward      = 'float',
                actions     = ('float', (trajectory_length, actor.dim_action_input)),
                actions_last_step = ('float', (trajectory_length, actor.dim_action_input)),
                terminated  = 'bool',
                invalidated = 'bool'
            )
        )

        # during roll out for gathering experience, they use positive advantage w/ cfg

        highest_advantage_id = actor.num_advantage_tokens - 1

        # mock env

        for _ in range(num_episodes):

            states = env.reset()

            actions_last_step = torch.zeros((trajectory_length, actor.dim_action_input), device = torch.device('cpu'))

            one_episode_kwargs = dict()
            if exists(task_id):
                one_episode_kwargs['task_id'] = tensor(task_id)
            if exists(recap_step):
                one_episode_kwargs['recap_step'] = tensor(recap_step)

            with experience_buffer.one_episode(**one_episode_kwargs):

                for _ in range(steps):

                    sampled_actions, _ = temp_batch_dim(actor)(
                        *states,
                        trajectory_length = trajectory_length,
                        steps = flow_sampling_steps,
                        critic = critic,
                        actions_last_step = actions_last_step,
                        advantage_ids = highest_advantage_id,
                        cond_scale = cond_scale,
                        task_id = task_id,
                        **sampling_kwargs
                    )

                    next_states, reward, truncated, terminated = env.step(sampled_actions)

                    images, text, internal, reward, terminated, actions = to_device([*states, reward, terminated, sampled_actions], torch.device('cpu'))

                    states = next_states

                    if normalize_reward_by_steps:
                        reward /= steps # normalize reward by the max task length defined in the config

                    experience_buffer.store(
                        images = images,
                        text = text,
                        internal = internal,
                        reward = reward,
                        actions = sampled_actions,
                        actions_last_step = actions_last_step
                    )

                    actions_last_step = sampled_actions.cpu()

                    if truncated or terminated:
                        break

        self.accelerate.wait_for_everyone()

        return experience_buffer

# fun

π0 = PiZero
