from __future__ import annotations
from typing import Any

import math
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
from numpy.lib.format import open_memmap

import torch
import torch.nn.functional as F
from torch import pi, nn, arange, cat, stack, tensor, Tensor, broadcast_tensors, is_tensor, full, from_numpy
from torch.nn import Module, ModuleList
from torch.distributions.beta import Beta

from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Dataset

from torchdiffeq import odeint

from scipy.optimize import linear_sum_assignment

from ema_pytorch import EMA

from adam_atan2_pytorch import AdamAtan2

from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, einsum, pack, unpack

from pi_zero_pytorch.tensor_typing import Float, Int, Bool

from hyper_connections import HyperConnections

from hl_gauss_pytorch import HLGaussLayer

from assoc_scan import AssocScan

from evolutionary_policy_optimization import LatentGenePool

from x_evolution import EvoStrategy

import tqdm

from pi_zero_pytorch.replay_buffer import ReplayBuffer, ReplayDataset, JoinedReplayDataset

from accelerate import Accelerator

from pydantic import BaseModel, Field, model_validator

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
    mask: Bool['b n']
):
    # the pi-zero attention is a triangular causal mask, but bidirectional attention for the actions at the very right hand side

    def mask_fn(batch_index, head_index, query_index, key_index):
        key_mask = mask[batch_index, key_index]   # variable length states
        causal_mask = query_index >= key_index    # causal

        bidirectional_action_mask = (             # bidirectional action mask
            key_index >= prefix_causal_length and
            query_index >= prefix_causal_length
        )

        return (key_mask & causal_mask) | bidirectional_action_mask

    return mask_fn

def softclamp_score_mod(value):
    def identity(score, b, h, q, k):
        return score

    def softclamped(score, b, h, q, k):
        score = score / value
        score = torch.tanh(score)
        score = score * value
        return score

    return softclamped if value > 0. else identity

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def sample(prob):
    return random() < prob

def xnor(x, y):
    return not (x ^ y)

def maybe(fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None

        return fn(t, *args, **kwargs)

    return inner

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

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim)

def straight_through(src, tgt):
    return src + (tgt - src).detach()

def softclamp(t, value):
    if value <= 0.:
        return t

    return (t / value).tanh() * value

def append_dims(t, dims):
    shape = t.shape
    ones = ((1,) * dims)
    return t.reshape(*shape, *ones)

def lens_to_mask(lens, max_len):
    seq = torch.arange(max_len, device = lens.device)
    return einx.less('j, i -> i j', seq, lens)

def maybe_and_masks(*masks):
    masks = [*filter(exists, masks)]

    if len(masks) == 0:
        return None
    elif len(masks) == 1:
        return masks[0]

    mask, *rest_mask = masks

    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse

def pack_one_with_inverse(t, pattern):
    packed, inverse = pack_with_inverse([t], pattern)

    def inverse_one(out, inv_pattern = None):
        out, = inverse(out, inv_pattern)
        return out

    return packed, inverse_one

def tree_flatten_with_inverse(input):
    out, tree_spec = tree_flatten(input)

    def inverse(output):
        return tree_unflatten(output, tree_spec)

    return out, inverse

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = l2norm(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

def pad_at_dim(
    t,
    pad: tuple[int, int],
    *,
    dim = -1,
    value = 0.
):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# flow related

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

def noise_assignment(data, noise):
    device = data.device
    data, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (data, noise))
    dist = torch.cdist(data, noise)
    _, assign = linear_sum_assignment(dist.cpu())
    return from_numpy(assign).to(device)

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
        frozen_actions: Float['b nfa d']
    ):
        traj_len = self.trajectory_length
        frozen_len = frozen_actions.shape[1]

        if frozen_len >= traj_len:
            return frozen_actions[:, :traj_len]

        return pad_at_dim(frozen_actions, (0, traj_len - frozen_len), dim = 1)

    def forward(
        self,
        frozen_actions: Float['b nfa d'],
        new_actions: Float['b na d']
    ):
        frozen_actions = self.pad_frozen(frozen_actions)

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
        frozen_actions: Float['b na d'],
        soft_mask: tuple[int, int, int] | Float['na'] | Float['1 na 1'],
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
        noise_actions: Float['b na d'],
        pred_actions: Float['b na d'],
        frozen_actions: Float['b na d'],
        times: Float[''] | Float['b'],
        soft_mask: Float['na'] | Float['1 na 1'],
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
        value: Float['...'],
        temperature = 0.1,
        hard = False
    ) -> Float['... bins']:
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

class Attention(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        softclamp_value = 50.,
        accept_memories = False,
        actions_norm_all = False,
        learned_value_action_residual_mix = False,
        rotary_emb: RotaryEmbedding | None = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.rotary_emb = rotary_emb

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.rmsnorm = nn.RMSNorm(dim)

        # state parameters

        self.to_qkv = LinearNoBias(dim, 3 * dim_inner)
        self.to_out = LinearNoBias(dim_inner, dim)

        # maybe memory parameters

        self.accept_memories = accept_memories

        self.mem_rmsnorm = nn.RMSNorm(dim) if accept_memories else None
        self.to_mem_qkv = LinearNoBias(dim, 3 * dim_inner) if accept_memories else None
        self.to_mem_out = LinearNoBias(dim_inner, dim) if accept_memories else None

        # action parameters

        self.to_actions_qkvg = LinearNoBias(dim, 4 * dim_inner)

        self.to_action_value_residual_mix = nn.Sequential(
            LinearNoBias(dim, heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        ) if learned_value_action_residual_mix else (lambda _: 0.5)

        self.to_actions_out = LinearNoBias(dim_inner, dim)

        # norms for all action linears
        # from Bytedance's GR-3

        self.actions_norm_all = actions_norm_all

        if actions_norm_all:
            self.actions_q_norm = nn.RMSNorm(dim_head)
            self.actions_k_norm = nn.RMSNorm(dim_head)
            self.actions_v_norm = nn.RMSNorm(dim_head)
            self.actions_out_norm = nn.RMSNorm(dim, elementwise_affine = False)

        self.softclamp_value = softclamp_value

    def forward_actions_with_cached_state(
        self,
        actions,
        cached_state_keys_values: tuple[Tensor, Tensor],
        memories: tuple[Tensor, Tensor] | None = None,
        rotary_emb = None,
        mask: Bool['b n'] | None = None,
        actions_value_residual: Tensor | None = None,
        return_keys_values = False,
        flex_attn_fn: Callable | None = None,
        knowledge_insulate = False
    ):
        aq, ak, av, ag = self.to_actions_qkvg(actions).chunk(4, dim = -1)

        aq, ak, av, ag = tuple(self.split_heads(t) for t in (aq, ak, av, ag))

        if self.actions_norm_all:
            aq, ak, av = tuple(norm(t) for norm, t in zip((self.actions_q_norm, self.actions_k_norm, self.actions_v_norm), (aq, ak, av)))

        if exists(actions_value_residual):
            mix = self.to_action_value_residual_mix(actions)
            av = av * mix + actions_value_residual * (1. - mix)

        q = aq
        mk, mv = cached_state_keys_values

        # able to stop gradients from actions to state - (knowledge insulation blogpost https://www.physicalintelligence.company/research/knowledge_insulation)

        if knowledge_insulate:
            mk, mv = tuple(t.detach() for t in (mk, mv))

        # concat cache key / values with action key / values

        k, v = tuple(cat(tensors, dim = -2) for tensors in zip((mk, mv), (ak, av)))

        # handle read, write memories

        assert not (self.accept_memories ^ exists(memories))

        if exists(memories):
            _, write_memories = memories
            write_memories = self.mem_rmsnorm(write_memories)
            # mqkv_write = self.to_mem_qkv(write_memories)

        if exists(rotary_emb):
            q = apply_rotary_emb(rotary_emb, q, freqs_seq_dim = -2)
            k = apply_rotary_emb(rotary_emb, k)

        elif exists(self.rotary_emb):
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # attention

        if exists(flex_attn_fn):
            out = flex_attn_fn(q, k, v)
        else:
            q = q * self.scale

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = softclamp(sim, self.softclamp_value)

            if exists(mask):
                sim = einx.where('b j, b h i j, -> b h i j', mask, sim, max_neg_value(sim))

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # gate

        out = out * ag.sigmoid()

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
        state: Float['b n d'],
        rotary_emb = None
    ) -> Float['b n d']:

        device = state.device

        q, k, v = self.to_qkv(state).chunk(3, dim = -1)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_emb(rotary_emb, q)
            k = apply_rotary_emb(rotary_emb, k)

        elif exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # attention

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        sim = softclamp(sim, self.softclamp_value)

        causal_mask = torch.ones(sim.shape[-2:], dtype = torch.bool, device = device).triu(1)

        sim = sim.masked_fill(causal_mask, max_neg_value(sim))

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge attention heads

        out = self.merge_heads(out)

        return self.to_out(out)

    def forward(
        self,
        multimodal_seq,
        actions,
        rotary_emb = None,
        memories: tuple[Tensor, Tensor] | None = None,
        mask: Bool['b n'] | None = None,
        actions_value_residual: Tensor | None = None,
        return_keys_values = False,
        flex_attn_fn: Callable | None = None,
        knowledge_insulate = False
    ):
        seq_len, device = multimodal_seq.shape[-2], multimodal_seq.device

        multimodal_seq = self.rmsnorm(multimodal_seq)

        # separate projections for multimodal seq vs actions

        mq, mk, mv = self.to_qkv(multimodal_seq).chunk(3, dim = -1)

        aq, ak, av, ag = self.to_actions_qkvg(actions).chunk(4, dim = -1)

        mq, mk, mv, aq, ak, av, ag = tuple(self.split_heads(t) for t in (mq, mk, mv, aq, ak, av, ag))

        if self.actions_norm_all:
            aq, ak, av = tuple(norm(t) for norm, t in zip((self.actions_q_norm, self.actions_k_norm, self.actions_v_norm), (aq, ak, av)))

        # able to stop gradients from actions to state - (knowledge insulation blogpost https://www.physicalintelligence.company/research/knowledge_insulation)

        if knowledge_insulate:
            mk, mv = tuple(t.detach() for t in (mk, mv))

        # value residual

        if exists(actions_value_residual):
            mix = self.to_action_value_residual_mix(actions)
            av = av * mix + actions_value_residual * (1. - mix)

        q, k, v = tuple(cat(tensors, dim = -2) for tensors in zip((mq, mk, mv), (aq, ak, av)))

        # handle read, write memories

        has_memories = exists(memories) and any([m.numel() > 0 for m in memories])

        assert not (self.accept_memories ^ has_memories)

        if has_memories:
            memories, unpack_memories = pack_with_inverse(memories, 'b * d')
            memories = self.mem_rmsnorm(memories)
            mqkv = self.to_mem_qkv(memories)
            mqkv_read, mqkv_write = unpack_memories(mqkv, 'b * d')

            mqr, mkr, mvr, mqw, mkw, mvw = tuple(self.split_heads(t) for t in (*mqkv_read.chunk(3, dim = -1), *mqkv_write.chunk(3, dim = -1)))

            k = cat((mkr, k, mkw), dim = -2)
            v = cat((mvr, v, mvw), dim = -2)
            q, attn_output_unpack_memories = pack_with_inverse((mqr, q, mqw), 'b h * d')

        # rotary embedding

        if exists(rotary_emb):
            q = apply_rotary_emb(rotary_emb, q)
            k = apply_rotary_emb(rotary_emb, k)
        elif exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        if exists(flex_attn_fn):
            out = flex_attn_fn(q, k, v)

        else:
            # attention

            q = q * self.scale

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = softclamp(sim, self.softclamp_value)

            causal_mask = torch.ones(sim.shape[-2:], dtype = torch.bool, device = device).triu(1)

            if exists(mask):
                causal_mask = einx.logical_or('b j, i j -> b 1 i j', ~mask, causal_mask)

            causal_mask[..., seq_len:, seq_len:] = False  # actions have bidirectional attention, lining up with Transfusion paper

            sim = sim.masked_fill(causal_mask, max_neg_value(sim))

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # gating of values, used in alphafold line of work

        gates = pad_at_dim(ag.sigmoid(), (out.shape[-2] - ag.shape[-2], 0), value = 1., dim = -2)

        out = out * gates

        # split out memories

        if self.accept_memories:
            mem_read_out, out, mem_write_out = attn_output_unpack_memories(out)

        # merge attention heads

        out = self.merge_heads(out)

        # separate projections for multimodal seq vs actions

        mout, aout = out[:, :seq_len], out[:, seq_len:]

        mout, aout = self.to_out(mout), self.to_actions_out(aout)

        if self.actions_norm_all:
            aout = self.actions_out_norm(aout)

        output = (mout, aout)

        if self.accept_memories:
            mem_out, unpack_memories = pack_with_inverse((mem_read_out, mem_write_out), 'b h * d')
            mem_out = self.merge_heads(mem_out)
            mem_out = self.to_mem_out(mem_out)

            output = (*output, unpack_memories(mem_out, 'b * d'))

        if not return_keys_values:
            return output

        return output, (mk, mv, ak, av)

# attention

class SwiGLUFeedForward(Module):
    def __init__(
        self,
        dim,
        expand_factor = 4.,
        dim_inner = None,
        rmsnorm = True,
        norm_all = False
    ):
        super().__init__()
        dim_inner = default(dim_inner, int(dim * expand_factor * 2 / 3))

        self.rmsnorm = nn.RMSNorm(dim) if rmsnorm else nn.Identity()
        self.proj_in = LinearNoBias(dim, dim_inner * 2)
        self.proj_out = LinearNoBias(dim_inner, dim)

        # maybe additional norms for action branch

        self.post_proj_in_norm = nn.RMSNorm(dim_inner) if norm_all else nn.Identity()
        self.post_proj_out_norm = nn.RMSNorm(dim, elementwise_affine = False) if norm_all else nn.Identity()

    def forward(
        self,
        seq
    ):
        seq = self.rmsnorm(seq)
        seq, gates = self.proj_in(seq).chunk(2, dim = -1)

        seq = seq * F.gelu(gates)
        seq = self.post_proj_in_norm(seq)

        out = self.proj_out(seq)
        return self.post_proj_out_norm(out)

# actions need time conditioning
# ada-ln zero from DiT - here we will improvise with adaptive rmsnorm

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        times = rearrange(times, '... -> ... 1')
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine = False)

        self.to_gamma = nn.Sequential(
            nn.Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = LinearNoBias(dim_cond, dim)

    def forward(self, actions, cond):

        if cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        normed = self.norm(actions)
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)

        return normed * gamma + beta

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
        dim_time_cond = None,
        depth = 12,
        dim_head = 64,
        heads = 8,
        use_flex_attn = False,
        ff_expand_factor = 4.,
        attn_softclamp_value = 50.,
        final_norm_softclamp_value = 30.,
        vit: Module | None = None,
        vit_dim = None,
        external_state_encoders: Module | list[Module] | None = None,
        dim_internal_state: int | None = None,
        num_action_register_tokens = 4,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        lm_pad_id = -1,
        lm_loss_weight = 1.,
        model_predict_output: Literal['flow', 'clean'] = 'flow', # dreamer4 as well as https://arxiv.org/abs/2511.13720 - something is going around, make sure it is not missed.
        max_timesteps = 16,
        flow_loss_weight = 1.,
        immiscible_flow = False,             # https://arxiv.org/abs/2406.12303
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
    ):
        super().__init__()
        dim_time_cond = default(dim_time_cond, dim * 2)

        self.dim = dim

        # flex attention related

        assert not (use_flex_attn and not exists(flex_attention)), 'flex attention cannot be used'
        self.use_flex_attn = use_flex_attn
        self.attn_softclamp_value = attn_softclamp_value

        # vit

        self.vit = vit

        self.maybe_to_image_tokens = nn.Linear(vit_dim, dim) if exists(vit_dim) and vit_dim != dim else nn.Identity()

        # embedding

        self.token_emb = nn.Embedding(num_tokens, dim)

        # internal states

        self.to_joint_state_tokens = nn.Linear(dim_joint_state, dim)

        self.dim_internal_state = default(dim_internal_state, dim)
        self.to_internal_state_tokens = nn.Linear(dim_internal_state, dim) if exists(dim_internal_state) else nn.Identity()

        # additional external states

        external_state_encoders = default(external_state_encoders, [])
        self.external_state_encoders = ModuleList(external_state_encoders)

        # actions

        self.dim_action_input = dim_action_input

        self.action_register_tokens = nn.Parameter(torch.zeros(num_action_register_tokens, dim))
        nn.init.normal_(self.action_register_tokens, std = 0.02)

        self.to_action_tokens = nn.Linear(dim_action_input, dim)

        # time conditioning

        self.to_time_cond = nn.Sequential(
            RandomFourierEmbed(dim),
            nn.Linear(dim, dim_time_cond),
            nn.SiLU(),
        )

        # latent variable / gene conditioning

        can_accept_latent = exists(dim_latent)
        self.can_accept_latent = can_accept_latent

        if can_accept_latent:
            self.to_latent_cond = nn.Sequential(
                nn.Linear(dim_latent, dim_time_cond * 2),
                nn.SiLU(),
                nn.Linear(dim_time_cond * 2, dim_time_cond),
            )

            nn.init.zeros_(self.to_latent_cond[-1].weight)
            nn.init.zeros_(self.to_latent_cond[-1].bias)

        # positional embedding

        self.rotary_emb = RotaryEmbedding(dim_head)

        # recurrent memory parameters and logic

        self.has_recurrent_memories = num_recurrent_memory_tokens > 0

        self.memory_tokens = nn.Parameter(torch.zeros(num_recurrent_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std = 0.02)

        self.final_norm_write_memories = nn.RMSNorm(dim) if self.has_recurrent_memories else None

        # residual functions, with maybe hyper connections

        assert num_residual_streams >= 1
        init_residual_fn, self.maybe_expand_residuals, self.maybe_reduce_residuals = HyperConnections.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        residual_fns = []
        counter = count()

        # attention and feedforward

        layers = []
        cond_layers = []

        for i in range(depth):
            is_first_block = i == 0

            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, actions_norm_all = action_dit_norm_all_linears, accept_memories = self.has_recurrent_memories, learned_value_action_residual_mix = not is_first_block, **attn_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, **ff_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, rmsnorm = False, norm_all = action_dit_norm_all_linears, **ff_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, **ff_kwargs) if self.has_recurrent_memories else None
            ]))

            residual_fns.append(ModuleList([
                init_residual_fn(dim = dim, layer_index = next(counter)),
                init_residual_fn(dim = dim, layer_index = next(counter)),
            ]))

            cond_layers.append(ModuleList([
                AdaptiveRMSNorm(dim, dim_time_cond),
                AdaptiveLayerscale(dim, dim_time_cond),
                AdaptiveRMSNorm(dim, dim_time_cond),
                AdaptiveLayerscale(dim, dim_time_cond)
            ]))

        self.layers = ModuleList(layers)
        self.cond_layers = ModuleList(cond_layers)

        self.residual_layers = ModuleList(residual_fns)

        self.final_norm_softclamp = partial(softclamp, value = final_norm_softclamp_value)

        self.final_norm = nn.RMSNorm(dim)
        self.final_actions_norm = nn.RMSNorm(dim)

        # unembedding

        self.state_to_logits = LinearNoBias(dim, num_tokens)

        # explicit task conditioning without prompt

        self.num_tasks = num_tasks
        self.has_task_cond = exists(num_tasks) and num_tasks > 0

        self.task_emb = nn.Embedding(num_tasks, dim) if self.has_task_cond else None

        # to task status prediction

        self.to_task_status = LinearNoBias(dim, num_task_status)
        self.task_status_is_invalid = task_status_is_invalid
        self.task_status_loss_weight = task_status_loss_weight

        # actor related

        self.actions_to_pred_flow = None
        self.loss_fn = None

        if not is_critic:
            self.actions_to_pred_flow = LinearNoBias(dim, dim_action_input)
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

        self.immiscible_flow = immiscible_flow

        # reward classifier free guidance

        self.reward_tokens_dropout_prob = reward_tokens_dropout_prob

        # advantage tokens and cfg related

        self.num_advantage_tokens = num_advantage_tokens
        self.can_advantage_token_cond = num_advantage_tokens > 0

        if self.can_advantage_token_cond:
            self.advantage_embed = nn.Embedding(num_advantage_tokens, dim)

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

        # loss related

        self.lm_loss_weight = lm_loss_weight
        self.flow_loss_weight = flow_loss_weight

        # sampling related

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # tensor typing

        self._nm = num_recurrent_memory_tokens

    def action_params_for_evolution(self):
        action_params = set()

        add_module = lambda m: action_params.update(set(m.parameters()))

        add_module(self.to_action_tokens)

        for (
            (attn, state_ff, actions_ff, memories_ff),
            (attn_ada_rmsnorm, attn_ada_layerscale, ff_ada_rmsnorm, ff_ada_layerscale),
            (attn_residual, actions_ff_residual),
        ) in zip(self.layers, self.cond_layers, self.residual_layers):

            add_module(attn.to_actions_out)
            add_module(attn.to_actions_qkvg)
            add_module(actions_ff)

        add_module(self.actions_to_pred_flow)
        return action_params

    @property
    def can_cfg(self):
        return self.reward_tokens_dropout_prob > 0.

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def load_pretrained_vlm_weights_(
        self,
        weights: dict[str, Tensor]
    ):
        raise NotImplementedError

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
        latents: Float['d'] | Float['b d'] = None,
        reward_tokens: Float['b d'] | None = None,
        advantage_ids: Int['b'] | int | None = None,
        internal_state_tokens: Float['b ns d'] | None = None,
        frozen_actions: Float['b nfa da'] | None = None,
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
    ):
        assert not self.is_critic

        batch_size = token_ids.shape[0]

        was_training = self.training
        self.eval()

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

                padded_timestep = append_dims(timestep, flow.ndim - 1)

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

        values, _ = critic(*input_args, **input_kwargs)

        return out, values

    @torch.no_grad()
    def forward_with_reward_cfg(
        self,
        *args,
        reward_tokens: Float['b d'] | None = None,
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
        images: Float['b nv d'] | Float['b c h w'] | Float['b c f h w'], # vision
        token_ids: Int['b nt'],                                          # language
    ) -> Float['b n d']:

        device = token_ids.device

        language_tokens = self.token_emb(token_ids)

        # vision

        if exists(self.vit):
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
            assert images.ndim == 3, 'images must be already encoded as (batch, seq, feature dimension)'
            visual_tokens = images

        visual_tokens = self.maybe_to_image_tokens(visual_tokens)

        # concat visual rep with language

        state_tokens, _ = pack_with_inverse([
            visual_tokens,
            language_tokens,
        ], 'b * d')

        # rotary embeddings

        seq_len = state_tokens.shape[-2]

        seq = torch.arange(seq_len, device = device)

        rotary_emb = self.rotary_emb(seq)

        # transformer

        for attn, ff, _, _ in self.layers:

            state_attn_out = attn.forward_only_vision_language(state_tokens, rotary_emb = rotary_emb)

            state_tokens = state_tokens + state_attn_out

            state_tokens = ff(state_tokens) + state_tokens

        embed = self.final_norm_softclamp(state_tokens)

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
        advantages: Float['b t'],
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
        old_values: Float['b'],
        advantages: Float['b'],
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
        images: Float['b nv d'] | Float['b c h w'] | Float['b c f h w'], # vision
        token_ids: Int['b nt'],                                          # language
        joint_state: Float['b djs'],                                     # joint state
        actions: Float['b na da'] | None = None,                         # action
        times: Float['b'] = None,
        noise: Float['b na da'] | None = None,
        latents: Float['d'] | Float['b d'] = None,
        reward_tokens: Float['b d'] | None = None,
        internal_state_tokens: Float['b ns d'] | None = None,
        external_states: tuple[Float['b ...']] | None = None,
        record_and_return_memory_tokens = False,
        past_recurrent_memory_tokens: Float['b {self._nm} d'] | None = None,
        task_id: Int['b'] | int | None = None,
        task_status: Int['b'] | None = None,
        advantage_ids: Int['b'] | int | None = None,
        return_actions_flow = False,
        return_state_keys_values = False,
        cached_state_keys_values: list[tuple[Tensor, Tensor]] | None = None,
        return_language_loss = True,
        return_action_flow_loss = True,
        knowledge_insulate = False,
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

            if self.immiscible_flow:
                assignment = noise_assignment(actions, noise)
                noise = noise[assignment]

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

        time_cond = self.to_time_cond(times)
        action_tokens = self.to_action_tokens(actions)

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

        joint_state_tokens = self.to_joint_state_tokens(joint_state)

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

        if time_cond.ndim == 3:
            orig_action_len = orig_actions.shape[-2]
            time_cond = pad_at_dim(time_cond, (action_with_registers_length - orig_action_len, 0), dim = -2)

        state_tokens = None

        if not inferencing:
            # language

            labels = token_ids[:, 1:]

            language_tokens = self.token_emb(token_ids)

            # vision

            if exists(self.vit):
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
                assert images.ndim == 3, 'images must be already encoded as (batch, seq, feature dimension)'
                visual_tokens = images

            visual_tokens = self.maybe_to_image_tokens(visual_tokens)

            # empty

            empty_token = visual_tokens.new_empty((batch, 0, self.dim))

            # maybe reward tokens

            if not exists(reward_tokens):
                reward_tokens = empty_token

            # maybe advantage tokens

            if exists(advantage_ids):
                assert self.can_advantage_token_cond

                if isinstance(advantage_ids, int):
                    advantage_ids = full((batch,), advantage_ids, device = self.device)

                advantage_tokens = self.advantage_embed(advantage_ids)
            else:
                advantage_tokens = empty_token

            # maybe dropout reward or advantage tokens for cfg

            if self.training and sample(self.reward_tokens_dropout_prob):
                reward_tokens = empty_token

            if self.training and sample(self.advantage_tokens_dropout_prob):
                advantage_tokens = empty_token

            # maybe task token

            task_token = empty_token

            if self.has_task_cond and exists(task_id):
                if isinstance(task_id, int):
                    task_id = full((batch,), task_id, device = self.device)

                task_token = self.task_emb(task_id)

            # additional external states

            if exists(external_states):
                external_state_tokens = [encode(external_state) for encode, external_state in zip(self.external_state_encoders, external_states)]
                external_state_tokens = pack(external_state_tokens, 'b * d')

            else:
                external_state_tokens = visual_tokens.new_empty((batch, 0, self.dim))

            # concat visual rep with language

            state_tokens, inverse_packed_states = pack_with_inverse([
                external_state_tokens,
                visual_tokens,
                language_tokens,
                reward_tokens,
                advantage_tokens,
                task_token
            ], 'b * d')

        # take care of masking for variable lengthed states, starting with the language tokens

        # which then leads to proper rotary embeddings

        command_length = token_ids.shape[-1]

        language_mask = token_ids != self.lm_pad_id

        if inferencing:
            state_length = cached_state_keys_values[0][0].shape[-2]
        else:
            state_length = state_tokens.shape[-2]

        mask = F.pad(language_mask, (state_length - command_length, action_with_registers_length), value = True) # assume fixed number of images for now, but address variable length modality states later

        # memory

        mask = F.pad(mask, (past_recurrent_memory_tokens.shape[-2], write_memory_tokens.shape[-2]), value = True)

        # rotary embeddings

        seq = mask.float().cumsum(dim = -1)
        rotary_emb = self.rotary_emb(seq)

        rotary_emb = rearrange(rotary_emb, 'b n d -> b 1 n d')

        # prepare maybe flex attention

        flex_attn_fn = None

        if not inferencing and self.use_flex_attn and state_tokens.is_cuda:

            prefix_length = state_tokens.shape[-2]
            seq_len = prefix_length + action_tokens.shape[-2]

            block_mask = create_block_mask(
                create_pizero_attn_mask(
                    prefix_length,
                    mask = mask,
                ),
                Q_LEN = seq_len,
                KV_LEN = seq_len,
                device = state_tokens.device,
                _compile = True,
            )

            score_mod_fn = softclamp_score_mod(self.attn_softclamp_value)

            flex_attn_fn = partial(
                flex_attention,
                block_mask = block_mask,
                score_mod = score_mod_fn
            )

        # state keys and values for caching during inference

        cached_state_key_values_iter = iter(default(cached_state_keys_values, []))

        # value residual learning

        actions_value_residual = None

        # maybe expand residual streams

        action_tokens = self.maybe_expand_residuals(action_tokens)

        # transformer

        if not inferencing:

            next_state_cached_keys_values = []

            for (
                (attn, state_ff, actions_ff, memories_ff),
                (attn_ada_rmsnorm, attn_ada_layerscale, ff_ada_rmsnorm, ff_ada_layerscale),
                (attn_residual, actions_ff_residual),
            ) in zip(self.layers, self.cond_layers, self.residual_layers):

                # joint attention

                action_tokens, add_action_residual = attn_residual(action_tokens)

                action_tokens = attn_ada_rmsnorm(action_tokens, time_cond)

                (state_attn_out, actions_attn_out, *maybe_mem_out), (state_keys, state_values, action_keys, action_values) = attn(
                    state_tokens,
                    action_tokens,
                    rotary_emb = rotary_emb,
                    flex_attn_fn = flex_attn_fn,
                    actions_value_residual = actions_value_residual,
                    mask = mask,
                    return_keys_values = True,
                    knowledge_insulate = knowledge_insulate,
                    memories = memory_tokens
                )

                next_state_cached_keys_values.append((state_keys, state_values))

                actions_value_residual = default(actions_value_residual, action_values)

                action_attn_out = attn_ada_layerscale(actions_attn_out, time_cond)

                state_tokens = state_tokens + state_attn_out
                action_tokens = add_action_residual(action_attn_out)

                if self.has_recurrent_memories:
                    (read_mem_attn_out, write_mem_attn_out), = maybe_mem_out
                    read_mem, write_mem = memory_tokens

                    memory_tokens = (read_mem + read_mem_attn_out, write_mem + write_mem_attn_out)

                # state feedforward

                state_tokens_out = state_ff(state_tokens)

                state_tokens = state_tokens + state_tokens_out

                # action feedforward

                action_tokens, add_action_ff_residual = actions_ff_residual(action_tokens)

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

                action_tokens_out = actions_ff(action_tokens)

                action_tokens_out = ff_ada_layerscale(action_tokens_out, time_cond)

                action_tokens = add_action_ff_residual(action_tokens_out)

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
                (attn_ada_rmsnorm, attn_ada_layerscale, ff_ada_rmsnorm, ff_ada_layerscale),
                (attn_residual, actions_ff_residual),
            ) in zip(self.layers, self.cond_layers, self.residual_layers):

                # actions attention

                action_tokens, add_action_residual = attn_residual(action_tokens)

                action_tokens = attn_ada_rmsnorm(action_tokens, time_cond)

                actions_attn_out, (state_keys, state_values, action_keys, action_values) = attn.forward_actions_with_cached_state(
                    action_tokens,
                    cached_state_keys_values = next(cached_state_key_values_iter),
                    rotary_emb = rotary_emb,
                    mask = mask,
                    return_keys_values = True
                )

                actions_value_residual = default(actions_value_residual, action_values)

                actions_attn_out = attn_ada_layerscale(actions_attn_out, time_cond)
                action_tokens = add_action_residual(actions_attn_out)

                # actions feed forward

                action_tokens, add_action_ff_residual = actions_ff_residual(action_tokens)

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

                action_out = actions_ff(action_tokens)

                action_out = ff_ada_layerscale(action_out, time_cond)

                action_tokens = add_action_residual(action_out)

                # maybe memory feed forward

                if self.has_recurrent_memories:
                    memory_tokens, unpack_memory = pack_with_inverse(memory_tokens, 'b * d')

                    memory_tokens = memories_ff(memory_tokens) + memory_tokens

                    memory_tokens = unpack_memory(memory_tokens)

        # maybe reduce residual streams

        action_tokens = self.maybe_reduce_residuals(action_tokens)

        if not inferencing:
            # unpack and unembed to predictions

            _, visual_tokens, tokens, *_ = inverse_packed_states(state_tokens, 'b * d')

            # gemma uses a final softclamp before norm

            tokens = self.final_norm_softclamp(tokens)

        *_, action_tokens = inverse_pack_action_registers(action_tokens)

        action_tokens = self.final_norm_softclamp(action_tokens)

        # memories

        read_memories, written_memory_tokens = memory_tokens

        # writeable memories norm

        if self.has_recurrent_memories:
            written_memory_tokens = self.final_norm_write_memories(written_memory_tokens)

        # final actions norm

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

            mask = maybe_and_masks(is_not_invalid_mask, action_prefix_mask)

            # mask out

            if exists(mask):
                flow_loss = flow_loss[mask]

            # average

            flow_loss = flow_loss.mean()

        # language cross entropy loss

        language_loss = self.zero

        if return_language_loss:
            tokens = self.final_norm(tokens)

            language_logits = self.state_to_logits(tokens)

            language_loss = F.cross_entropy(
                rearrange(language_logits[:, :-1], 'b n l -> b l n'),
                labels.long(),
                ignore_index = self.lm_pad_id
            )

        # loss breakdown

        loss_breakdown = (language_loss, flow_loss)

        # total loss and return breakdown

        total_loss = (
            language_loss * self.lm_loss_weight +
            flow_loss * self.flow_loss_weight
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

        # evolutionary policy optimization related
        # Wang et al. https://web3.arxiv.org/abs/2503.19037

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
        accelerate_kwargs: dict = dict()
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
        **sampling_kwargs
    ):
        self.agent.eval()

        actor = self.unwrapped_actor
        critic = self.unwrapped_critic

        all_episode_memories = []

        for _ in range(num_episodes):

            states = env.reset()

            memories = []

            for _ in range(steps):

                sampled_actions, values = temp_batch_dim(actor)(
                    *states,
                    trajectory_length = trajectory_length,
                    steps = flow_sampling_steps,
                    critic = critic,
                    **sampling_kwargs
                )

                next_states, reward, truncated, terminated = env.step(sampled_actions)

                memories.append(to_device([*states, reward, terminated, sampled_actions, values], torch.device('cpu')))

                states = next_states

            stacked_memories = tuple(map(stack, zip(*memories)))

            all_episode_memories.append(stacked_memories)

        self.accelerate.wait_for_everyone()

        return all_episode_memories

    def learn_agent(
        self,
        memories,
        fitnesses = None,
        epochs = 2,
        batch_size = 16
    ):
        actor_fpo_loss_fn = self.agent.actor_fpo_loss_fn

        self.agent.train()

        (
            images,
            commands,
            joint_state,
            rewards,
            terminated,
            actions,
            old_values
        ) = map(stack, zip(*memories))

        flow_timesteps = actions.shape[1]

        # actions go out into the environment, rewards are received, generalized advantage calculated with critic values

        boundaries = terminated

        advantages, returns = calc_generalized_advantage_estimate(rewards, old_values, boundaries, use_accelerated = False)

        data_tensors = (
            images,
            commands,
            joint_state,
            rewards,
            terminated,
            actions,
            old_values,
            advantages
        )

        data_tensors = tuple(rearrange(t, 'b t ... -> (b t) ...') for t in data_tensors)

        # dataset and dataloader

        dataset = TensorDataset(*data_tensors)

        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

        # copy of old actor

        old_actor = deepcopy(self.unwrapped_actor)

        # training loop

        for _ in range(epochs):
            for (
                images,
                commands,
                joint_state,
                rewards,
                terminated,
                actions,
                old_values,
                advantages
            ) in dataloader:

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
                    actions,
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
        values_and_advantages.meta_data['episode_lens'][:] = pretrain_data.meta_data['episode_lens'][:pretrain_data.num_episodes]
        values_and_advantages.num_episodes = pretrain_data.num_episodes

        dataset = self.dataset(pretrain_data)
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
        values_and_advantages.meta_data['episode_lens'][:] = pretrain_data.meta_data['episode_lens'][dataset.valid_episodes]
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

        all_rollout_data = [ReplayBuffer.from_config(data_dir) for data_dir in all_data_folders]

        all_datasets = [
            self.dataset(pretrain_data, task_id = task_id),
            *[self.dataset(data) for data in all_rollout_data]
        ]

        num_meta_episodes = sum(len(d.valid_episodes) for d in all_datasets)
        max_timesteps = max([d.experiences.max_timesteps for d in all_datasets])

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
            values_and_advantages.meta_data['task_id'][offset:offset+num_episodes] = d.experiences.meta_data['task_id'][d.valid_episodes]
            values_and_advantages.meta_data['episode_lens'][offset:offset+num_episodes] = d.experiences.meta_data['episode_lens'][d.valid_episodes]
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
        return_indices = False
    ):
        return ReplayDataset(
            experiences,
            fields = default(fields, [
                'images',
                'text',
                'internal',
                'actions'
            ]),
            task_id = task_id,
            return_indices = return_indices,
            fieldname_map = dict(
                text = 'token_ids',
                internal = 'joint_state',
            )
        )

    def dataloader(
        self,
        experiences: ReplayBuffer | Dataset,
        batch_size = 8,
        task_id: int | None = None,
        fields: list[str] | None = None,
        return_indices = False,
        **dl_kwargs

    ) -> DataLoader:

        if not isinstance(experiences, Dataset):
            dataset = self.dataset(experiences, task_id = task_id, fields = fields, return_indices = return_indices)
        else:
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
            **dl_kwargs
        )

        model = self.agent.critic
        critic_loss_fn = model.to_critic_value.loss_fn

        dataloader, optim = self.accelerate.prepare(dataloader, optim)

        dl_iter = cycle(dataloader)

        for _ in range(num_train_steps):

            batch_dict = next(dl_iter)

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

        fields = [
            'images',
            'text',
            'internal',
            'actions',
        ]

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
        experiences: ReplayBuffer | ReplayDataset,
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
            fields = ['images', 'text', 'internal', 'actions'],
            return_indices = True,
            shuffle = False
        )

        dataloader = self.accelerate.prepare(dataloader)

        dest_buffer = default(destination_buffer, experiences if isinstance(experiences, ReplayBuffer) else experiences.experiences)

        for batch in dataloader:
            indices = batch.pop('indices')
            batch_size = indices.shape[0]

            batch = tree_map_tensor(batch, lambda t: t.to(device))

            batch['times'] = torch.ones((batch_size,), device = device)

            values, _ = critic(task_id = task_id, **batch)

            for i, (episode_id, timestep_index) in enumerate(indices):

                if exists(destination_buffer):
                    if isinstance(experiences, ReplayDataset):
                        relative_episode_idx = torch.searchsorted(experiences.valid_episodes, episode_id)
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
        experiences: ReplayBuffer | ReplayDataset,
        type: Literal['returns', 'advantages', 'returns_and_advantages'] = 'returns_and_advantages',
        gamma = 1.,
        lam = 0.95,
        mode: Literal['pretrain', 'finetune'] = 'finetune',
        use_accelerated = None,
        destination_buffer: ReplayBuffer | None = None,
        destination_episode_offset: int = 0
    ):

        buffer = experiences.experiences if isinstance(experiences, ReplayDataset) else experiences
        dest_buffer = default(destination_buffer, buffer)

        episode_ids = experiences.valid_episodes if isinstance(experiences, ReplayDataset) else range(buffer.num_episodes)

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
        experiences: ReplayBuffer | ReplayDataset,
        num_advantages_sample = 10_000,
        mode: Literal['pretrain', 'finetune'] = 'finetune',
        destination_buffer: ReplayBuffer | None = None,
        destination_episode_offset: int = 0
    ):
        buffer = experiences.experiences if isinstance(experiences, ReplayDataset) else experiences
        dest_buffer = default(destination_buffer, buffer)

        if isinstance(experiences, ReplayDataset):
            num_episodes = len(experiences.valid_episodes)
            all_task_ids = from_numpy(buffer.meta_data['task_id'][experiences.valid_episodes])
            episode_lens = from_numpy(buffer.meta_data['episode_lens'][experiences.valid_episodes])
            episode_ids = experiences.valid_episodes
        else:
            num_episodes = buffer.num_episodes
            all_task_ids = from_numpy(buffer.meta_data['task_id'][:num_episodes])
            episode_lens = from_numpy(buffer.meta_data['episode_lens'][:num_episodes])
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
                terminated  = 'bool',
                task_id     = 'int',
                invalidated = 'bool'
            )
        )

        # during roll out for gathering experience, they use positive advantage w/ cfg

        highest_advantage_id = actor.num_advantage_tokens - 1

        # mock env

        for _ in range(num_episodes):

            states = env.reset()

            with experience_buffer.one_episode(
                task_id = tensor(task_id),
                recap_step = tensor(recap_step)
            ):

                for _ in range(steps):

                    sampled_actions, _ = temp_batch_dim(actor)(
                        *states,
                        trajectory_length = trajectory_length,
                        steps = flow_sampling_steps,
                        critic = critic,
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
                        task_id = tensor(task_id)
                    )

                    if truncated or terminated:
                        break

        self.accelerate.wait_for_everyone()

        return experience_buffer

# fun

π0 = PiZero
