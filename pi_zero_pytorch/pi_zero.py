from __future__ import annotations

from random import random, randrange

from beartype import beartype
from beartype.typing import Callable, Literal

from functools import partial, wraps
from itertools import count

import torch
import torch.nn.functional as F
from torch import pi, nn, stack, tensor, is_tensor
from torch.nn import Module, ModuleList
from torch.distributions import Normal
from torch.distributions.beta import Beta

from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from torchdiffeq import odeint

from scipy.optimize import linear_sum_assignment

from ema_pytorch import EMA

from adam_atan2_pytorch import AdoptAtan2

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

import tqdm

# ein notation

# b - batch
# n - sequence
# na - seq of actions
# nt - seq of text tokens
# nv - seq of visual tokens
# ns - seq of additional internal state tokens
# nm - seq of memory tokens
# d - dimension
# da - action dimension
# djs - joint state dimension
# c - image channels
# h - image height
# w - image width
# f - image frames
# s - residual streams (hyper connections paper)

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

        return (key_mask and causal_mask) or bidirectional_action_mask

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

def to_device(t, device):
    return tree_map(lambda el: el.to(device) if is_tensor(el) else el, t)

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

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim)

def softclamp(t, value):
    if value <= 0.:
        return t

    return (t / value).tanh() * value

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

    alpha = torch.full(shape, alpha, device = device)
    beta = torch.full(shape, beta, device = device)
    sampled = Beta(alpha, beta).sample()
    return (1. - sampled) * s

def noise_assignment(data, noise):
    device = data.device
    data, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (data, noise))
    dist = torch.cdist(data, noise)
    _, assign = linear_sum_assignment(dist.cpu())
    return torch.from_numpy(assign).to(device)

# policy optimization related

class GaussianNLL(Module):
    def forward(self, mu_sigma, target):

        mean, variance = mu_sigma.unbind(dim = -1)
        dist = Normal(mean, variance)
        log_probs = dist.log_prob(target)

        return -log_probs.mean()

class LinearToMeanVariance(Module):
    def __init__(
        self,
        dim,
        dim_out,
    ):
        super().__init__()

        self.linear = nn.Sequential(
            LinearNoBias(dim, dim_out * 2),
            Rearrange('... (d mu_sigma) -> ... d mu_sigma', mu_sigma = 2)
        )

    def forward(self, embed):
        out = self.linear(embed)
        mean, log_variance = out.unbind(dim = -1)
        variance = log_variance.exp()
        return stack((mean, variance), dim = -1)

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
        flex_attn_fn: Callable | None = None
    ):
        aq, ak, av, ag = self.to_actions_qkvg(actions).chunk(4, dim = -1)

        aq, ak, av, ag = tuple(self.split_heads(t) for t in (aq, ak, av, ag))

        if exists(actions_value_residual):
            mix = self.to_action_value_residual_mix(actions)
            av = av * mix + actions_value_residual * (1. - mix)

        q = aq
        mk, mv = cached_state_keys_values

        # concat cache key / values with action key / values

        k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mk, mv), (ak, av)))

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
        flex_attn_fn: Callable | None = None
    ):
        seq_len, device = multimodal_seq.shape[-2], multimodal_seq.device

        multimodal_seq = self.rmsnorm(multimodal_seq)

        # separate projections for multimodal seq vs actions

        mq, mk, mv = self.to_qkv(multimodal_seq).chunk(3, dim = -1)

        aq, ak, av, ag = self.to_actions_qkvg(actions).chunk(4, dim = -1)

        mq, mk, mv, aq, ak, av, ag = tuple(self.split_heads(t) for t in (mq, mk, mv, aq, ak, av, ag))

        if exists(actions_value_residual):
            mix = self.to_action_value_residual_mix(actions)
            av = av * mix + actions_value_residual * (1. - mix)

        q, k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mq, mk, mv), (aq, ak, av)))

        # handle read, write memories

        has_memories = exists(memories) and any([m.numel() > 0 for m in memories])

        assert not (self.accept_memories ^ has_memories)

        if has_memories:
            memories, unpack_memories = pack_with_inverse(memories, 'b * d')
            memories = self.mem_rmsnorm(memories)
            mqkv = self.to_mem_qkv(memories)
            mqkv_read, mqkv_write = unpack_memories(mqkv, 'b * d')

            mqr, mkr, mvr, mqw, mkw, mvw = tuple(self.split_heads(t) for t in (*mqkv_read.chunk(3, dim = -1), *mqkv_write.chunk(3, dim = -1)))

            k = torch.cat((mkr, k, mkw), dim = -2)
            v = torch.cat((mvr, v, mvw), dim = -2)
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

        output =  self.to_out(mout), self.to_actions_out(aout)

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
        rmsnorm = True
    ):
        super().__init__()
        dim_inner = default(dim_inner, int(dim * expand_factor * 2 / 3))

        self.rmsnorm = nn.RMSNorm(dim) if rmsnorm else nn.Identity()
        self.proj_in = LinearNoBias(dim, dim_inner * 2)
        self.proj_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        seq
    ):
        seq = self.rmsnorm(seq)
        seq, gates = self.proj_in(seq).chunk(2, dim = -1)
        seq = seq * F.gelu(gates)
        return self.proj_out(seq)

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
        flow_loss_weight = 1.,
        immiscible_flow = False, # https://arxiv.org/abs/2406.12303
        sample_times_fn = default_sample_times,
        reward_tokens_dropout_prob = 0.,
        num_recurrent_memory_tokens = 0,
        num_residual_streams = 1,
        dim_latent_gene = None,
        policy_optimizable = False,         # if set to True, will use mean variance network for access to log prob
        is_critic = False,                  # whether this model is used as the critic, with the histogram classification loss from Imani et al. https://arxiv.org/html/2402.13425v1
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

        needs_latent = exists(dim_latent_gene)
        self.needs_latent = needs_latent

        if needs_latent:
            self.to_latent_cond = nn.Sequential(
                nn.Linear(dim_latent_gene, dim_time_cond * 2),
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
                Attention(dim = dim, dim_head = dim_head, heads = heads, accept_memories = self.has_recurrent_memories, learned_value_action_residual_mix = not is_first_block, **attn_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, **ff_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, rmsnorm = False, **ff_kwargs),
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

        # actor related

        self.actions_to_pred_flow = None
        self.loss_fn = None

        if not is_critic:
            if not policy_optimizable:
                self.actions_to_pred_flow = LinearNoBias(dim, dim_action_input)
                self.loss_fn = nn.MSELoss()
            else:
                self.actions_to_pred_flow = LinearToMeanVariance(dim, dim_action_input)
                self.loss_fn = GaussianNLL()

        self.is_mean_variance_output = policy_optimizable
        self.policy_optimizable = policy_optimizable

        # critic related

        self.is_critic = is_critic

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

        # time sampling related

        self.sample_times_fn = default(sample_times_fn, torch.rand)

        # loss related

        self.lm_loss_weight = lm_loss_weight
        self.flow_loss_weight = flow_loss_weight

        # sampling related

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # tensor typing

        self._nm = num_recurrent_memory_tokens

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

    def create_critic(self, **kwargs) -> PiZero:
        assert self.policy_optimizable and not self.is_critic, 'base model must be policy optimizable as well as not a critic already'

        assert 'is_critic' not in kwargs
        kwargs.update(is_critic = True)

        orig_args, orig_kwargs = self._init_args_kwargs
        critic = PiZero(*orig_args, **orig_kwargs, **kwargs)

        # load all possible shared parameters except for output head to logits (for histogram loss)

        state_dict = self.state_dict()
        critic.load_state_dict(state_dict, strict = False)

        return critic.to(self.device)

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        token_ids,
        joint_states,
        trajectory_length: int,
        latents: Float['d'] | Float['b d'] = None,
        reward_tokens: Float['b d'] | None = None,
        internal_state_tokens: Float['b ns d'] | None = None,
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

        # accumulate log probs for ppo

        assert not (return_states_for_replay and not self.is_mean_variance_output), 'only pi-zero with `policy_optimizable` turned on can return log probs'

        timesteps = []
        log_probs = []
        sampled_flows = []
        denoised_actions_across_time = []

        critic_values = []

        # ode step function

        cached_state_kv = None
        null_cached_state_kv = None

        def ode_fn(timestep, denoised_actions):
            nonlocal cached_state_kv
            nonlocal null_cached_state_kv

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
                internal_state_tokens = internal_state_tokens,
                cached_state_keys_values = (cached_state_kv, null_cached_state_kv),
                cond_scale = cond_scale,
                remove_parallel_component = remove_parallel_component,
                keep_parallel_frac = keep_parallel_frac
            )

            output, (new_cached_state_kv, new_null_cached_state_kv) = self.forward_with_reward_cfg(*input_args, **input_kwargs)

            if exists(critic):
                critic_value, _ = critic.forward_with_reward_cfg(*input_args, **input_kwargs)
                critic_values.append(critic_value)

            flow = output

            if self.is_mean_variance_output:
                mean, variance = output.unbind(dim = -1)

                flow = torch.normal(mean, variance * temperature)

                log_prob = Normal(mean, variance).log_prob(flow)

                # save for replaying for optimizing actor

                denoised_actions_across_time.append(denoised_actions)
                timesteps.append(repeat(timestep, ' -> b', b = batch_size))
                log_probs.append(log_prob)
                sampled_flows.append(flow)

            if cache_kv:
                cached_state_kv = new_cached_state_kv
                null_cached_state_kv = new_null_cached_state_kv

            pbar.update(1)

            return flow

        # start with random gaussian noise - y0

        noise = torch.randn((batch_size, trajectory_length, self.dim_action_input), device = self.device)

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = self.odeint_fn(ode_fn, noise, times)

        sampled_actions = trajectory[-1]

        self.train(was_training)

        pbar.close()

        if not return_states_for_replay:
            return sampled_actions

        # place the time step dimension after batch

        timesteps = stack(timesteps, dim = 1)
        log_probs = stack(log_probs, dim = 1)
        sampled_flow = stack(sampled_flows, dim = 1)
        denoised_actions_across_time = stack(denoised_actions_across_time, dim = 1)

        policy_optimization_outputs = (denoised_actions_across_time, timesteps, log_probs, sampled_flow)

        # return critic value predictions if passed in - will just deepcopy pi-zero + a critic head

        if exists(critic):
            critic_values = stack(critic_values, dim = 1)
            policy_optimization_outputs = (*policy_optimization_outputs, critic_values)

        return sampled_actions, policy_optimization_outputs

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

    @move_input_tensors_to_device
    def forward_for_policy_loss(
        self,
        images,
        commands,
        joint_state,
        actions,
        flow,
        times,
        old_log_probs: Float['b na'],
        advantages: Float['b t'],
        clip_eps = 0.2,
        entropy_weight = 1e-2,
        norm_eps = 1e-5,
        **kwargs,
    ):
        assert not self.is_critic
        assert self.policy_optimizable
        assert 'return_actions_flow' not in kwargs

        # flatten the time into the batch for actions at timestep, sampled flow, and log prob

        if times.ndim == 2:
            times = rearrange(times, 'b t -> (b t)')

        if flow.ndim == 4:
            flow = rearrange(flow, 'b t ... -> (b t) ...')

        if old_log_probs.ndim == 4:
            old_log_probs = rearrange(old_log_probs, 'b t ... -> (b t) ...')

        if actions.ndim == 4:
            actions = rearrange(actions, 'b t ... -> (b t) ...')

        # expand inputs across timesteps if need be

        (
            images,
            commands,
            joint_state,
        ) = tuple(repeat(inp, 'b ... -> (b t) ...', t = times.shape[0] // inp.shape[0]) for inp in (
            images,
            commands,
            joint_state
        ))

        mean_variance = self.forward(
            images,
            commands,
            joint_state,
            actions,
            return_actions_flow = True,
            **kwargs
        )

        normal_dist = Normal(*mean_variance.unbind(dim = -1))

        new_log_probs = normal_dist.log_prob(actions)

        # ppo surrogate loss

        ratio = (new_log_probs - old_log_probs).exp()

        advantages = F.layer_norm(advantages, advantages.shape, eps = norm_eps)

        advantages = rearrange(advantages, 'b t -> (b t) 1 1')

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1. - clip_eps, 1. + clip_eps) * advantages

        clipped_surr_loss = torch.min(surr1, surr2).sum(dim = -1)

        # entropy

        entropy = (normal_dist.entropy() * entropy_weight).sum(dim = -1)

        return -(clipped_surr_loss + entropy * entropy_weight).mean()

    @move_input_tensors_to_device
    def forward_for_critic_loss(
        self,
        *args,
        old_values: Float['b t'],
        advantages: Float['b t'],
        clip_eps = 0.4,
        **kwargs
    ):
        assert self.is_critic

        eps = clip_eps
        loss_fn = self.to_critic_value.loss_fn

        critic_value, critic_logits = self.forward(*args, **kwargs)

        # value clipping

        advantages = rearrange(advantages, 'b t -> (b t)')
        old_values = rearrange(old_values, 'b t -> (b t)')

        returns = old_values + advantages

        clipped_value = old_values + (critic_value - old_values).clamp(-eps, eps)

        clipped_loss = loss_fn(clipped_value, returns, reduction = 'none')
        loss = loss_fn(critic_logits, returns, reduction = 'none')

        return torch.max(clipped_loss, loss).mean()

    @move_input_tensors_to_device
    def forward(
        self,
        images: Float['b nv d'] | Float['b c h w'] | Float['b c f h w'], # vision
        token_ids: Int['b nt'],                                          # language
        joint_state: Float['b djs'],                                     # joint state
        actions: Float['b na da'] | None = None,                         # action
        times: Float['b'] = None,
        latents: Float['d'] | Float['b d'] = None,
        reward_tokens: Float['b d'] | None = None,
        internal_state_tokens: Float['b ns d'] | None = None,
        external_states: tuple[Float['b ...']] | None = None,
        record_and_return_memory_tokens = False,
        past_recurrent_memory_tokens: Float['b {self._nm} d'] | None = None,
        return_actions_flow = False,
        return_state_keys_values = False,
        cached_state_keys_values: list[tuple[Tensor, Tensor]] | None = None,
        return_language_loss = True,
        return_action_flow_loss = True,
        **kwargs
    ):
        inferencing = exists(cached_state_keys_values)
        assert not (inferencing and not return_actions_flow), 'must be generating action trajectory if receiving cached state key values'

        if not exists(actions) and not self.is_critic:
            return self.sample_actions(images, token_ids, joint_state, **kwargs)

        batch, device = token_ids.shape[0], token_ids.device

        # noising the action for flow matching

        if not exists(times):
            times = self.sample_times_fn((batch,), device = device)

        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        # handle latent genes

        if exists(latents) and latents.ndim == 1:
            latents = repeat(latents, 'd -> b d', b = batch)

        # if not returning the actions predicted flow, assume training and noise the actions for loss

        if not return_actions_flow and not self.is_critic:
            noise = torch.randn_like(actions)

            if self.immiscible_flow:
                assignment = noise_assignment(actions, noise)
                noise = noise[assignment]

            flow = actions - noise
            padded_times = rearrange(times, 'b -> b 1 1')

            actions = noise.lerp(actions, padded_times)

        # actions

        time_cond = self.to_time_cond(times)
        action_tokens = self.to_action_tokens(actions)

        # handle maybe latents

        assert xnor(exists(latents), self.needs_latent)

        if self.needs_latent:
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

            # maybe reward tokens

            if not exists(reward_tokens):
                reward_tokens = visual_tokens.new_empty((batch, 0, self.dim))

            # maybe dropout reward tokens

            if self.training and random() < self.reward_tokens_dropout_prob:
                reward_tokens = reward_tokens[:, 0:0]

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
                reward_tokens
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

        assert return_language_loss or return_action_flow_loss

        # flow loss for actions tokens

        pred_actions_flow = self.actions_to_pred_flow(action_embeds)

        if return_actions_flow:

            if not return_state_keys_values and not record_and_return_memory_tokens:
                return pred_actions_flow

            if not return_state_keys_values:
                return pred_actions_flow, written_memory_tokens

            return pred_actions_flow, next_state_cached_keys_values

        flow_loss = self.zero

        if return_action_flow_loss:
            flow_loss = self.loss_fn(pred_actions_flow, flow)

        # language cross entropy loss

        language_loss = self.zero

        if return_language_loss:
            tokens = self.final_norm(tokens)

            language_logits = self.state_to_logits(tokens)

            language_loss = F.cross_entropy(
                rearrange(language_logits[:, :-1], 'b n l -> b l n'),
                labels,
                ignore_index = self.lm_pad_id
            )

        # loss breakdown

        loss_breakdown = (language_loss, flow_loss)

        # total loss and return breakdown

        total_loss = (
            language_loss * self.lm_loss_weight +
            flow_loss * self.flow_loss_weight
        )

        if not record_and_return_memory_tokens:
            return total_loss, loss_breakdown

        return total_loss, loss_breakdown, written_memory_tokens

# generalized advantage estimate

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

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[:, :-1], values[:, 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    return scan(gates, delta)

# agent

class Agent(Module):
    def __init__(
        self,
        model: PiZero,
        optim_klass = AdoptAtan2,
        actor_lr = 3e-4,
        critic_lr = 3e-4,
        actor_weight_decay = 1e-3,
        critic_weight_decay = 1e-3,
        actor_optim_kwargs: dict = dict(),
        critic_optim_kwargs: dict = dict(),
    ):
        super().__init__()
        assert model.policy_optimizable

        self.actor = model
        self.critic = model.create_critic()

        # optimizers

        self.actor_optim = optim_klass(self.actor.parameters(), lr = actor_lr, weight_decay = actor_weight_decay, **actor_optim_kwargs)
        self.critic_optim = optim_klass(self.critic.parameters(), lr = critic_lr, weight_decay = critic_weight_decay, **critic_optim_kwargs)

    def forward(
        self,
        memories
    ):
        raise NotImplementedError

from torch.utils.data import TensorDataset, DataLoader

class EPO(Module):
    def __init__(
        self,
        agent: Agent,
        env,
    ):
        super().__init__()
        self.agent = agent
        self.env = env

    @torch.no_grad()
    def gather_experience_from_env(
        self,
        steps
    ):
        self.agent.eval()

        states = self.env.reset()

        memories = []

        for _ in range(steps):

            sampled_actions, replay_tensors = temp_batch_dim(self.agent.actor)(
                *states,
                trajectory_length = 4,
                steps = 2,
                return_states_for_replay = True
            )

            next_states, reward, truncated, terminated = self.env.step(sampled_actions)

            memories.append(to_device([*states, reward, terminated, *replay_tensors], torch.device('cpu')))

            states = next_states

        return memories

    def learn_agent(
        self,
        memories,
        epochs = 2,
        batch_size = 16
    ):
        self.agent.train()

        (
            images,
            commands,
            joint_state,
            rewards,
            terminated,
            actions,
            timesteps,
            sampled_flows,
            log_probs
        ) = map(torch.stack, zip(*memories))

        flow_timesteps = actions.shape[1]

        values, _ = self.agent.critic(
            repeat(images, 't ... -> (t ft) ...', ft = flow_timesteps),
            repeat(commands, 't ... -> (t ft) ...', ft = flow_timesteps),
            repeat(joint_state, 't ... -> (t ft) ...', ft = flow_timesteps),
            actions = rearrange(actions, 't ft ... -> (t ft) ...'),
            times = rearrange(timesteps, 't ft ... -> (t ft) ...')
        )

        values = rearrange(values, '(t ft) -> ft t', ft = flow_timesteps)
        values = values.detach().cpu()

        # actions go out into the environment, rewards are received, generalized advantage calculated with critic values

        boundaries = repeat(terminated, 't -> ft t', ft = flow_timesteps)

        advantages = calc_generalized_advantage_estimate(rewards, values, boundaries, use_accelerated = False).detach()

        # move time back to first dimension to be batched for learning

        advantages = rearrange(advantages, 'ft t -> t ft')
        values = rearrange(values, 'ft t -> t ft')

        # dataset and dataloader

        dataset = TensorDataset(
            images,
            commands,
            joint_state,
            rewards,
            terminated,
            actions,
            timesteps,
            sampled_flows,
            log_probs,
            values,
            advantages
        )

        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

        # training loop

        for _ in range(epochs):
            for (
                images,
                commands,
                joint_state,
                rewards,
                terminated,
                actions,
                timesteps,
                sampled_flows,
                log_probs,
                values,
                advantages
            ) in dataloader:

                # optimize policy with replay tensors from above

                actor_loss = self.agent.actor.forward_for_policy_loss(
                    images,
                    commands,
                    joint_state,
                    actions,
                    times = timesteps,
                    flow = sampled_flows,
                    old_log_probs = log_probs,
                    advantages = advantages,
                )

                actor_loss.backward()

                self.agent.actor_optim.step()
                self.agent.actor_optim.zero_grad()

                critic_loss = self.agent.critic.forward_for_critic_loss(
                    repeat(images, 't ... -> (ft t) ...', ft = flow_timesteps),
                    repeat(commands, 't ... -> (ft t) ...', ft = flow_timesteps),
                    repeat(joint_state, 't ... -> (ft t) ...', ft = flow_timesteps),
                    rearrange(actions, 't ft ... -> (ft t) ...'),
                    old_values = values,
                    advantages = advantages,
                )

                critic_loss.backward()

                self.agent.critic_optim.step()
                self.agent.critic_optim.zero_grad()

# fun

π0 = PiZero
