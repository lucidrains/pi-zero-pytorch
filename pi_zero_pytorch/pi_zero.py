from __future__ import annotations

from random import random

from beartype.typing import Callable
from beartype import beartype

from functools import partial

import torch
import torch.nn.functional as F
from torch import pi, nn, tensor, is_tensor
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from torchdiffeq import odeint

from scipy.optimize import linear_sum_assignment

from ema_pytorch import EMA

from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum, pack, unpack

from pi_zero_pytorch.tensor_typing import Float, Int, Bool

import tqdm

# ein notation

# b - batch
# n - sequence
# na - seq of actions
# nt - seq of text tokens
# nv - seq of visual tokens
# ns - seq of additional internal state tokens
# d - dimension
# da - action dimension
# djs - joint state dimension
# c - image channels
# h - image height
# w - image width
# f - image frames

# constants

LinearNoBias = partial(nn.Linear, bias = False)

# flex attention related
# https://pytorch.org/blog/flexattention/

flex_attention = None

if torch.cuda.is_available():
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention = torch.compile(flex_attention)

def create_pizero_attn_mask(prefix_causal_length, mask: Bool['b n']):
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

# tensor helpers

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

def noise_assignment(data, noise):
    device = data.device
    data, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (data, noise))
    dist = torch.cdist(data, noise)
    _, assign = linear_sum_assignment(dist.cpu())
    return torch.from_numpy(assign).to(device)

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
        rotary_emb: RotaryEmbedding | None = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.rotary_emb = rotary_emb

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.rmsnorm = nn.RMSNorm(dim)

        self.to_qkv = LinearNoBias(dim, 3 * dim_inner)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_actions_qkvg = LinearNoBias(dim, 4 * dim_inner)
        self.to_actions_out = LinearNoBias(dim_inner, dim)

        self.softclamp_value = softclamp_value

    def forward_actions_with_cached_state(
        self,
        actions,
        cached_state_keys_values: tuple[Tensor, Tensor],
        rotary_emb = None,
        mask: Bool['b n'] | None = None,
        actions_value_residual: Tensor | None = None,
        return_keys_values = False,
        flex_attn_fn: Callable | None = None
    ):
        aq, ak, av, ag = self.to_actions_qkvg(actions).chunk(4, dim = -1)

        aq, ak, av, ag = tuple(self.split_heads(t) for t in (aq, ak, av, ag))

        if exists(actions_value_residual):
            av = 0.5 * (av + actions_value_residual)

        q = aq
        mk, mv = cached_state_keys_values

        k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mk, mv), (ak, av)))

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

    def forward(
        self,
        multimodal_seq,
        actions,
        rotary_emb = None,
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
            av = 0.5 * (av + actions_value_residual)

        q, k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mq, mk, mv), (aq, ak, av)))

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

            causal_mask[..., seq_len:] = False  # actions have bidirectional attention, lining up with Transfusion paper

            sim = sim.masked_fill(causal_mask, max_neg_value(sim))

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # gating of values, used in alphafold line of work

        gates = pad_at_dim(ag.sigmoid(), (out.shape[-2] - ag.shape[-2], 0), value = 1., dim = -2)

        out = out * gates

        # merge attention heads

        out = self.merge_heads(out)

        # separate projections for multimodal seq vs actions

        mout, aout = out[:, :seq_len], out[:, seq_len:]

        output =  self.to_out(mout), self.to_actions_out(aout)

        if not return_keys_values:
            return output

        return output, (mk, mv, ak, av)

# attention

class SwiGLUFeedForward(Module):
    def __init__(
        self,
        dim,
        expand_factor = 4.,
        dim_inner = None
    ):
        super().__init__()
        dim_inner = default(dim_inner, int(dim * expand_factor * 2 / 3))

        self.rmsnorm = nn.RMSNorm(dim)
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
        reward_tokens_dropout_prob = 0.,
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

        # positional embedding

        self.rotary_emb = RotaryEmbedding(dim_head)

        # attention and feedforward

        layers = []
        cond_layers = []

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, **attn_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, **ff_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, **ff_kwargs)
            ]))

            cond_layers.append(ModuleList([
                AdaptiveRMSNorm(dim, dim_time_cond),
                AdaptiveLayerscale(dim, dim_time_cond),
                AdaptiveRMSNorm(dim, dim_time_cond),
                AdaptiveLayerscale(dim, dim_time_cond)
            ]))

        self.layers = ModuleList(layers)
        self.cond_layers = ModuleList(cond_layers)

        self.final_norm_softclamp = partial(softclamp, value = final_norm_softclamp_value)

        self.final_norm = nn.RMSNorm(dim)
        self.final_actions_norm = nn.RMSNorm(dim)

        # unembedding

        self.state_to_logits = LinearNoBias(dim, num_tokens)
        self.actions_to_pred_flow = LinearNoBias(dim, dim_action_input)

        # the language token id padding id, for fine-tuning as well as taking care of the masking on top of causal mask

        self.lm_pad_id = lm_pad_id

        # flow related

        self.immiscible_flow = immiscible_flow

        # reward classifier free guidance

        self.reward_tokens_dropout_prob = reward_tokens_dropout_prob

        # loss related

        self.lm_loss_weight = lm_loss_weight
        self.flow_loss_weight = flow_loss_weight

        # sampling related

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

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

    @torch.inference_mode()
    def sample_actions(
        self,
        images,
        token_ids,
        joint_states,
        trajectory_length: int,
        reward_tokens: Float['b d'] = None,
        internal_state_tokens: Float['b ns d'] | None = None,
        steps = 18,
        show_pbar = True,
        cond_scale = 0.,
        remove_parallel_component = True,
        keep_parallel_frac = 0.,
        cache_kv = True
    ):
        batch_size = token_ids.shape[0]

        was_training = self.training
        self.eval()

        pbar = tqdm.tqdm(desc = 'sampling action trajectory', disable = not show_pbar, total = steps)

        # ode step function

        cached_state_kv = None
        null_cached_state_kv = None

        def ode_fn(timestep, denoised_actions):
            nonlocal cached_state_kv
            nonlocal null_cached_state_kv

            flow, (new_cached_state_kv, new_null_cached_state_kv) = self.forward_with_reward_cfg(
                images,
                token_ids,
                joint_states,
                denoised_actions,
                times = timestep,
                reward_tokens = reward_tokens,
                internal_state_tokens = internal_state_tokens,
                cached_state_keys_values = (cached_state_kv, null_cached_state_kv),
                cond_scale = cond_scale,
                remove_parallel_component = remove_parallel_component,
                keep_parallel_frac = keep_parallel_frac,
            )

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

        return sampled_actions

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

    def forward(
        self,
        images: Float['b nv d'] | Float['b c h w'] | Float['b c f h w'], # vision
        token_ids: Int['b nt'],                                          # language
        joint_state: Float['b djs'],                                  # joint state
        actions: Float['b na da'] | None = None,                         # action
        times: Float['b'] = None,
        reward_tokens: Float['b d'] | None = None,
        internal_state_tokens: Float['b ns d'] | None = None,
        external_states: tuple[Float['b ...']] | None = None,
        return_actions_flow = False,
        return_state_keys_values = False,
        cached_state_keys_values: list[tuple[Tensor, Tensor]] | None = None,
        **kwargs
    ):
        inferencing = exists(cached_state_keys_values)
        assert not (inferencing and not return_actions_flow), 'must be generating action trajectory if receiving cached state key values'

        if not exists(actions):
            return self.sample_actions(images, token_ids, joint_state, **kwargs)

        batch, device = token_ids.shape[0], token_ids.device

        # noising the action for flow matching

        if not exists(times):
            times = torch.rand((batch,), device = device)

        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        # if not returning the actions predicted flow, assume training and noise the actions for loss

        if not return_actions_flow:
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

        action_register_tokens = repeat(self.action_register_tokens, '... -> b ...', b = batch)

        action_tokens, inverse_pack_action_registers = pack_with_inverse([action_register_tokens, action_tokens], 'b * d')

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

            # joint state

            joint_state_tokens = self.to_joint_state_tokens(joint_state)

            # maybe reward tokens

            if not exists(reward_tokens):
                reward_tokens = visual_tokens.new_empty((batch, 0, self.dim))

            # maybe dropout reward tokens

            if self.training and random() < self.reward_tokens_dropout_prob:
                reward_tokens = reward_tokens[:, 0:0]

            # additional internal state tokens

            if not exists(internal_state_tokens):
                internal_state_tokens = joint_state_tokens.new_empty((batch, 0, self.dim_internal_state))

            internal_state_tokens = self.to_internal_state_tokens(internal_state_tokens)

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
                joint_state_tokens,
                internal_state_tokens,
                reward_tokens
            ], 'b * d')


        # take care of masking for variable lengthed states, starting with the language tokens

        # which then leads to proper rotary embeddings

        command_length = token_ids.shape[-1]

        language_mask = token_ids != self.lm_pad_id

        action_with_registers_length = action_tokens.shape[-2]

        if inferencing:
            state_length = cached_state_keys_values[0][0].shape[-2]
        else:
            state_length = state_tokens.shape[-2]

        mask = F.pad(language_mask, (state_length - command_length - 1, 1 + action_with_registers_length), value = True) # assume fixed number of images for now, but address variable length modality states later

        # rotary embeddings

        seq = torch.cumsum(mask.float(), dim = -1)
        rotary_emb = self.rotary_emb(seq)

        rotary_emb = rearrange(rotary_emb, 'b n d -> b 1 n d')

        # prepare maybe flex attention

        flex_attn_fn = None

        if not inferencing and self.use_flex_attn and state_tokens.is_cuda:

            prefix_length = state_tokens.shape[-2]
            seq_len = prefix_length + action_tokens.shape[-2]

            block_mask = create_block_mask(
                create_pizero_attn_mask(prefix_length, mask = mask),
                Q_LEN = seq_len,
                KV_LEN = seq_len,
                device = state_tokens.device
            )

            score_mod_fn = softclamp_score_mod(self.attn_softclamp_value)

            flex_attn_fn = partial(
                flex_attention,
                block_mask = block_mask,
                score_mod = score_mod_fn
            )

        # state keys and values for caching during inference

        cached_state_key_values_iter = iter(default(cached_state_keys_values, []))

        state_cached_keys_values = []

        # value residual learning

        actions_value_residual = None

        # transformer

        if not inferencing:
            for (
                (attn, state_ff, actions_ff),
                (attn_ada_rmsnorm, attn_ada_layerscale, ff_ada_rmsnorm, ff_ada_layerscale)
            ) in zip(self.layers, self.cond_layers):

                action_tokens = attn_ada_rmsnorm(action_tokens, time_cond)

                (state_attn_out, actions_attn_out), (state_keys, state_values, action_keys, action_values) = attn(state_tokens, action_tokens, rotary_emb = rotary_emb, flex_attn_fn = flex_attn_fn, actions_value_residual = actions_value_residual, mask = mask, return_keys_values = True)

                state_cached_keys_values.append((state_keys, state_values))

                actions_value_residual = default(actions_value_residual, action_values)

                action_tokens = attn_ada_layerscale(action_tokens, time_cond)

                state_tokens = state_tokens + state_attn_out
                action_tokens = action_tokens + actions_attn_out

                state_tokens = state_ff(state_tokens) + state_tokens

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

                action_tokens = actions_ff(action_tokens) + action_tokens

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

        else:

            for (
                (attn, state_ff, actions_ff),
                (attn_ada_rmsnorm, attn_ada_layerscale, ff_ada_rmsnorm, ff_ada_layerscale)
            ) in zip(self.layers, self.cond_layers):

                action_tokens = attn_ada_rmsnorm(action_tokens, time_cond)

                actions_attn_out, (state_keys, state_values, action_keys, action_values) = attn.forward_actions_with_cached_state(action_tokens, cached_state_keys_values = next(cached_state_key_values_iter), rotary_emb = rotary_emb, mask = mask, return_keys_values = True)

                state_cached_keys_values.append((state_keys, state_values))

                actions_value_residual = default(actions_value_residual, action_values)

                action_tokens = attn_ada_layerscale(action_tokens, time_cond)

                action_tokens = action_tokens + actions_attn_out

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

                action_tokens = actions_ff(action_tokens) + action_tokens

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

        if not inferencing:
            # unpack and unembed to predictions

            _, visual_tokens, tokens, *_ = inverse_packed_states(state_tokens, 'b * d')

            # gemma uses a final softclamp before norm

            tokens = self.final_norm_softclamp(tokens)

        action_register_tokens, action_tokens = inverse_pack_action_registers(action_tokens)

        action_tokens = self.final_norm_softclamp(action_tokens)

        # projection

        actions = self.final_actions_norm(action_tokens)

        # flow loss for actions tokens

        pred_actions_flow = self.actions_to_pred_flow(actions)

        if return_actions_flow:
            if not return_state_keys_values:
                return pred_actions_flow

            return pred_actions_flow, state_cached_keys_values

        flow_loss = F.mse_loss(flow, pred_actions_flow)
        
        # language cross entropy loss

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

        return total_loss, loss_breakdown

# fun

π0 = PiZero
