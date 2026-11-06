from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from beartype.typing import Callable
from einops import einsum, rearrange, repeat
from torch import Tensor
from torch_einops_utils.shape import shape, size

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# flex attention

flex_attention = None
create_block_mask = None

if torch.cuda.is_available():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    flex_attention = torch.compile(flex_attention)


# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max


# gradient masks

@dataclass(frozen = True)
class StopGradMask:
    # rows x cols block where gradient is stopped. dense masks are allow masks
    rows: Tensor
    cols: Tensor

GradMask = Tensor | StopGradMask

def create_knowledge_insulation_grad_mask(
    seq_len: int,
    start: int,
    end: int,
    device = None
) -> StopGradMask:
    rows = torch.ones(seq_len, dtype = torch.bool, device = device)
    rows[start:end] = False

    cols = torch.zeros(seq_len, dtype = torch.bool, device = device)
    cols[start:end] = True

    return StopGradMask(rows = rows, cols = cols)

def _apply_grad_mask(x, mask):
    if not exists(mask):
        return x

    if isinstance(mask, StopGradMask):
        rows = mask.rows.to(x.device).bool()
        cols = mask.cols.to(x.device).bool()
        mask = ~(rows[..., :, None] & cols[..., None, :])

    assert mask.dtype == torch.bool, 'dense gradient mask must be boolean'

    return x.masked_fill(~mask, 0.)


# attention with per-edge gradient control

class _AttentionWithGradMask(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attn_mask,
        grad_mask_q,
        grad_mask_k,
        grad_mask_v,
        scale,
        flex_attn_fn
    ):
        if exists(flex_attn_fn):
            out = flex_attn_fn(q, k, v)
            ctx.save_for_backward(q, k, v)
            ctx.recompute_attn = True
        else:
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * scale

            if exists(attn_mask):
                sim = sim.masked_fill(~attn_mask, max_neg_value(sim))

            attn = sim.softmax(dim = -1, dtype = torch.float32).to(sim.dtype)
            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

            ctx.save_for_backward(q, k, v, attn)
            ctx.recompute_attn = False

        ctx.attn_mask = attn_mask
        ctx.grad_mask_q = grad_mask_q
        ctx.grad_mask_k = grad_mask_k
        ctx.grad_mask_v = grad_mask_v
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, do):
        if ctx.recompute_attn:
            q, k, v = ctx.saved_tensors

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * ctx.scale

            if exists(ctx.attn_mask):
                sim = sim.masked_fill(~ctx.attn_mask, max_neg_value(sim))

            attn = sim.softmax(dim = -1, dtype = torch.float32).to(sim.dtype)
        else:
            q, k, v, attn = ctx.saved_tensors

        dattn = einsum(do, v, 'b h i d, b h j d -> b h i j')

        # mask attention-matrix gradients after softmax

        ds = attn * (dattn - (dattn * attn).sum(dim = -1, keepdim = True))

        dq = dk = dv = None

        if ctx.needs_input_grad[0]:
            ds_q = _apply_grad_mask(ds, ctx.grad_mask_q)
            dq = einsum(ds_q, k, 'b h i j, b h j d -> b h i d') * ctx.scale

        if ctx.needs_input_grad[1]:
            ds_k = _apply_grad_mask(ds, ctx.grad_mask_k)
            dk = einsum(ds_k, q, 'b h i j, b h i d -> b h j d') * ctx.scale

        if ctx.needs_input_grad[2]:
            p = _apply_grad_mask(attn, ctx.grad_mask_v)
            dv = einsum(p, do, 'b h i j, b h i d -> b h j d')

        return dq, dk, dv, None, None, None, None, None, None


def _manual_attention(q, k, v, attn_mask, scale):
    q = q * scale

    sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

    if exists(attn_mask):
        sim = sim.masked_fill(~attn_mask, max_neg_value(sim))

    attn = sim.softmax(dim = -1, dtype = torch.float32).to(sim.dtype)

    return einsum(attn, v, 'b h i j, b h j d -> b h i d')


# triton flash attention with stop grad masks

if exists(triton):

    @triton.jit
    def _flash_fwd(
        Q, K, V, Mask, RowK, ColK, RowV, ColV, Out, Lse,
        scale,
        sqb, sqh, sqm, sqk,
        skb, skh, skn, skk,
        svb, svh, svn, svk,
        smb, smm, smn,
        sob, soh, som, sok,
        n_heads, N, D: tl.constexpr,
        HAS_MASK: tl.constexpr, HAS_K: tl.constexpr, HAS_V: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        sm = tl.program_id(0)
        hb = tl.program_id(1)
        b, h = hb // n_heads, hb % n_heads

        offs_m = sm * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)

        q = tl.load(
            Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
            mask = offs_m[:, None] < N, other = 0.
        )

        m_i = tl.zeros([BLOCK_M], dtype = tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype = tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype = tl.float32)

        mask_base = Mask + b * smb + offs_m[:, None] * smm + offs_n[None, :] * smn

        for start_n in range(0, N, BLOCK_N):
            offs_n_curr = start_n + offs_n

            k = tl.load(
                K + b * skb + h * skh + offs_n_curr[:, None] * skn + offs_d[None, :] * skk,
                mask = offs_n_curr[:, None] < N, other = 0.
            )

            qk = tl.dot(q, tl.trans(k), allow_tf32 = False) * scale

            if HAS_MASK:
                m = tl.load(
                    mask_base + start_n * smn,
                    mask = (offs_m[:, None] < N) & (offs_n_curr[None, :] < N),
                    other = 0
                ).to(tl.int1)
                qk = tl.where(m, qk, float('-inf'))
            else:
                qk = tl.where(offs_n_curr[None, :] < N, qk, float('-inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            p = tl.where(qk == float('-inf'), 0., p)
            alpha = tl.exp(m_i - m_ij)
            alpha = tl.where(m_i == float('-inf'), 0., alpha)

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v = tl.load(
                V + b * svb + h * svh + offs_n_curr[:, None] * svn + offs_d[None, :] * svk,
                mask = offs_n_curr[:, None] < N, other = 0.
            )

            acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32 = False)
            m_i = m_ij

        l_safe = tl.where(l_i == 0., 1., l_i)
        acc = acc / l_safe[:, None]

        tl.store(
            Out + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sok,
            acc.to(Q.dtype.element_ty),
            mask = offs_m[:, None] < N
        )

        tl.store(Lse + hb * N + offs_m, m_i + tl.log(l_safe), mask = offs_m < N)

    @triton.jit
    def _flash_bwd(
        Q, K, V, Mask, RowK, ColK, RowV, ColV,
        Out, DO, Lse, DQ, DK, DV,
        scale,
        sqb, sqh, sqm, sqk,
        skb, skh, skn, skk,
        svb, svh, svn, svk,
        smb, smm, smn,
        sob, soh, som, sok,
        sdb, sdh, sdm, sdk,
        n_heads, N, D: tl.constexpr,
        HAS_MASK: tl.constexpr, HAS_K: tl.constexpr, HAS_V: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        sn = tl.program_id(0)
        hb = tl.program_id(1)
        b, h = hb // n_heads, hb % n_heads

        offs_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        k = tl.load(
            K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :] * skk,
            mask = offs_n[:, None] < N, other = 0.
        )
        v = tl.load(
            V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :] * svk,
            mask = offs_n[:, None] < N, other = 0.
        )

        dk = tl.zeros([BLOCK_N, D], dtype = tl.float32)
        dv = tl.zeros([BLOCK_N, D], dtype = tl.float32)

        mask_base = Mask + b * smb + offs_m[:, None] * smm + offs_n[None, :] * smn

        for start_m in range(0, N, BLOCK_M):
            offs_m_curr = start_m + offs_m

            q = tl.load(
                Q + b * sqb + h * sqh + offs_m_curr[:, None] * sqm + offs_d[None, :] * sqk,
                mask = offs_m_curr[:, None] < N, other = 0.
            )
            do = tl.load(
                DO + b * sdb + h * sdh + offs_m_curr[:, None] * sdm + offs_d[None, :] * sdk,
                mask = offs_m_curr[:, None] < N, other = 0.
            )
            out = tl.load(
                Out + b * sob + h * soh + offs_m_curr[:, None] * som + offs_d[None, :] * sok,
                mask = offs_m_curr[:, None] < N, other = 0.
            )
            lse = tl.load(Lse + hb * N + offs_m_curr, mask = offs_m_curr < N, other = 0.)

            qk = tl.dot(q, tl.trans(k), allow_tf32 = False) * scale

            if HAS_MASK:
                m = tl.load(
                    mask_base + start_m * smm,
                    mask = (offs_m_curr[:, None] < N) & (offs_n[None, :] < N),
                    other = 0
                ).to(tl.int1)
                qk = tl.where(m, qk, float('-inf'))
            else:
                qk = tl.where(offs_n[None, :] < N, qk, float('-inf'))

            qk = tl.where(offs_m_curr[:, None] < N, qk, float('-inf'))

            p = tl.where(qk == float('-inf'), 0., tl.exp(qk - lse[:, None]))

            dp = tl.dot(do, tl.trans(v), allow_tf32 = False)
            di = tl.sum(do * out, axis = 1)
            ds = p * (dp - di[:, None])

            ds_k = ds
            if HAS_K:
                row_k = tl.load(RowK + offs_m_curr, mask = offs_m_curr < N, other = 0).to(tl.int1)
                col_k = tl.load(ColK + offs_n, mask = offs_n < N, other = 0).to(tl.int1)
                ds_k = tl.where(row_k[:, None] & col_k[None, :], 0., ds)

            p_v = p
            if HAS_V:
                row_v = tl.load(RowV + offs_m_curr, mask = offs_m_curr < N, other = 0).to(tl.int1)
                col_v = tl.load(ColV + offs_n, mask = offs_n < N, other = 0).to(tl.int1)
                p_v = tl.where(row_v[:, None] & col_v[None, :], 0., p)

            dk += tl.dot(tl.trans(ds_k.to(Q.dtype.element_ty)), q, allow_tf32 = False) * scale
            dv += tl.dot(tl.trans(p_v.to(DO.dtype.element_ty)), do, allow_tf32 = False)

            dq = tl.dot(ds.to(K.dtype.element_ty), k, allow_tf32 = False) * scale
            tl.atomic_add(
                DQ + b * sqb + h * sqh + offs_m_curr[:, None] * sqm + offs_d[None, :] * sqk,
                dq.to(DQ.dtype.element_ty),
                mask = offs_m_curr[:, None] < N
            )

        tl.atomic_add(
            DK + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :] * skk,
            dk.to(DK.dtype.element_ty),
            mask = offs_n[:, None] < N
        )
        tl.atomic_add(
            DV + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :] * svk,
            dv.to(DV.dtype.element_ty),
            mask = offs_n[:, None] < N
        )


def can_use_triton(q, k, v, mask, grad_mask_q, grad_mask_k, grad_mask_v):
    if not exists(triton) or not q.is_cuda:
        return False

    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False

    query_heads = size(q, 'b [h] n d')

    if query_heads != size(k, 'b [h] n d') or query_heads != size(v, 'b [h] n d'):
        return False

    dim = size(q, '... [d]')
    if dim < 16 or dim != triton.next_power_of_2(dim):
        return False

    seq_len = size(q, 'b h [n] d')
    if seq_len != size(k, 'b h [n] d') or seq_len != size(v, 'b h [n] d'):
        return False

    if not exists(mask) or mask.dtype != torch.bool or mask.ndim != 4:
        return False

    if size(mask, 'b [h] nq nk') != 1:
        return False

    if size(mask, 'b h [nq] nk') != seq_len or size(mask, 'b h nq [nk]') != seq_len:
        return False

    if size(mask, '[b] h nq nk') not in (1, size(q, '[b] h n d')):
        return False

    if exists(grad_mask_q):
        return False

    for grad_mask in (grad_mask_k, grad_mask_v):
        if exists(grad_mask) and not (
            isinstance(grad_mask, StopGradMask) and
            grad_mask.rows.ndim == 1 and grad_mask.cols.ndim == 1 and
            size(grad_mask.rows, '[n]') == seq_len and size(grad_mask.cols, '[n]') == seq_len
        ):
            return False

    return True


class _FlashAttentionWithGradMask(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, mask, grad_mask_k, grad_mask_v, scale):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        mask = mask.contiguous()

        batch, heads, seq_len, dim = shape(q, 'b h n d')

        row_k = grad_mask_k.rows.contiguous() if exists(grad_mask_k) else torch.empty(0, dtype = torch.bool, device = q.device)
        col_k = grad_mask_k.cols.contiguous() if exists(grad_mask_k) else torch.empty(0, dtype = torch.bool, device = q.device)
        row_v = grad_mask_v.rows.contiguous() if exists(grad_mask_v) else torch.empty(0, dtype = torch.bool, device = q.device)
        col_v = grad_mask_v.cols.contiguous() if exists(grad_mask_v) else torch.empty(0, dtype = torch.bool, device = q.device)

        out = torch.empty_like(q)
        lse = torch.empty((batch, heads, seq_len), dtype = torch.float32, device = q.device)

        grid = lambda META: (triton.cdiv(seq_len, META['BLOCK_M']), batch * heads)

        _flash_fwd[grid](
            q, k, v, mask, row_k, col_k, row_v, col_v, out, lse,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            mask.stride(0) if size(mask, '[b] h nq nk') > 1 else 0, mask.stride(2), mask.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            heads, seq_len, dim,
            HAS_MASK = True, HAS_K = exists(grad_mask_k), HAS_V = exists(grad_mask_v),
            BLOCK_M = 32, BLOCK_N = 32
        )

        ctx.save_for_backward(q, k, v, mask, row_k, col_k, row_v, col_v, out, lse)
        ctx.scale = scale
        ctx.has_k = exists(grad_mask_k)
        ctx.has_v = exists(grad_mask_v)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, mask, row_k, col_k, row_v, col_v, out, lse = ctx.saved_tensors
        q, k, v, do = q.contiguous(), k.contiguous(), v.contiguous(), do.contiguous()

        batch, heads, seq_len, dim = shape(q, 'b h n d')

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        grid = lambda META: (triton.cdiv(seq_len, META['BLOCK_N']), batch * heads)

        _flash_bwd[grid](
            q, k, v, mask, row_k, col_k, row_v, col_v,
            out, do, lse, dq, dk, dv,
            ctx.scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            mask.stride(0) if size(mask, '[b] h nq nk') > 1 else 0, mask.stride(2), mask.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            heads, seq_len, dim,
            HAS_MASK = True, HAS_K = ctx.has_k, HAS_V = ctx.has_v,
            BLOCK_M = 32, BLOCK_N = 32
        )

        return dq, dk, dv, None, None, None, None


def flash_attn_with_grad_mask(q, k, v, mask, grad_mask_k = None, grad_mask_v = None, scale = None):
    assert exists(triton), 'triton must be installed for flash_attn_with_grad_mask'
    scale = default(scale, size(q, '... [d]') ** -0.5)
    return _FlashAttentionWithGradMask.apply(q, k, v, mask, grad_mask_k, grad_mask_v, scale)


def attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float | None = None,
    mask: Tensor | None = None,
    grad_mask: GradMask | None = None,
    grad_mask_q: GradMask | None = None,
    grad_mask_k: GradMask | None = None,
    grad_mask_v: GradMask | None = None,
    flex_attn_fn: Callable | None = None,
    use_triton: bool | None = None
) -> Tensor:
    # grad_mask controls per-edge gradient flow into q / k / v
    # use_triton = None auto-picks fused triton when supported

    scale = default(scale, size(q, '... [d]') ** -0.5)

    if exists(mask):
        assert mask.dtype == torch.bool, '`mask` must be a boolean allow mask'

    if exists(grad_mask):
        grad_mask_q = default(grad_mask_q, grad_mask)
        grad_mask_k = default(grad_mask_k, grad_mask)
        grad_mask_v = default(grad_mask_v, grad_mask)

    has_grad_mask = any(map(exists, (grad_mask_q, grad_mask_k, grad_mask_v)))

    if not has_grad_mask and exists(flex_attn_fn):
        return flex_attn_fn(q, k, v)

    # gqa

    query_heads = size(q, 'b [h] n d')
    key_heads = size(k, 'b [h] n d')

    if query_heads != key_heads:
        assert query_heads % key_heads == 0, 'query heads must be divisible by key heads'

        groups = query_heads // key_heads
        k = repeat(k, 'b h n d -> b (h g) n d', g = groups)
        v = repeat(v, 'b h n d -> b (h g) n d', g = groups)

    if has_grad_mask:
        triton_available = can_use_triton(q, k, v, mask, grad_mask_q, grad_mask_k, grad_mask_v)

        if use_triton is True:
            assert triton_available, 'triton flash attention does not support this configuration'
        elif use_triton is False:
            triton_available = False

        if triton_available:
            return flash_attn_with_grad_mask(q, k, v, mask, grad_mask_k, grad_mask_v, scale)

        if exists(flex_attn_fn):
            assert exists(mask), 'flex attention with a gradient mask needs a dense mask for the backward'

        return _AttentionWithGradMask.apply(
            q,
            k,
            v,
            mask,
            grad_mask_q,
            grad_mask_k,
            grad_mask_v,
            scale,
            flex_attn_fn
        )

    return _manual_attention(q, k, v, mask, scale)


# pizero masks

def create_pizero_attn_mask(
    prefix_causal_length,
    mask: Tensor, # bool[b n]
    prefix_bidirectional_length = 0,
    discretized_action_length = 0
):
    # causal, bidirectional prefix and actions

    fast_start = prefix_causal_length - discretized_action_length

    def mask_fn(batch_index, head_index, query_index, key_index):
        key_mask = mask[batch_index, key_index]                        # variable length states
        causal_mask = query_index >= key_index                         # causal
        is_prefix = key_index < prefix_bidirectional_length            # bidirectional prefix (vision, etc)

        bidirectional_action_mask = (                                  # bidirectional action mask
            (key_index >= prefix_causal_length) &
            (query_index >= prefix_causal_length)
        )

        action_ignore_fast = (
            (query_index >= prefix_causal_length) &
            (key_index >= fast_start) &
            (key_index < prefix_causal_length)
        )

        return key_mask & (is_prefix | (causal_mask & ~action_ignore_fast) | bidirectional_action_mask)

    return mask_fn


def create_pizero_dense_attn_mask(
    seq_len: int,
    prefix_len: int,
    mask: Tensor | None = None,       # bool[b n]
    ar_mask: Tensor | None = None,    # bool[b n]
    prefix_bidirectional_length = 0,
    discretized_action_length = 0,
    device = None
):
    # dense version of create_pizero_attn_mask, true = attend

    device = default(device, mask.device if exists(mask) else None)

    causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).triu(1)

    if prefix_bidirectional_length > 0:
        causal_mask[:prefix_bidirectional_length, :prefix_bidirectional_length] = False

    if exists(ar_mask):
        cumsum = ar_mask.cumsum(dim = -1)
        q_seg, k_seg = cumsum.unsqueeze(-1), cumsum.unsqueeze(-2)

        causal_mask = (k_seg > q_seg) | (causal_mask & (q_seg == 0) & (k_seg == 0))
        causal_mask = rearrange(causal_mask, 'b i j -> b 1 i j')
    else:
        causal_mask = rearrange(causal_mask, 'i j -> 1 1 i j')

    if exists(mask):
        causal_mask = causal_mask | rearrange(~mask, 'b j -> b 1 1 j')

    if discretized_action_length > 0:
        causal_mask[..., prefix_len:, (prefix_len - discretized_action_length):prefix_len] = True

    return ~causal_mask


def create_pizero_flex_attn_fn(
    batch: int,
    seq_len: int,
    prefix_len: int,
    mask: Tensor, # bool[b n]
    prefix_bidirectional_length = 0,
    discretized_action_length = 0,
    device = None
):
    assert exists(flex_attention) and exists(create_block_mask), 'flex attention is not available'

    block_mask = create_block_mask(
        create_pizero_attn_mask(
            prefix_len,
            mask = mask,
            prefix_bidirectional_length = prefix_bidirectional_length,
            discretized_action_length = discretized_action_length
        ),
        B = batch,
        H = None,
        Q_LEN = seq_len,
        KV_LEN = seq_len,
        device = device,
        _compile = True
    )

    return partial(flex_attention, block_mask = block_mask)


def pizero_attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float,
    prefix_len: int,
    mask: Tensor | None = None,       # bool[b n]
    mem_read_len = 0,
    ar_mask: Tensor | None = None,    # bool[b n]
    prefix_bidirectional_length = 0,
    discretized_action_length = 0,
    knowledge_insulate = False,
    flex_attn_fn: Callable | None = None
) -> Tensor:

    grad_mask = None

    if knowledge_insulate and prefix_len > mem_read_len:
        grad_mask = create_knowledge_insulation_grad_mask(
            seq_len = size(q, '... [n] d'),
            start = mem_read_len,
            end = prefix_len,
            device = q.device
        )

    attn_mask = None

    if not exists(flex_attn_fn) or exists(grad_mask):
        attn_mask = create_pizero_dense_attn_mask(
            seq_len = size(q, '... [n] d'),
            prefix_len = prefix_len,
            mask = mask,
            ar_mask = ar_mask,
            prefix_bidirectional_length = prefix_bidirectional_length,
            discretized_action_length = discretized_action_length,
            device = q.device
        )

    return attend(
        q, k, v,
        scale = scale,
        mask = attn_mask,
        grad_mask_k = grad_mask,
        grad_mask_v = grad_mask,
        flex_attn_fn = flex_attn_fn
    )


if __name__ == '__main__':
    torch.manual_seed(0)

    # 4 state, 3 action tokens
    seq_len, state_len, dim = 7, 4, 8

    q, k, v = (torch.randn(1, 1, seq_len, dim, requires_grad = True) for _ in range(3))
    mask = torch.ones(1, 1, seq_len, seq_len, dtype = torch.bool).tril()

    # action queries cannot backprop into state
    grad_mask = StopGradMask(
        rows = torch.arange(seq_len) >= state_len,
        cols = torch.arange(seq_len) < state_len
    )

    out = attend(q, k, v, mask = mask, grad_mask_k = grad_mask, grad_mask_v = grad_mask)
    out[:, :, state_len:].square().sum().backward()

    print('state k grad:', k.grad[:, :, :state_len].abs().max().item())  # 0.
    print('state v grad:', v.grad[:, :, :state_len].abs().max().item())  # 0.
    print('action q grad:', q.grad[:, :, state_len:].abs().sum().item()) # > 0.
