from __future__ import annotations

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else (d() if callable(d) else d)


def calculate_returns_vectorized(
    marked_timestep: int,
    episode_length: int,
    max_duration: float = 100.0,
    rewards: Optional[np.ndarray] = None,
    gamma: float = 0.99
) -> np.ndarray:
    returns = np.full((episode_length,), np.nan, dtype=np.float32)

    if exists(rewards) and len(rewards) > 0:
        running_return = 0.0
        for t in reversed(range(min(len(rewards), episode_length))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        return returns

    if marked_timestep >= 0:
        max_t = min(marked_timestep + 1, episode_length)
        steps = np.arange(max_t, dtype=np.float32)
        returns[:max_t] = (steps - float(marked_timestep)) / max_duration

    return returns


def calculate_gae_vectorized(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    masks: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    masks = default(masks, lambda: torch.ones_like(rewards))
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
        delta = rewards[t] + gamma * next_value * masks[t] - values[t]
        advantages[t] = last_gae = delta + gamma * lam * masks[t] * last_gae

    return advantages, advantages + values


def calculate_advantage_stats_vectorized(
    advantages_list: List[np.ndarray],
    quantile: float = 0.5
) -> Dict[str, float]:
    valid_advs = [adv[~np.isnan(adv)] for adv in advantages_list if exists(adv) and len(adv) > 0 and (~np.isnan(adv)).any()]
    if not valid_advs:
        raise ValueError("No valid advantage data available.")

    all_advs = np.concatenate(valid_advs)
    return {
        "cutoff": float(np.quantile(all_advs, quantile)),
        "count": int(len(all_advs)),
        "mean": float(np.mean(all_advs)),
        "std": float(np.std(all_advs))
    }


def binarize_advantages_vectorized(
    advantages: np.ndarray,
    cutoff: float = 0.0,
    expert_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    adv_ids = np.full(advantages.shape, -1, dtype=np.int32)
    valid_mask = ~np.isnan(advantages)
    adv_ids[valid_mask] = np.where(advantages[valid_mask] >= cutoff, 1, 0)

    if exists(expert_mask):
        adv_ids[expert_mask] = 1

    return adv_ids
