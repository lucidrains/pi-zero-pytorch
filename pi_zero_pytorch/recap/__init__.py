from __future__ import annotations

"""
RECAP Framework for Vision-Language-Action (VLA) and Vision-Action Models (VAM).
Provides modular environment rollouts, vectorized advantage estimation,
value network training, and advantage-conditioned policy fine-tuning.
"""

from .env import BaseRecapEnv, GymRecapEnv
from .ops import (
    calculate_returns_vectorized,
    calculate_gae_vectorized,
    binarize_advantages_vectorized,
    calculate_advantage_stats_vectorized
)
from .models import SmallValueNetwork, SmallPiZero
from .trainer import RecapValueTrainer, RecapPolicyTrainer
from .engine import RecapSimEngine

__all__ = [
    "BaseRecapEnv",
    "GymRecapEnv",
    "calculate_returns_vectorized",
    "calculate_gae_vectorized",
    "binarize_advantages_vectorized",
    "calculate_advantage_stats_vectorized",
    "SmallValueNetwork",
    "SmallPiZero",
    "RecapValueTrainer",
    "RecapPolicyTrainer",
    "RecapSimEngine"
]
