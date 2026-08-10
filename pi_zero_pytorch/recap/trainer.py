from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict
from .models import SmallValueNetwork, SmallPiZero


def exists(val):
    return val is not None


class RecapValueTrainer:
    def __init__(
        self,
        model: SmallValueNetwork,
        lr: float = 1e-4,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train_epoch(
        self,
        images: torch.Tensor,
        returns: torch.Tensor,
        num_steps: int = 5
    ) -> Dict[str, float]:
        self.model.train()
        images, returns = images.to(self.device), returns.to(self.device)

        total_loss = 0.0
        for _ in range(num_steps):
            self.optimizer.zero_grad()
            pred_values = self.model(images)
            loss = nn.functional.mse_loss(pred_values, returns)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {"loss": total_loss / num_steps}


class RecapPolicyTrainer:
    def __init__(
        self,
        model: SmallPiZero,
        lr: float = 1e-4,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train_step(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        joint_state: torch.Tensor,
        actions: torch.Tensor,
        advantage_ids: torch.Tensor
    ) -> Dict[str, float]:
        self.model.train()
        images = images.to(self.device)
        token_ids = token_ids.to(self.device)
        joint_state = joint_state.to(self.device)
        actions = actions.to(self.device)
        advantage_ids = advantage_ids.to(self.device)

        self.optimizer.zero_grad()
        loss = self.model(
            images=images,
            token_ids=token_ids,
            joint_state=joint_state,
            actions=actions,
            advantage_ids=advantage_ids
        )
        if isinstance(loss, torch.Tensor):
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item()}
        return {"loss": 0.0}
