import asyncio
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms.functional as TF
import tqdm

from pi_zero_pytorch.pi_zero import calc_generalized_advantage_estimate

from .errors import ApiError
from .networks import PI_ZERO_CONFIGS, VALUE_NETWORK_CONFIGS, SmallPiZero, SmallValueNetwork
from .state import AppState
from .storage import get_frame_count, get_video_path
from .store import store

logger = logging.getLogger(__name__)


# training threads


def _broadcast_training_update(state: AppState, loop: asyncio.AbstractEventLoop):
    asyncio.run_coroutine_threadsafe(
        state.manager.broadcast(json.dumps({
            "type": "training_update",
            "state": state.training_state
        })),
        loop
    )


def _stop_training(state: AppState, loop: asyncio.AbstractEventLoop):
    state.training_state["is_training"] = False
    _broadcast_training_update(state, loop)


def _prepare_images(images: torch.Tensor, device: torch.device, target_size) -> torch.Tensor:
    """uint8 (b, c, [views,] h, w) -> normalized float (b, c, *target_size), first view only."""
    images = images.to(device).float().div_(255.0)
    if images.ndim == 5:
        images = images[:, :, 0, :, :]
    if images.shape[-2:] != target_size:
        images = TF.resize(images, target_size, antialias=True)
    return images


def _run_training(state: AppState, loop, loader, num_epochs: int, mock: bool, broadcast_every: int, step_fn):
    """Shared epoch/step loop: runs `step_fn` per batch, tracking and broadcasting progress.

    Mock configs stop after a single gradient step so UI flows stay fast.
    """
    max_steps = 1 if mock else len(loader) * num_epochs
    state.training_state.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": 1 if mock else num_epochs,
        "current_step": 0,
        "total_steps": max_steps,
        "last_loss": 0.0
    })
    _broadcast_training_update(state, loop)

    step = 0
    for epoch in range(num_epochs):
        state.training_state["current_epoch"] = epoch + 1
        for i, batch in enumerate(loader):
            loss = step_fn(batch)
            step += 1
            state.training_state["current_step"] = step
            state.training_state["last_loss"] = float(loss.item())

            if step >= max_steps:
                return
            if i % broadcast_every == 0:
                _broadcast_training_update(state, loop)
        _broadcast_training_update(state, loop)


def train_value_network_thread(state: AppState, config_name: str, loop: asyncio.AbstractEventLoop):
    mock = config_name == "mock"

    if mock:
        dataset = torch.utils.data.TensorDataset(
            torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8),
            torch.randn(1)
        )
    else:
        all_images, all_returns = [], []
        for i in range(len(state.replay_buffer)):
            returns = state.replay_buffer.data['returns'][i]
            valid_mask = ~np.isnan(returns)
            if valid_mask.any():
                all_images.append(torch.from_numpy(state.replay_buffer.data['images'][i][valid_mask]))
                all_returns.append(torch.from_numpy(returns[valid_mask]))

        if not all_images:
            logger.warning("no labelled returns available for value training")
            _stop_training(state, loop)
            return

        dataset = torch.utils.data.TensorDataset(torch.cat(all_images), torch.cat(all_returns))

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    config = VALUE_NETWORK_CONFIGS.get(config_name, VALUE_NETWORK_CONFIGS["small"])
    model = SmallValueNetwork(**config).to(state.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()

    target_size = (config.get('image_size', 224),) * 2

    def step_fn(batch):
        images, returns = batch
        images = _prepare_images(images, state.device, target_size)
        returns = returns.to(state.device)

        _, logits = model(images, return_value_and_logits=True)
        loss = model.to_value.loss_fn(logits, returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    num_epochs = 10
    _run_training(state, loop, loader, num_epochs, mock, broadcast_every=5, step_fn=step_fn)

    state.value_network = model
    _save_value_checkpoint(state, model, config, config_name, num_epochs)
    _stop_training(state, loop)


def _save_value_checkpoint(state: AppState, model, config, config_name: str, num_epochs: int):
    if not state.recap_workspace:
        return

    networks_dir = state.recap_workspace / "value_networks"
    networks_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = networks_dir / f"{config_name}_{timestamp}.pt"

    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "config_name": config_name,
        "epochs": num_epochs,
        "final_loss": state.training_state["last_loss"],
        "timestamp": timestamp
    }, str(model_path))
    logger.info("value network saved to %s", model_path)


def train_policy_network_thread(state: AppState, config_name: str, loop: asyncio.AbstractEventLoop):
    """Advantage-conditioned finetuning on timesteps with binarized advantages."""
    mock = config_name == "mock"
    has_proprio = (state.replay_buffer is not None) and ('proprio' in state.replay_buffer.data)
    proprio_dim = None

    if mock:
        dataset = torch.utils.data.TensorDataset(
            torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8),
            torch.zeros(1, 32, dtype=torch.long),
            torch.randn(1, 32),
            torch.randn(1, 16, 6),
            torch.zeros(1, dtype=torch.long)
        )
    else:
        field_names = ['images', 'text', 'internal', 'actions', 'advantage_ids']
        if has_proprio:
            field_names.append('proprio')

        columns = {name: [] for name in field_names}
        for i in range(len(state.replay_buffer)):
            valid_mask = state.replay_buffer.data['advantage_ids'][i] != -1
            if not valid_mask.any():
                continue
            for name in field_names:
                columns[name].append(torch.from_numpy(state.replay_buffer.data[name][i][valid_mask]))

        if not columns['images']:
            logger.warning("no binarized advantages available for policy finetuning")
            _stop_training(state, loop)
            return

        tensors = [torch.cat(columns[name]) for name in field_names]
        dataset = torch.utils.data.TensorDataset(*tensors)
        if has_proprio:
            proprio_dim = tensors[-1].shape[-1]

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    config = PI_ZERO_CONFIGS.get(config_name, PI_ZERO_CONFIGS["small"]).copy()
    if proprio_dim is not None:
        config["dim_joint_state"] = proprio_dim
        logger.info("policy dim_joint_state set from proprio data: %d", proprio_dim)

    model = SmallPiZero(**config).to(state.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # lower lr for finetuning
    model.train()

    target_size = (config.get('image_size', 32),) * 2

    def step_fn(batch):
        if len(batch) == 6:
            images, text, internal, actions, adv_ids, joint_state = batch
        else:
            images, text, internal, actions, adv_ids = batch
            joint_state = internal

        images = _prepare_images(images, state.device, target_size)
        text, joint_state, actions, adv_ids = (
            t.to(state.device) for t in (text, joint_state, actions, adv_ids)
        )

        output = model(images, text, joint_state, actions, advantage_ids=adv_ids)
        loss = output[0] if isinstance(output, tuple) else output

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    _run_training(state, loop, loader, num_epochs=1, mock=mock, broadcast_every=2, step_fn=step_fn)

    if state.recap_workspace:
        policy_dir = state.recap_workspace / "policy_finetuned"
        policy_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.pizero.state_dict(), str(policy_dir / "actor.pt"))
        logger.info("finetuned policy saved to %s", policy_dir / "actor.pt")

    _stop_training(state, loop)


# value / advantage estimation


async def calculate_episode_value(state: AppState, filename: str, max_t: int = None):
    episode_id = store.resolve(filename)

    if state.value_network is None:
        raise ApiError("Value network not initialized", status_code=409)

    marked_timestep = state.replay_buffer.meta_data['marked_timestep'][episode_id].item()
    if marked_timestep != -1:
        max_t = marked_timestep + 1

    try:
        values = await _calculate_episode_value_internal(state, episode_id, filename, max_t=max_t)
    except Exception as e:
        logger.exception("value calculation failed for %s", filename)
        raise ApiError(str(e), status_code=500)

    return {"status": "ok", "value": values}


async def _calculate_episode_value_internal(state: AppState, episode_id: int, filename: str, max_t: int = None):
    images = state.replay_buffer.data['images'][episode_id]  # (max_timesteps, c, num_views, h, w)

    num_frames = get_frame_count(get_video_path(state, filename))
    calc_to_t = num_frames if max_t is None else min(num_frames, max_t)

    model = state.value_network
    model.eval()
    target_size = (model.image_size, model.image_size)

    values = []
    batch_size = 8
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, calc_to_t, batch_size)):
            batch_images = torch.from_numpy(images[i: min(i + batch_size, calc_to_t)])
            batch_images = _prepare_images(batch_images, state.device, target_size)
            values.extend(model(batch_images).cpu().tolist())

    final_values = torch.full((images.shape[0],), float('nan'))
    final_values[:len(values)] = torch.tensor(values)
    state.replay_buffer.data['value'][episode_id] = final_values.numpy()
    state.replay_buffer.store_meta_datapoint(episode_id, 'invalidated', False)
    state.replay_buffer.flush()
    return values


async def calculate_episode_advantage(state: AppState, filename: str, gamma: float, lam: float):
    episode_id = store.resolve(filename)

    try:
        num_frames = get_frame_count(get_video_path(state, filename))

        marked_timestep = state.replay_buffer.meta_data['marked_timestep'][episode_id].item()
        if marked_timestep != -1:
            num_frames = min(num_frames, marked_timestep + 1)

        # Values are a prerequisite; compute them on demand
        values_np = state.replay_buffer.data['value'][episode_id]
        if np.isnan(values_np[:num_frames]).any():
            logger.info("values missing for %s, calculating first", filename)
            await _calculate_episode_value_internal(state, episode_id, filename, max_t=num_frames)
            values_np = state.replay_buffer.data['value'][episode_id]

        rewards = torch.from_numpy(state.replay_buffer.data['reward'][episode_id][:num_frames])
        values = torch.from_numpy(values_np[:num_frames])
        masks = torch.ones_like(rewards)

        gae_return = calc_generalized_advantage_estimate(
            rewards=rewards,
            values=values,
            masks=masks,
            gamma=gamma,
            lam=lam
        )
        advantages = gae_return.advantages.tolist()

        final_advantages = torch.full((state.replay_buffer.data['advantages'].shape[1],), float('nan'))
        final_advantages[:len(advantages)] = torch.tensor(advantages)
        state.replay_buffer.data['advantages'][episode_id] = final_advantages.numpy()

        # RECAP conditions on binarized advantages; expert segments are always positive,
        # regular steps stay unbinarized until a global cutoff is chosen.
        expert_mask = state.replay_buffer.data['expert_segment'][episode_id][:num_frames]
        adv_ids = state.replay_buffer.data['advantage_ids'][episode_id]
        adv_ids[:num_frames][expert_mask] = 1
        state.replay_buffer.data['advantage_ids'][episode_id] = adv_ids

        state.replay_buffer.store_meta_datapoint(episode_id, 'invalidated', False)
        state.replay_buffer.flush()

        logger.info("calculated %d advantages for %s (episode %d)", len(advantages), filename, episode_id)
        return {"status": "ok", "advantages": advantages, "value": values.tolist(), "advantage_ids": adv_ids.tolist()}
    except Exception as e:
        logger.exception("advantage calculation failed for %s", filename)
        raise ApiError(str(e), status_code=500)
