import asyncio
import json
import traceback
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms.functional as TF
import tqdm

from pi_zero_pytorch.pi_zero import calc_generalized_advantage_estimate

from .networks import PI_ZERO_CONFIGS, VALUE_NETWORK_CONFIGS, SmallPiZero, SmallValueNetwork
from .state import AppState
from .storage import get_frame_count, get_video_path


def _broadcast_training_update(state: AppState, loop: asyncio.AbstractEventLoop):
    asyncio.run_coroutine_threadsafe(
        state.manager.broadcast(json.dumps({
            "type": "training_update",
            "state": state.training_state
        })),
        loop
    )


def train_value_network_thread(state: AppState, config_name: str, loop: asyncio.AbstractEventLoop):
    # 1. Prepare data
    if config_name == "mock":
        # Extremely fast mock data
        images = torch.randn(1, 3, 32, 32)
        returns = torch.randn(1)
        dataset = torch.utils.data.TensorDataset(images, returns)
    else:
        all_images = []
        all_returns = []
        for i in range(len(state.replay_buffer)):
            returns = state.replay_buffer.data['returns'][i]
            valid_mask = ~np.isnan(returns)
            if valid_mask.any():
                images = state.replay_buffer.data['images'][i][valid_mask]
                all_images.append(torch.from_numpy(images))
                all_returns.append(torch.from_numpy(returns[valid_mask]))

        if not all_images:
            state.training_state["is_training"] = False
            _broadcast_training_update(state, loop)
            return

        dataset = torch.utils.data.TensorDataset(torch.cat(all_images), torch.cat(all_returns))
    batch_size = 16
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize model
    config = VALUE_NETWORK_CONFIGS.get(config_name, VALUE_NETWORK_CONFIGS["small"])
    model = SmallValueNetwork(**config).to(state.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()

    num_epochs = 10
    total_steps = len(loader) * num_epochs if config_name != "mock" else 1
    state.training_state.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": num_epochs if config_name != "mock" else 1,
        "current_step": 0,
        "total_steps": total_steps,
        "last_loss": 0.0
    })
    _broadcast_training_update(state, loop)

    # 3. Training loop
    target_size = (config.get('image_size', 224), config.get('image_size', 224))

    for epoch in range(num_epochs):
        state.training_state["current_epoch"] = epoch + 1
        for i, (images, returns) in enumerate(loader):
            images = images.to(state.device)
            if images.ndim == 5:
                images = images[:, :, 0, :, :]
            if images.shape[-2:] != target_size:
                images = TF.resize(images, target_size, antialias=True)

            returns = returns.to(state.device)

            values, logits = model(images, return_value_and_logits=True)
            loss = model.to_value.loss_fn(logits, returns).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state.training_state["current_step"] += 1
            state.training_state["last_loss"] = float(loss.item())

            # For mock config, we only do one gradient step total
            if config_name == "mock":
                print("Mock training: finishing after one gradient step.")
                break

            if i % 5 == 0:
                _broadcast_training_update(state, loop)

        if config_name == "mock":
            break

        _broadcast_training_update(state, loop)

    # 4. Finalize
    state.value_network = model

    # Save the model
    if state.recap_workspace:
        networks_dir = state.recap_workspace / "value_networks"
        networks_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{config_name}_{timestamp}.pt"
        model_path = networks_dir / model_filename

        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
            "config_name": config_name,
            "epochs": num_epochs,
            "final_loss": state.training_state["last_loss"],
            "timestamp": timestamp
        }, str(model_path))
        print(f"Value network saved to {model_path}")

    state.training_state["is_training"] = False
    _broadcast_training_update(state, loop)


def train_policy_network_thread(state: AppState, config_name: str, loop: asyncio.AbstractEventLoop):
    has_proprio = (state.replay_buffer is not None) and ('proprio' in state.replay_buffer.data)
    all_proprio = []

    # 1. Prepare data - conditioned on binarized advantages (advantage_ids)
    if config_name == "mock":
        # Extremely fast mock data
        images = torch.randn(1, 3, 32, 32)
        text = torch.zeros(1, 32, dtype=torch.long)
        internal = torch.randn(1, 32)
        actions = torch.randn(1, 16, 6)
        adv_ids = torch.zeros(1, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(images, text, internal, actions, adv_ids)
    else:
        all_images = []
        all_text = []
        all_internal = []
        all_actions = []
        all_advantage_ids = []

        for i in range(len(state.replay_buffer)):
            advantage_ids = state.replay_buffer.data['advantage_ids'][i]
            valid_mask = advantage_ids != -1
            if valid_mask.any():
                all_images.append(torch.from_numpy(state.replay_buffer.data['images'][i][valid_mask]))
                all_text.append(torch.from_numpy(state.replay_buffer.data['text'][i][valid_mask]))
                all_internal.append(torch.from_numpy(state.replay_buffer.data['internal'][i][valid_mask]))
                all_actions.append(torch.from_numpy(state.replay_buffer.data['actions'][i][valid_mask]))
                all_advantage_ids.append(torch.from_numpy(advantage_ids[valid_mask]))
                if has_proprio:
                    all_proprio.append(torch.from_numpy(state.replay_buffer.data['proprio'][i][valid_mask]))

        if not all_images:
            print("No valid data for policy training")
            state.training_state["is_training"] = False
            _broadcast_training_update(state, loop)
            return

        tensors = [
            torch.cat(all_images),
            torch.cat(all_text),
            torch.cat(all_internal),
            torch.cat(all_actions),
            torch.cat(all_advantage_ids)
        ]

        if has_proprio:
            tensors.append(torch.cat(all_proprio))

        dataset = torch.utils.data.TensorDataset(*tensors)

    batch_size = 4
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize model
    config = PI_ZERO_CONFIGS.get(config_name, PI_ZERO_CONFIGS["small"]).copy()

    # If we have proprio data in the dataset, update dim_joint_state to match
    if has_proprio and len(all_proprio) > 0:
        actual_proprio_dim = all_proprio[0].shape[-1]
        config["dim_joint_state"] = actual_proprio_dim
        print(f"[RECAP] Using actual proprio_dim for policy: {actual_proprio_dim}")
    else:
        print(f"[RECAP] Using default dim_joint_state: {config.get('dim_joint_state', 32)}")

    model = SmallPiZero(**config).to(state.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # lower lr for finetuning
    model.train()

    num_epochs = 1
    total_steps = len(loader) * num_epochs if config_name != "mock" else 1
    state.training_state.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": num_epochs if config_name != "mock" else 1,
        "current_step": 0,
        "total_steps": total_steps,
        "last_loss": 0.0
    })
    _broadcast_training_update(state, loop)

    # 3. Training loop
    target_size = (config.get('image_size', 32), config.get('image_size', 32))

    print(f"Starting policy fine-tuning for {num_epochs} epoch...")
    for epoch in range(num_epochs):
        state.training_state["current_epoch"] = epoch + 1
        for i, batch in enumerate(loader):
            if len(batch) == 6:
                images, text, internal, actions, adv_ids, proprio = batch
                joint_state = proprio
            else:
                images, text, internal, actions, adv_ids = batch
                joint_state = internal

            images = images.to(state.device)
            if images.ndim == 5:
                images = images[:, :, 0, :, :]
            if images.shape[-2:] != target_size:
                images = TF.resize(images, target_size, antialias=True)

            text = text.to(state.device)
            joint_state = joint_state.to(state.device)
            actions = actions.to(state.device)
            adv_ids = adv_ids.to(state.device)

            # Conditioned on advantage_ids
            output = model(images, text, joint_state, actions, advantage_ids=adv_ids)
            loss = output[0] if isinstance(output, tuple) else output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state.training_state["current_step"] += 1
            state.training_state["last_loss"] = float(loss.item())

            # For mock config, we only do one gradient step total
            if config_name == "mock":
                print("Mock finetuning: finishing after one gradient step.")
                break

            if i % 2 == 0:
                _broadcast_training_update(state, loop)

        if config_name == "mock":
            break

        _broadcast_training_update(state, loop)

    # 4. Finalize
    if state.recap_workspace:
        policy_dir = state.recap_workspace / "policy_finetuned"
        policy_dir.mkdir(parents=True, exist_ok=True)

        model_path = policy_dir / "actor.pt"
        torch.save(model.pizero.state_dict(), str(model_path))
        print(f"Finetuned policy saved to {model_path}")

    state.training_state["is_training"] = False
    _broadcast_training_update(state, loop)


async def calculate_episode_value(state: AppState, filename: str, max_t: int = None):
    if state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = state.video_to_episode.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    if state.value_network is None:
        return {"error": "Value network not initialized"}

    try:
        # Check for marked_timestep
        marked_timestep = state.replay_buffer.meta_data['marked_timestep'][episode_id].item()
        if marked_timestep != -1:
            max_t = marked_timestep + 1

        values = await _calculate_episode_value_internal(state, episode_id, filename, max_t=max_t)
        return {"status": "ok", "value": values}
    except Exception as e:
        return {"error": str(e)}


async def _calculate_episode_value_internal(state: AppState, episode_id: int, filename: str, max_t: int = None):
    # Get images for this episode
    images = state.replay_buffer.data['images'][episode_id]  # (max_timesteps, c, 1, h, w)

    video_path = get_video_path(state, filename)
    num_frames = get_frame_count(video_path)

    calc_to_t = num_frames
    if max_t is not None:
        calc_to_t = min(num_frames, max_t)

    values = []
    state.value_network.eval()

    batch_size = 8
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, calc_to_t, batch_size)):
            batch_images = images[i: min(i + batch_size, calc_to_t)]
            batch_images = torch.from_numpy(batch_images).to(state.device)
            # Stored as (t, c, num_views, h, w); use first view for scoring
            if batch_images.ndim == 5:
                batch_images = batch_images[:, :, 0, :, :]

            # Use model's expected image size
            target_size = (state.value_network.image_size, state.value_network.image_size)
            if batch_images.shape[-2:] != target_size:
                batch_images = TF.resize(batch_images, target_size, antialias=True)

            batch_values = state.value_network(batch_images)
            values.extend(batch_values.cpu().tolist())

    # Store values back to replay buffer
    final_values = torch.full((images.shape[0],), float('nan'))
    final_values[:len(values)] = torch.tensor(values)
    state.replay_buffer.data['value'][episode_id] = final_values.numpy()
    state.replay_buffer.store_meta_datapoint(episode_id, 'invalidated', False)
    state.replay_buffer.flush()
    return values


async def calculate_episode_advantage(state: AppState, filename: str, gamma: float, lam: float):
    if state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = state.video_to_episode.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    try:
        # Get actual frame count
        video_path = get_video_path(state, filename)
        num_frames = get_frame_count(video_path)

        marked_timestep = state.replay_buffer.meta_data['marked_timestep'][episode_id].item()
        if marked_timestep != -1:
            num_frames = min(num_frames, marked_timestep + 1)

        # Check if values exist, otherwise calculate them
        values_np = state.replay_buffer.data['value'][episode_id]
        if np.isnan(values_np[:num_frames]).any():
            print(f"Values not found for {filename}, calculating first...")
            await _calculate_episode_value_internal(state, episode_id, filename, max_t=num_frames)
            values_np = state.replay_buffer.data['value'][episode_id]

        # Prepare inputs for GAE
        rewards = torch.from_numpy(state.replay_buffer.data['reward'][episode_id][:num_frames])
        values = torch.from_numpy(values_np[:num_frames])
        masks = torch.ones_like(rewards)  # Assume all frames are valid for now

        # Calculate GAE
        gae_return = calc_generalized_advantage_estimate(
            rewards=rewards,
            values=values,
            masks=masks,
            gamma=gamma,
            lam=lam
        )

        advantages = gae_return.advantages.tolist()

        # Store advantages back
        final_advantages = torch.full((state.replay_buffer.data['advantages'].shape[1],), float('nan'))
        final_advantages[:len(advantages)] = torch.tensor(advantages)
        state.replay_buffer.data['advantages'][episode_id] = final_advantages.numpy()
        state.replay_buffer.flush()

        print(f"[RECAP] Calculated advantages for {filename} (ID: {episode_id}). Count: {len(advantages)}")
        valid_advs = torch.tensor(advantages)[~torch.isnan(torch.tensor(advantages))]
        if len(valid_advs) > 0:
            print(f"[RECAP] Advantages - Min: {valid_advs.min().item():.4f}, Max: {valid_advs.max().item():.4f}, Mean: {valid_advs.mean().item():.4f}")

        # RECAP requires binarized advantages in the buffer.
        # If it's an expert segment, we MUST set it to 1.
        expert_mask = state.replay_buffer.data['expert_segment'][episode_id][:num_frames]
        adv_ids = state.replay_buffer.data['advantage_ids'][episode_id]

        # We don't have a cutoff here, so we don't binarize regular steps yet.
        # But we DO ensure expert steps are marked.
        adv_ids[:num_frames][expert_mask] = 1
        state.replay_buffer.data['advantage_ids'][episode_id] = adv_ids

        state.replay_buffer.store_meta_datapoint(episode_id, 'invalidated', False)
        state.replay_buffer.flush()

        return {"status": "ok", "advantages": advantages, "value": values.tolist(), "advantage_ids": adv_ids.tolist()}
    except Exception as e:
        traceback.print_exc()
        print(f"Error calculating advantage for {filename}: {str(e)}")
        return {"error": str(e)}
