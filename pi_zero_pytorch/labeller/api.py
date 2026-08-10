import asyncio
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from .models import ActionUpdateRequest, InterventionRequest, LabelRequest
from .networks import SmallValueNetwork
from .recap import (
    get_recap_state,
    recap_collect,
    recap_iterate,
    recap_pretrain,
    recap_specialize,
    simulate_collection,
)
from .state import app_state
from .storage import extract_frames, get_video_path, init_replay_buffer
from .training import (
    calculate_episode_advantage,
    calculate_episode_value,
    train_policy_network_thread,
    train_value_network_thread,
)

router = APIRouter()


@router.get("/api/status")
async def get_status():
    print(f"[RECAP] /api/status check: {app_state.conversion_status}")
    return app_state.conversion_status


@router.get("/api/training/status")
async def get_training_status():
    return app_state.training_state


@router.get("/api/videos")
async def list_videos():
    if app_state.replay_buffer is None:
        return []

    video_list = []
    for filename, ep_id in app_state.video_to_episode.items():
        num_frames = int(app_state.replay_buffer.meta_data['episode_lens'][ep_id].item())
        video_list.append({
            "filename": filename,
            "url": f"/videos/{filename}",
            "frames": num_frames,
            "num_views": int(app_state.num_views)
        })
    return video_list


@router.get("/api/tasks")
async def get_tasks():
    if not app_state.recap_config:
        return []

    tasks = []
    for task_name, config in app_state.recap_config.get('tasks', {}).items():
        tasks.append({
            "id": task_name,
            "name": task_name.replace('_', ' ').title(),
            "max_duration": config.get('max_episode_length', 0),
            "pretrain": config.get('pretrain', {}),
            "finetune": config.get('finetune', {})
        })
    return tasks


@router.post("/api/episode/task")
async def assign_task(req: dict):
    filename = req.get("filename")
    task_id_str = req.get("task_id")

    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = app_state.video_to_episode.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    task_keys = list(app_state.recap_config.get('tasks', {}).keys())
    try:
        task_idx = task_keys.index(task_id_str)
    except ValueError:
        return {"error": f"Task {task_id_str} not found in config"}

    app_state.replay_buffer.store_meta_datapoint(episode_id, 'task_id', torch.tensor(task_idx))
    app_state.replay_buffer.flush()

    return {"status": "ok"}


@router.get("/videos/{filename}")
async def serve_video(filename: str):
    if not app_state.video_dirs:
        return {"error": "Video directories not set"}

    # Handle filename like episode_0 (which should serve episode_0.0.mp4 by default)
    # or handle explicit requests like episode_0.0.mp4
    video_path = get_video_path(app_state, filename)
    if video_path:
        return FileResponse(video_path)

    # Try appending .0.mp4 if it's a grouped name
    video_path = get_video_path(app_state, f"{filename}.0.mp4")
    if video_path:
        return FileResponse(video_path)

    return {"error": "Video not found"}


@router.get("/api/viewpoints")
async def get_viewpoints():
    for vdir in app_state.video_dirs:
        vp_json = vdir / "viewpoints.json"
        if vp_json.exists():
            with open(vp_json) as f:
                return json.load(f)
    return {"viewpoints": {}}


@router.get("/api/video/{filename}/frames")
async def get_video_frames(filename: str):
    # Determine all views for this filename
    # filename is the episode base name
    views = []

    # First try multi-view naming (episode.viewIdx.mp4)
    view_idx = 0
    while True:
        v_name = f"{filename}.{view_idx}.mp4"
        v_path = get_video_path(app_state, v_name)
        if not v_path or not v_path.exists():
            break
        views.append(v_path)
        view_idx += 1

    # If no multi-view files found, try single-view naming
    if not views:
        # Try episode.mp4
        single_path = get_video_path(app_state, f"{filename}.mp4")
        if single_path and single_path.exists():
            views.append(single_path)
        else:
            # Check video_to_path for the episode name directly
            for key, path in app_state.video_to_path.items():
                # Match by episode base name
                if key.replace('.mp4', '') == filename or key == filename:
                    views.append(path)
                    break

    if not views:
        print(f"[FRAMES] Video not found: {filename}")
        print(f"[FRAMES] video_to_path keys: {list(app_state.video_to_path.keys())}")
        return {"error": "Video not found", "frames": []}

    # Extract frames for each view
    all_frames = []
    max_f = 0
    for i, v_path in enumerate(views):
        c_name = v_path.name
        cache_path = app_state.cache_dir / c_name
        extract_frames(v_path, cache_path)
        frames = sorted([f.name for f in cache_path.glob("*.jpg")])
        all_frames.append([f"/cache/{c_name}/{f}" for f in frames])
        max_f = max(max_f, len(frames))

    # Return 2D array [timestep][view]
    result = []
    for t in range(max_f):
        t_views = []
        for v in range(len(all_frames)):
            if t < len(all_frames[v]):
                t_views.append(all_frames[v][t])
            else:
                t_views.append(None)
        result.append(t_views)

    return {"frames": result}


@router.get("/api/video/{filename}/proprio")
async def get_video_proprio(filename: str):
    """Return proprioception data for a video/episode if available."""
    proprio = app_state.video_to_proprio.get(filename)
    if proprio is None:
        return {"proprio": None, "dim_names": [], "num_dims": 0}

    # proprio should be a 2D list [T, D] or 1D [T]
    if isinstance(proprio, list) and len(proprio) > 0:
        if isinstance(proprio[0], list):
            num_dims = len(proprio[0])
        else:
            num_dims = 1
            proprio = [[v] for v in proprio]  # Convert 1D to 2D
    else:
        num_dims = 0

    # Generate dimension names
    dim_names = [f"Joint {i}" for i in range(num_dims)]

    return {"proprio": proprio, "dim_names": dim_names, "num_dims": num_dims}


@router.get("/api/labels")
async def get_all_labels():
    if app_state.replay_buffer is None:
        return {}

    # Return mapping of filename to its current label status
    result = {}
    task_keys = list(app_state.recap_config.get('tasks', {}).keys())

    for filename, episode_id in app_state.video_to_episode.items():
        task_completed = app_state.replay_buffer.meta_data['task_completed'][episode_id].item()
        marked_timestep = app_state.replay_buffer.meta_data['marked_timestep'][episode_id].item()
        task_idx = app_state.replay_buffer.meta_data['task_id'][episode_id].item()

        task_id = task_keys[task_idx] if 0 <= task_idx < len(task_keys) else None

        # Get returns, values, advantages
        returns = app_state.replay_buffer.data['returns'][episode_id].tolist()
        value = app_state.replay_buffer.data['value'][episode_id].tolist()
        advantages = app_state.replay_buffer.data['advantages'][episode_id].tolist()
        advantage_ids = app_state.replay_buffer.data['advantage_ids'][episode_id].tolist()

        # replace nan with None for JSON compliance
        returns = [r if not (isinstance(r, float) and np.isnan(r)) else None for r in returns]
        value = [v if not (isinstance(v, float) and np.isnan(v)) else None for v in value]
        advantages = [a if not (isinstance(a, float) and np.isnan(a)) else None for a in advantages]

        result[filename] = {
            "task_completed": task_completed,
            "marked_timestep": marked_timestep,
            "task_id": task_id,
            "returns": returns,
            "value": value,
            "advantages": advantages,
            "advantage_ids": advantage_ids,
            "is_expert_intervention": app_state.replay_buffer.meta_data['is_expert_intervention'][episode_id].item(),
            "expert_segment": app_state.replay_buffer.data['expert_segment'][episode_id].tolist(),
            "invalidated": app_state.replay_buffer.meta_data['invalidated'][episode_id].item()
        }
    return result


@router.post("/api/label/reset")
async def reset_label(req: dict):
    filename = req.get("filename")
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = app_state.video_to_episode.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    # Clear metadata
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'fail', False)
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'task_completed', -1)
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'marked_timestep', -1)
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'task_id', torch.tensor(-1))
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'is_expert_intervention', False)

    # Reset fields
    app_state.replay_buffer.data['returns'][episode_id] = float('nan')
    app_state.replay_buffer.data['value'][episode_id] = float('nan')
    app_state.replay_buffer.data['advantages'][episode_id] = float('nan')
    app_state.replay_buffer.data['advantage_ids'][episode_id] = -1
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'invalidated', False)
    app_state.replay_buffer.data['expert_segment'][episode_id] = False

    app_state.replay_buffer.flush()
    return {"status": "ok"}


@router.post("/api/label")
async def label_video(req: LabelRequest):
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = app_state.video_to_episode.get(req.filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    if req.success:
        app_state.replay_buffer.store_datapoint(episode_id, req.timestep, 'reward', torch.tensor(0.0))
        app_state.replay_buffer.store_meta_datapoint(episode_id, 'fail', False)
        app_state.replay_buffer.store_meta_datapoint(episode_id, 'task_completed', 1)
        app_state.replay_buffer.store_meta_datapoint(episode_id, 'marked_timestep', req.timestep)
    else:
        # Use custom penalty
        app_state.replay_buffer.store_datapoint(episode_id, req.timestep, 'reward', torch.tensor(req.penalty))
        app_state.replay_buffer.store_meta_datapoint(episode_id, 'fail', True)
        app_state.replay_buffer.store_meta_datapoint(episode_id, 'task_completed', 0)
        app_state.replay_buffer.store_meta_datapoint(episode_id, 'marked_timestep', req.timestep)

    # Reset value and advantages to NaN as they are now stale
    app_state.replay_buffer.data['value'][episode_id] = float('nan')
    app_state.replay_buffer.data['advantages'][episode_id] = float('nan')
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'invalidated', False)

    # Calculate returns
    timesteps = app_state.replay_buffer.data['returns'].shape[1]
    returns = torch.full((timesteps,), float('nan'))

    # Get max duration for normalization
    task_idx = app_state.replay_buffer.meta_data['task_id'][episode_id].item()
    task_keys = list(app_state.recap_config.get('tasks', {}).keys())
    max_duration = 1.0
    if 0 <= task_idx < len(task_keys):
        task_key = task_keys[task_idx]
        max_duration = app_state.recap_config['tasks'][task_key].get('max_episode_length', 1.0)

    for t in range(req.timestep + 1):
        # normalize by max duration
        returns[t] = float(t - req.timestep) / max_duration

    app_state.replay_buffer.data['returns'][episode_id] = returns.numpy()

    app_state.replay_buffer.flush()

    returns_list = returns.tolist()
    returns_list = [r if not np.isnan(r) else None for r in returns_list]

    return {"status": "ok", "returns": returns_list}


@router.post("/api/label/intervention")
async def label_intervention(req: InterventionRequest):
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = app_state.video_to_episode.get(req.filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    # Set meta flag
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'is_expert_intervention', True)

    # Set segment: everything up to current timestep is expert controlled
    expert_mask = app_state.replay_buffer.data['expert_segment'][episode_id]
    expert_mask[:req.timestep + 1] = True
    app_state.replay_buffer.data['expert_segment'][episode_id] = expert_mask

    # Force advantage_ids to 1 (Positive) for the expert segment
    # This is a core RECAP mechanic: expert interventions are "ground truth" positives
    adv_ids = app_state.replay_buffer.data['advantage_ids'][episode_id]
    adv_ids[:req.timestep + 1] = 1
    app_state.replay_buffer.data['advantage_ids'][episode_id] = adv_ids

    app_state.replay_buffer.flush()

    return {
        "status": "ok",
        "is_expert_intervention": True,
        "expert_segment": expert_mask.tolist(),
        "advantage_ids": adv_ids.tolist()
    }


@router.get("/api/video/{filename}/action/{timestep}")
async def get_action(filename: str, timestep: int):
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = app_state.video_to_episode.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    # actions shape: (num_episodes, max_timesteps, horizon, action_dim)
    # Get the action at the specific timestep.
    # For now, we return the first action in the horizon (index 0).
    action = app_state.replay_buffer.data['actions'][episode_id, timestep, 0].tolist()

    return {
        "action": action,
        "horizon": 16,  # Fixed for now as per init_replay_buffer
        "dim": 6
    }


@router.post("/api/video/{filename}/action/{timestep}")
async def update_action(filename: str, timestep: int, req: ActionUpdateRequest):
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = app_state.video_to_episode.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    # Validate action dim
    action_tensor = torch.tensor(req.action)
    if action_tensor.shape[0] != 6:
        return {"error": f"Invalid action dimension. Expected 6, got {action_tensor.shape[0]}"}

    # Update replay buffer. index 0 of the horizon for now.
    app_state.replay_buffer.data['actions'][episode_id, timestep, 0] = action_tensor.numpy()

    # Mark it as expert segment for this timestep.
    expert_mask = app_state.replay_buffer.data['expert_segment'][episode_id]
    expert_mask[timestep] = True
    app_state.replay_buffer.data['expert_segment'][episode_id] = expert_mask

    # Force advantage_id to 1 (Positive) for this edited step
    adv_ids = app_state.replay_buffer.data['advantage_ids'][episode_id]
    adv_ids[timestep] = 1
    app_state.replay_buffer.data['advantage_ids'][episode_id] = adv_ids

    app_state.replay_buffer.flush()

    return {"status": "ok", "action": req.action}


@router.post("/api/returns/calculate")
async def calculate_returns(req: dict):
    filename = req.get("filename")
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = app_state.video_to_episode.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    task_completed = app_state.replay_buffer.meta_data['task_completed'][episode_id].item()
    marked_timestep = app_state.replay_buffer.meta_data['marked_timestep'][episode_id].item()

    if task_completed == -1:
        return {"error": "Video not labelled yet"}

    timesteps = app_state.replay_buffer.data['returns'].shape[1]
    returns = torch.full((timesteps,), float('nan'))

    # Get max duration for normalization
    task_idx = app_state.replay_buffer.meta_data['task_id'][episode_id].item()
    task_keys = list(app_state.recap_config.get('tasks', {}).keys())
    max_duration = 1.0
    if 0 <= task_idx < len(task_keys):
        task_key = task_keys[task_idx]
        max_duration = app_state.recap_config['tasks'][task_key].get('max_episode_length', 1.0)

    for t in range(marked_timestep + 1):
        returns[t] = float(t - marked_timestep) / max_duration

    app_state.replay_buffer.data['returns'][episode_id] = returns.numpy()

    # Reset value and advantages to NaN as they are now stale
    app_state.replay_buffer.data['value'][episode_id] = float('nan')
    app_state.replay_buffer.data['advantages'][episode_id] = float('nan')
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'invalidated', False)

    app_state.replay_buffer.flush()

    returns_list = returns.tolist()
    returns_list = [r if not np.isnan(r) else None for r in returns_list]

    return {"status": "ok", "returns": returns_list}


@router.post("/api/episode/value/calculate")
async def episode_value_calculate(req: dict):
    filename = req.get("filename")
    return await calculate_episode_value(app_state, filename)


@router.post("/api/episode/advantage/calculate")
async def episode_advantage_calculate(req: dict):
    filename = req.get("filename")
    gamma = req.get("gamma", 0.99)
    lam = req.get("lam", 0.95)
    return await calculate_episode_advantage(app_state, filename, gamma, lam)


@router.post("/api/advantage/stats")
async def calculate_global_advantage_stats(req: dict):
    percentile = req.get("percentile", 90)
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    # Get all advantages across all episodes
    all_advs = app_state.replay_buffer.data['advantages']

    # Filter valid ones (not NaN)
    valid_advs = all_advs[~np.isnan(all_advs)]

    if len(valid_advs) == 0:
        return {"error": "No advantages calculated yet"}

    cutoff = np.percentile(valid_advs, percentile)

    return {
        "status": "ok",
        "cutoff": float(cutoff),
        "count": len(valid_advs)
    }


@router.post("/api/advantage/binarize")
async def binarize_advantages(req: dict):
    filename = req.get("filename")
    cutoff = req.get("cutoff")

    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = app_state.video_to_episode.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    advs = app_state.replay_buffer.data['advantages'][episode_id]

    # Calculate binarized IDs: 1 if >= cutoff, 0 if < cutoff, -1 if NaN
    adv_ids = np.full(advs.shape, -1, dtype=int)
    valid_mask = ~np.isnan(advs)
    adv_ids[valid_mask] = (advs[valid_mask] >= cutoff).astype(int)

    app_state.replay_buffer.data['advantage_ids'][episode_id] = adv_ids
    app_state.replay_buffer.flush()

    print(f"[RECAP] Binarized advantages for {filename} (ID: {episode_id}) with cutoff {cutoff:.4f}. Pos: {np.sum(adv_ids == 1)}, Neg: {np.sum(adv_ids == 0)}")

    return {"status": "ok", "advantage_ids": adv_ids.tolist()}


@router.post("/api/episode/invalidate")
async def invalidate_episode_timesteps(req: dict):
    filename = req.get("filename")
    cutoff = req.get("cutoff", 0.0)

    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    episode_id = app_state.video_to_episode.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    advs = app_state.replay_buffer.data['advantages'][episode_id]

    # Calculate invalidated mask: True if advantage <= cutoff and not NaN
    valid_mask = ~np.isnan(advs)
    # If any advantage is below cutoff, invalidate the whole episode (RECAP policy)
    is_invalid = bool((advs[valid_mask] <= cutoff).any())
    app_state.replay_buffer.store_meta_datapoint(episode_id, 'invalidated', is_invalid)
    app_state.replay_buffer.flush()

    return {"status": "ok", "invalidated": is_invalid}


@router.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = Path(__file__).parent / "web_ui" / "index.html"
    return HTMLResponse(content=index_path.read_text(), status_code=200)


@router.get("/api/export")
async def export_labels():
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    # Create export directory in cache if it doesn't exist
    export_dir = app_state.cache_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    zip_filename = f"labels_export_{timestamp}.zip"
    zip_path = export_dir / zip_filename

    # Extract ALL data fields
    data_dict = {name: app_state.replay_buffer.data[name][:] for name in app_state.replay_buffer.data.keys()}

    # Extract ALL meta data fields
    meta_data_dict = {name: app_state.replay_buffer.meta_data[name][:] for name in app_state.replay_buffer.meta_data.keys()}

    # Metadata mapping
    metadata = {
        "video_to_index": app_state.video_to_episode,
        "config": app_state.recap_config,
        "timestamp": timestamp
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Save data.npz
        data_buffer = io.BytesIO()
        np.savez(data_buffer, **data_dict)
        zip_file.writestr("data.npz", data_buffer.getvalue())

        # Save meta_data.npz
        meta_data_buffer = io.BytesIO()
        np.savez(meta_data_buffer, **meta_data_dict)
        zip_file.writestr("meta_data.npz", meta_data_buffer.getvalue())

        # Save metadata.json
        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))

    return FileResponse(
        path=str(zip_path),
        filename=zip_filename,
        media_type="application/zip"
    )


@router.get("/api/recap/state")
async def recap_state():
    return get_recap_state(app_state)


@router.post("/api/recap/pretrain")
async def recap_pretrain_route():
    return recap_pretrain(app_state)


@router.post("/api/recap/specialize")
async def recap_specialize_route(req: dict):
    return recap_specialize(app_state, req.get("task_name"))


@router.post("/api/recap/collect")
async def recap_collect_route(req: dict):
    return recap_collect(app_state, req.get("task_name"), req.get("iter_id", 0))


@router.post("/api/recap/iterate")
async def recap_iterate_route(req: dict):
    return recap_iterate(app_state, req.get("task_name"), req.get("iter_id", 0))


@router.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    await app_state.manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Just keep alive
    except WebSocketDisconnect:
        app_state.manager.disconnect(websocket)


@router.post("/api/recap/finetune")
async def start_policy_finetune(req: dict):
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    if app_state.training_state["is_training"]:
        return {"error": "Training already in progress"}

    config_name = req.get("config", "mock")  # use mock for speed in e2e
    print(f"[RECAP] start_policy_finetune called with config: {config_name}")
    loop = asyncio.get_event_loop()
    threading.Thread(target=train_policy_network_thread, args=(app_state, config_name, loop), daemon=True).start()

    return {"status": "ok"}


@router.post("/api/value/train")
async def start_training(req: dict):
    if app_state.replay_buffer is None:
        return {"error": "ReplayBuffer not initialized"}

    if app_state.training_state["is_training"]:
        return {"error": "Training already in progress"}

    config_name = req.get("config", "small")
    loop = asyncio.get_event_loop()
    threading.Thread(target=train_value_network_thread, args=(app_state, config_name, loop), daemon=True).start()

    return {"status": "ok"}


@router.post("/api/recap/simulate_collection")
async def simulate_collection_api(req: dict):
    """Simulates collecting a new batch of data for a task/iteration."""
    return simulate_collection(app_state, req.get("task_name"), req.get("iter_id", 0))


@router.post("/api/recap/load_data")
async def recap_load_data(req: dict):
    """Mounts a specific data folder to view in the labeller."""
    task_name = req.get("task_name")
    iter_id = req.get("iter_id")
    data_id = req.get("data_id")
    is_pretrained = req.get("is_pretrained", False)

    if is_pretrained:
        target_dir = app_state.recap_workspace / "pretrained_data"
    else:
        if not all([task_name, iter_id is not None, data_id]):
            return {"error": "Missing required parameters"}
        target_dir = app_state.recap_workspace / task_name / str(iter_id) / data_id

    if not target_dir.exists():
        return {"error": f"Data directory {target_dir} does not exist"}

    # Reset current state
    app_state.video_dirs = [target_dir]
    app_state.replay_buffer = None
    app_state.video_to_episode = {}

    # Start conversion in background
    app_state.conversion_status["is_converting"] = True
    app_state.conversion_status["progress"] = 0
    threading.Thread(target=init_replay_buffer, args=(app_state, app_state.video_dirs), daemon=True).start()

    return {"status": "ok", "video_dir": str(target_dir)}


@router.get("/api/value/networks/list")
async def list_value_networks():
    if not app_state.recap_workspace:
        return []

    networks_dir = app_state.recap_workspace / "value_networks"
    if not networks_dir.exists():
        return []

    networks = []
    for f in networks_dir.glob("*.pt"):
        try:
            checkpoint = torch.load(str(f), map_location='cpu', weights_only=False)
            networks.append({
                "filename": f.name,
                "config_name": checkpoint.get("config_name", "unknown"),
                "epochs": checkpoint.get("epochs", 0),
                "final_loss": checkpoint.get("final_loss", 0.0),
                "timestamp": checkpoint.get("timestamp", "")
            })
        except Exception as e:
            print(f"Error loading checkpoint {f}: {e}")

    # sort by timestamp descending
    networks.sort(key=lambda x: x['timestamp'], reverse=True)
    return networks


@router.post("/api/value/networks/load")
async def load_value_network(req: dict):
    filename = req.get("filename")
    if not filename or not app_state.recap_workspace:
        return {"error": "Missing filename or RECAP_WORKSPACE"}

    model_path = app_state.recap_workspace / "value_networks" / filename
    if not model_path.exists():
        return {"error": "Model file not found"}

    try:
        checkpoint = torch.load(str(model_path), map_location=app_state.device, weights_only=False)
        config = checkpoint["config"]
        model = SmallValueNetwork(**config).to(app_state.device)
        model.load_state_dict(checkpoint["state_dict"])
        app_state.value_network = model
        return {"status": "ok", "config_name": checkpoint.get("config_name")}
    except Exception as e:
        return {"error": str(e)}
