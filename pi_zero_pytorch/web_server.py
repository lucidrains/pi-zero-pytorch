import os
import cv2
import click
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
import torch
import torchvision.transforms.functional as TF
import numpy as np
import shutil
from typing import List
from memmap_replay_buffer import ReplayBuffer
from pi_zero_pytorch.pi_zero import calc_generalized_advantage_estimate, SigLIP, BinnedValueLayer
import json
import tqdm
import threading

CONVERSION_STATUS = {
    "is_converting": False,
    "progress": 0,
    "total": 0,
    "current_video": ""
}

recap_config_path = Path(__file__).parent / "recap_config.json"
RECAP_CONFIG = {}
if recap_config_path.exists():
    with open(recap_config_path) as f:
        RECAP_CONFIG = json.load(f)

# small value network for introspecting on reward/returns in web server

class SmallValueNetwork(torch.nn.Module):
    def __init__(
        self,
        image_size = 224,
        patch_size = 14,
        dim = 256,
        depth = 4,
        heads = 8,
        min_value = -1.,
        max_value = 0.,
        num_bins = 201
    ):
        super().__init__()

        self.siglip = SigLIP(
            image_size = image_size,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            heads = heads
        )

        self.to_value = BinnedValueLayer(
            dim = dim,
            min_value = min_value,
            max_value = max_value,
            num_bins = num_bins
        )

    def forward(self, images):
        embeds = self.siglip(images)
        pooled = embeds.mean(dim = 1)
        return self.to_value(pooled)

app = FastAPI()

# will be set via CLI
VIDEO_DIR = Path(".")
CACHE_DIR = Path(".cache/frames")
REPLAY_BUFFER = None
VIDEO_TO_EPISODE = {}
VALUE_NETWORK = None
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class LabelRequest(BaseModel):
    filename: str
    timestep: int
    success: bool
    penalty: float = -50.0

class VideoInfo(BaseModel):
    filename: str
    frames: int
    url: str

def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def extract_frames(video_path: Path, cache_path: Path):
    if cache_path.exists():
        return
    
    cache_path.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # save every frame as jpg
        cv2.imwrite(str(cache_path / f"frame_{count:04d}.jpg"), frame)
        count += 1
    cap.release()

def init_replay_buffer(video_dir: Path):
    global REPLAY_BUFFER, VIDEO_TO_EPISODE, CONVERSION_STATUS
    
    CONVERSION_STATUS["is_converting"] = True
    CONVERSION_STATUS["progress"] = 0
    
    tmp_buffer_dir = Path("tmp/replay_buffer")
    if tmp_buffer_dir.exists():
        shutil.rmtree(tmp_buffer_dir)
    tmp_buffer_dir.mkdir(parents=True, exist_ok = True)

    # Get all videos
    extensions = {".mp4", ".webm", ".avi", ".mov"}
    video_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in extensions])
    
    if not video_files:
        return

    # For simplicity, we need to know the image size and max frames
    # Let's peek at the first video
    cap = cv2.VideoCapture(str(video_files[0]))
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
    h, w, c = frame.shape
    max_frames = 0
    for vf in video_files:
        max_frames = max(max_frames, get_frame_count(vf))
    cap.release()

    REPLAY_BUFFER = ReplayBuffer(
        str(tmp_buffer_dir),
        max_episodes = len(video_files),
        max_timesteps = max_frames,
        meta_fields = dict(
            task_id        = ('int', (), -1),
            fail           = 'bool',
            task_completed = ('int', (), -1),
            marked_timestep = ('int', (), -1),
            invalidated    = 'bool',
            recap_step     = ('int', (), -1)
        ),
        fields = dict(
            images = ('float', (c, 1, h, w)), # matching (c, num_images, h, w) pattern
            reward = 'float',
            returns = ('float', (), float('nan')),
            value = ('float', (), float('nan')),
            advantages = ('float', (), float('nan')),
            advantage_ids = ('int', (), -1)
        )
    )

    print(f"Initializing ReplayBuffer from {len(video_files)} videos...")
    CONVERSION_STATUS["total"] = len(video_files)

    for i, video_path in enumerate(video_files):
        CONVERSION_STATUS["current_video"] = video_path.name
        CONVERSION_STATUS["progress"] = i
        
        VIDEO_TO_EPISODE[video_path.name] = i
        cap = cv2.VideoCapture(str(video_path))
        
        # we'll assume a dummy task_id for now
        with REPLAY_BUFFER.one_episode(task_id = torch.tensor(-1)):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to float and [C, 1, H, W]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(1) # Add num_images dim
                REPLAY_BUFFER.store(images = frame_tensor, reward = 0.0)
        cap.release()
    
    CONVERSION_STATUS["is_converting"] = False
    CONVERSION_STATUS["progress"] = len(video_files)
    print("ReplayBuffer initialization complete.")

@app.get("/api/status")
async def get_status():
    return CONVERSION_STATUS

@app.get("/api/tasks")
async def get_tasks():
    if not RECAP_CONFIG:
        return []
        
    tasks = []
    for task_name, config in RECAP_CONFIG.get('tasks', {}).items():
        tasks.append({
            "id": task_name,
            "name": task_name.replace('_', ' ').title(),
            "max_duration": config.get('max_episode_length', 0),
            "pretrain": config.get('pretrain', {}),
            "finetune": config.get('finetune', {})
        })
    return tasks

@app.post("/api/episode/task")
async def assign_task(req: dict):
    filename = req.get("filename")
    task_id_str = req.get("task_id")
    
    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    episode_id = VIDEO_TO_EPISODE.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    # Task IDs are indices in the recap config task keys
    task_keys = list(RECAP_CONFIG.get('tasks', {}).keys())
    try:
        task_idx = task_keys.index(task_id_str)
    except ValueError:
        return {"error": f"Task {task_id_str} not found in config"}

    REPLAY_BUFFER.store_meta_datapoint(episode_id, 'task_id', torch.tensor(task_idx))
    REPLAY_BUFFER.flush()
    
    return {"status": "ok"}

@app.get("/api/videos", response_model=List[VideoInfo])
async def list_videos():
    videos = []
    # common video extensions
    extensions = {".mp4", ".webm", ".avi", ".mov"}
    
    if not VIDEO_DIR.exists():
        return []

    for file in VIDEO_DIR.iterdir():
        if file.suffix.lower() in extensions:
            frames = get_frame_count(file)
            videos.append(VideoInfo(
                filename=file.name,
                frames=frames,
                url=f"/videos/{file.name}"
            ))
    
    # sort by filename
    videos.sort(key=lambda x: x.filename)
    return videos

@app.get("/api/video/{filename}/frames")
async def get_video_frames(filename: str):
    video_path = VIDEO_DIR / filename
    if not video_path.exists():
        return {"error": "Video not found"}
    
    cache_path = CACHE_DIR / filename
    extract_frames(video_path, cache_path)
    
    frames = sorted([f.name for f in cache_path.glob("*.jpg")])
    return {"frames": [f"/cache/{filename}/{f}" for f in frames]}

@app.get("/api/labels")
async def get_all_labels():
    if REPLAY_BUFFER is None:
        return {}
    
    # Return mapping of filename to its current label status
    result = {}
    task_keys = list(RECAP_CONFIG.get('tasks', {}).keys())
    
    for filename, episode_id in VIDEO_TO_EPISODE.items():
        task_completed = REPLAY_BUFFER.meta_data['task_completed'][episode_id].item()
        marked_timestep = REPLAY_BUFFER.meta_data['marked_timestep'][episode_id].item()
        task_idx = REPLAY_BUFFER.meta_data['task_id'][episode_id].item()
        
        task_id = task_keys[task_idx] if 0 <= task_idx < len(task_keys) else None
        
        if task_completed != -1:
            # Also get returns if they exist
            returns = REPLAY_BUFFER.data['returns'][episode_id].tolist()
            # replace nan with None for JSON compliance
            returns = [r if not np.isnan(r) else None for r in returns]
            
            result[filename] = {
                "task_completed": task_completed,
                "marked_timestep": marked_timestep,
                "task_id": task_id,
                "returns": returns
            }
        elif task_id:
            result[filename] = {
                "task_completed": -1,
                "marked_timestep": -1,
                "task_id": task_id,
                "returns": [],
                "value": [],
                "advantages": [],
                "advantage_ids": []
            }

    # Add value/advantage data if it exists
    for filename, episode_id in VIDEO_TO_EPISODE.items():
        if filename in result:
            value = REPLAY_BUFFER.data['value'][episode_id].tolist()
            adv = REPLAY_BUFFER.data['advantages'][episode_id].tolist()
            adv_ids = REPLAY_BUFFER.data['advantage_ids'][episode_id].tolist()
            # replace nan with None for JSON compliance
            value = [v if not np.isnan(v) else None for v in value]
            adv = [a if not np.isnan(a) else None for a in adv]
            result[filename]["value"] = value
            result[filename]["advantages"] = adv
            result[filename]["advantage_ids"] = adv_ids

    return result

@app.post("/api/label/reset")
async def reset_label(req: dict):
    filename = req.get("filename")
    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    episode_id = VIDEO_TO_EPISODE.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    # Clear metadata
    REPLAY_BUFFER.store_meta_datapoint(episode_id, 'fail', False)
    REPLAY_BUFFER.store_meta_datapoint(episode_id, 'task_completed', -1)
    REPLAY_BUFFER.store_meta_datapoint(episode_id, 'marked_timestep', -1)
    REPLAY_BUFFER.store_meta_datapoint(episode_id, 'task_id', torch.tensor(-1))
    
    # Reset returns, value, and advantages to NaN
    REPLAY_BUFFER.data['returns'][episode_id] = float('nan')
    REPLAY_BUFFER.data['value'][episode_id] = float('nan')
    REPLAY_BUFFER.data['advantages'][episode_id] = float('nan')
    
    REPLAY_BUFFER.flush()
    return {"status": "ok"}

@app.post("/api/label")
async def label_video(req: LabelRequest):
    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    episode_id = VIDEO_TO_EPISODE.get(req.filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    if req.success:
        REPLAY_BUFFER.store_datapoint(episode_id, req.timestep, 'reward', torch.tensor(0.0))
        REPLAY_BUFFER.store_meta_datapoint(episode_id, 'fail', False)
        REPLAY_BUFFER.store_meta_datapoint(episode_id, 'task_completed', 1)
        REPLAY_BUFFER.store_meta_datapoint(episode_id, 'marked_timestep', req.timestep)
    else:
        # Use custom penalty
        REPLAY_BUFFER.store_datapoint(episode_id, req.timestep, 'reward', torch.tensor(req.penalty))
        REPLAY_BUFFER.store_meta_datapoint(episode_id, 'fail', True)
        REPLAY_BUFFER.store_meta_datapoint(episode_id, 'task_completed', 0)
        REPLAY_BUFFER.store_meta_datapoint(episode_id, 'marked_timestep', req.timestep)
    
    # Reset value and advantages to NaN as they are now stale
    REPLAY_BUFFER.data['value'][episode_id] = float('nan')
    REPLAY_BUFFER.data['advantages'][episode_id] = float('nan')
    
    # Calculate returns
    timesteps = REPLAY_BUFFER.data['returns'].shape[1]
    returns = torch.full((timesteps,), float('nan'))
    
    # Get max duration for normalization
    task_idx = REPLAY_BUFFER.meta_data['task_id'][episode_id].item()
    task_keys = list(RECAP_CONFIG.get('tasks', {}).keys())
    max_duration = 1.0
    if 0 <= task_idx < len(task_keys):
        task_key = task_keys[task_idx]
        max_duration = RECAP_CONFIG['tasks'][task_key].get('max_episode_length', 1.0)

    for t in range(req.timestep + 1):
        # normalize by max duration
        returns[t] = float(t - req.timestep) / max_duration
    
    REPLAY_BUFFER.data['returns'][episode_id] = returns.numpy()
    
    REPLAY_BUFFER.flush()
    
    returns_list = returns.tolist()
    returns_list = [r if not np.isnan(r) else None for r in returns_list]
    
    return {"status": "ok", "returns": returns_list}

@app.post("/api/returns/calculate")
async def calculate_returns(req: dict):
    filename = req.get("filename")
    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    episode_id = VIDEO_TO_EPISODE.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    task_completed = REPLAY_BUFFER.meta_data['task_completed'][episode_id].item()
    marked_timestep = REPLAY_BUFFER.meta_data['marked_timestep'][episode_id].item()

    if task_completed == -1:
        return {"error": "Video not labelled yet"}

    timesteps = REPLAY_BUFFER.data['returns'].shape[1]
    returns = torch.full((timesteps,), float('nan'))
    
    # Get max duration for normalization
    task_idx = REPLAY_BUFFER.meta_data['task_id'][episode_id].item()
    task_keys = list(RECAP_CONFIG.get('tasks', {}).keys())
    max_duration = 1.0
    if 0 <= task_idx < len(task_keys):
        task_key = task_keys[task_idx]
        max_duration = RECAP_CONFIG['tasks'][task_key].get('max_episode_length', 1.0)

    for t in range(marked_timestep + 1):
        returns[t] = float(t - marked_timestep) / max_duration
    
    REPLAY_BUFFER.data['returns'][episode_id] = returns.numpy()
    
    # Reset value and advantages to NaN as they are now stale
    REPLAY_BUFFER.data['value'][episode_id] = float('nan')
    REPLAY_BUFFER.data['advantages'][episode_id] = float('nan')

    REPLAY_BUFFER.flush()
    
    returns_list = returns.tolist()
    returns_list = [r if not np.isnan(r) else None for r in returns_list]
    
    return {"status": "ok", "returns": returns_list}

async def _calculate_episode_value_internal(episode_id: int, filename: str, max_t: int = None):
    if VALUE_NETWORK is None:
        raise ValueError("Value network not initialized")

    # Get images for this episode
    images = REPLAY_BUFFER.data['images'][episode_id] # (max_timesteps, c, 1, h, w)
    
    video_path = VIDEO_DIR / filename
    num_frames = get_frame_count(video_path)
    
    calc_to_t = num_frames
    if max_t is not None:
        calc_to_t = min(num_frames, max_t)

    values = []
    VALUE_NETWORK.eval()
    
    batch_size = 8
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, calc_to_t, batch_size)):
            batch_images = images[i : min(i + batch_size, calc_to_t)]
            batch_images = torch.from_numpy(batch_images).to(DEVICE)
            batch_images = batch_images.squeeze(2)
            
            # Resize if needed
            if batch_images.shape[-2:] != (224, 224):
                batch_images = TF.resize(batch_images, (224, 224), antialias = True)
            
            batch_values = VALUE_NETWORK(batch_images)
            values.extend(batch_values.cpu().tolist())

    # Store values back to replay buffer
    final_values = torch.full((images.shape[0],), float('nan'))
    final_values[:len(values)] = torch.tensor(values)
    REPLAY_BUFFER.data['value'][episode_id] = final_values.numpy()
    REPLAY_BUFFER.flush()
    return values

@app.post("/api/episode/value/calculate")
async def calculate_episode_value(req: dict):
    filename = req.get("filename")
    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    episode_id = VIDEO_TO_EPISODE.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    try:
        # Check for marked_timestep
        marked_timestep = REPLAY_BUFFER.meta_data['marked_timestep'][episode_id].item()
        max_t = None
        if marked_timestep != -1:
            max_t = marked_timestep + 1

        values = await _calculate_episode_value_internal(episode_id, filename, max_t = max_t)
        return {"status": "ok", "value": values}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/episode/advantage/calculate")
async def calculate_episode_advantage(req: dict):
    filename = req.get("filename")
    gamma = req.get("gamma", 0.99)
    lam = req.get("lam", 0.95)

    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    episode_id = VIDEO_TO_EPISODE.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    # Get actual frame count
    video_path = VIDEO_DIR / filename
    num_frames = get_frame_count(video_path)
    
    marked_timestep = REPLAY_BUFFER.meta_data['marked_timestep'][episode_id].item()
    if marked_timestep != -1:
        num_frames = min(num_frames, marked_timestep + 1)

    # Check if values exist, otherwise calculate them
    values_np = REPLAY_BUFFER.data['value'][episode_id]
    if np.isnan(values_np[:num_frames]).any():
        print(f"Values not found for {filename}, calculating first...")
        await _calculate_episode_value_internal(episode_id, filename, max_t = num_frames)
        values_np = REPLAY_BUFFER.data['value'][episode_id]

    # Prepare inputs for GAE
    rewards = torch.from_numpy(REPLAY_BUFFER.data['reward'][episode_id][:num_frames])
    values = torch.from_numpy(values_np[:num_frames])
    masks = torch.ones_like(rewards) # Assume all frames are valid for now

    # Calculate GAE
    gae_return = calc_generalized_advantage_estimate(
        rewards = rewards,
        values = values,
        masks = masks,
        gamma = gamma,
        lam = lam
    )

    advantages = gae_return.advantages.tolist()

    # Store advantages back
    final_advantages = torch.full((REPLAY_BUFFER.data['advantages'].shape[1],), float('nan'))
    final_advantages[:len(advantages)] = torch.tensor(advantages)
    REPLAY_BUFFER.data['advantages'][episode_id] = final_advantages.numpy()
    REPLAY_BUFFER.flush()

    return {"status": "ok", "advantages": advantages, "value": values.tolist()}

@app.post("/api/advantage/stats")
async def calculate_global_advantage_stats(req: dict):
    percentile = req.get("percentile", 90)
    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}

    # Get all advantages across all episodes
    all_advs = REPLAY_BUFFER.data['advantages']
    
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

@app.post("/api/advantage/binarize")
async def binarize_advantages(req: dict):
    filename = req.get("filename")
    cutoff = req.get("cutoff")

    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    episode_id = VIDEO_TO_EPISODE.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    advs = REPLAY_BUFFER.data['advantages'][episode_id]
    
    # Calculate binarized IDs: 1 if >= cutoff, 0 if < cutoff, -1 if NaN
    adv_ids = np.full(advs.shape, -1, dtype=int)
    valid_mask = ~np.isnan(advs)
    adv_ids[valid_mask] = (advs[valid_mask] >= cutoff).astype(int)

    REPLAY_BUFFER.data['advantage_ids'][episode_id] = adv_ids
    REPLAY_BUFFER.flush()

    return {"status": "ok", "advantage_ids": adv_ids.tolist()}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = Path(__file__).parent / "web_ui" / "index.html"
    return HTMLResponse(content=index_path.read_text(), status_code=200)

@click.command()
@click.option('--folder', default='./video-rollout', help='Folder containing the videos.')
@click.option('--port', default=8000, help='Port to run the server on.')
def main(folder, port):
    global VIDEO_DIR
    VIDEO_DIR = Path(folder)
    
    if not VIDEO_DIR.exists():
        print(f"Error: Folder {folder} does not exist.")
        return

    # Initialize ReplayBuffer in background
    threading.Thread(target=init_replay_buffer, args=(VIDEO_DIR,), daemon=True).start()

    # Initialize Value Network
    global VALUE_NETWORK
    print(f"Initializing SmallValueNetwork on {DEVICE}...")
    VALUE_NETWORK = SmallValueNetwork().to(DEVICE)

    # Mount the video directory to serve files
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
    
    # Mount the cache directory for frames
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/cache", StaticFiles(directory=str(CACHE_DIR)), name="cache")

    # Mount the UI assets
    ui_dir = Path(__file__).parent / "web_ui"
    app.mount("/ui", StaticFiles(directory=str(ui_dir)), name="ui")

    print(f"Starting Video Labeller at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
