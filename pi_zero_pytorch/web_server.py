import os
import time
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
from typing import List, Optional
from memmap_replay_buffer import ReplayBuffer
from pi_zero_pytorch.pi_zero import calc_generalized_advantage_estimate, SigLIP, BinnedValueLayer, PiZero, LinearNoBias
import json
import tqdm
import threading
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

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
        self.image_size = image_size
        self.patch_size = patch_size

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

    def forward(self, images, return_value_and_logits = False):
        embeds = self.siglip(images)
        pooled = embeds.mean(dim = 1)
        return self.to_value(pooled, return_value_and_logits = return_value_and_logits)

VALUE_NETWORK_CONFIGS = {
    "mock": {"dim": 8, "depth": 1, "heads": 1, "image_size": 32, "patch_size": 16},
    "small": {"dim": 64, "depth": 2, "heads": 4, "image_size": 224, "patch_size": 14},
    "medium": {"dim": 128, "depth": 4, "heads": 8, "image_size": 224, "patch_size": 14},
    "large": {"dim": 256, "depth": 6, "heads": 8, "image_size": 224, "patch_size": 14}
}

# small mock pi-zero for specialized fine-tuning in recap loop

class SmallPiZero(torch.nn.Module):
    def __init__(
        self,
        dim = 32,
        dim_action = 32, 
        dim_action_input = 6,
        dim_joint_state = 32,
        num_tokens = 256,
        depth = 2,
        heads = 4,
        image_size = 32,
        patch_size = 4,
        max_text_len = 32,
        num_advantage_tokens = 2
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_text_len = max_text_len

        # minimal vit
        self.vit = SigLIP(
            image_size = image_size,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            heads = heads
        )

        # minimal pizero
        self.pizero = PiZero(
            dim = dim,
            num_tokens = num_tokens,
            dim_action_input = dim_action_input,
            dim_joint_state = dim_joint_state,
            dim_action = dim_action,
            depth = depth,
            heads = heads,
            vit = self.vit,
            vit_dim = dim,
            num_advantage_tokens = num_advantage_tokens
        )

    def forward(self, images, token_ids, joint_state, actions, advantage_ids = None, **kwargs):
        return self.pizero(
            images = images,
            token_ids = token_ids,
            joint_state = joint_state,
            actions = actions,
            advantage_ids = advantage_ids,
            **kwargs
        )

PI_ZERO_CONFIGS = {
    "mock": {"dim": 4, "depth": 1, "heads": 1, "image_size": 32, "patch_size": 16},
    "small": {"dim": 16, "depth": 1, "heads": 2, "image_size": 32, "patch_size": 8}
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()
TRAINING_STATE = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_step": 0,
    "total_steps": 0,
    "last_loss": 0.0
}

app = FastAPI()

# will be set via CLI
VIDEO_DIRS = []
CACHE_DIR = Path(".cache/frames")
REPLAY_BUFFER = None
VIDEO_TO_EPISODE = {}
VALUE_NETWORK = None
RECAP_WORKSPACE = None  # Set via --recap-workspace CLI option
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
FAST_MOCK = os.getenv("FAST_MOCK", "false").lower() == "true"


class LabelRequest(BaseModel):
    filename: str
    timestep: int
    success: bool
    penalty: float = -50.0

class InterventionRequest(BaseModel):
    filename: str
    timestep: int

class VideoInfo(BaseModel):
    filename: str
    frames: int
    url: str
    folder: str = ""

def get_video_path(filename: str) -> Optional[Path]:
    for vdir in VIDEO_DIRS:
        video_path = vdir / filename
        if video_path.exists():
            return video_path
    return None

def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def extract_frames(video_path: Path, cache_path: Path):
    if cache_path.exists() and any(cache_path.glob("*.jpg")):
        print(f"[RECAP] Frames already exist in {cache_path}, skipping extraction.")
        return

    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"[RECAP] Extracting frames from {video_path} to {cache_path}...")
    
    if FAST_MOCK:
        print("[RECAP] FAST_MOCK is enabled, creating dummy frame.")
        (cache_path / "frame_0000.jpg").touch()
        return

    cap = cv2.VideoCapture(str(video_path))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(cache_path / f"frame_{count:04d}.jpg"), frame)
        count += 1
    cap.release()
    print(f"[RECAP] Extracted {count} frames.")

def init_replay_buffer(video_dirs: List[Path]):
    global REPLAY_BUFFER, VIDEO_TO_EPISODE, CONVERSION_STATUS
    
    CONVERSION_STATUS["is_converting"] = True
    CONVERSION_STATUS["progress"] = 0
    
    tmp_buffer_dir = Path("tmp/replay_buffer")
    if tmp_buffer_dir.exists():
        shutil.rmtree(tmp_buffer_dir)
    tmp_buffer_dir.mkdir(parents=True, exist_ok = True)

    extensions = {".mp4", ".webm", ".avi", ".mov"}
    video_files = []
    
    # Support both single Path and list of Paths
    if isinstance(video_dirs, Path):
        video_dirs = [video_dirs]

    for vdir in video_dirs:
        video_files.extend([f for f in vdir.iterdir() if f.suffix.lower() in extensions and f.stat().st_size > 0])
    
    video_files = sorted(video_files, key=lambda x: x.name)
    
    if not video_files:
        print("No valid video files found")
        CONVERSION_STATUS["is_converting"] = False
        return

    h, w, c, max_frames = None, None, None, 0
    for vf in video_files:
        cap = cv2.VideoCapture(str(vf))
        ret, frame = cap.read()
        cap.release()
        if ret:
            h, w, c = frame.shape
            break
    
    if h is None:
        print("Could not read any video files")
        CONVERSION_STATUS["is_converting"] = False
        return
    
    for vf in video_files:
        max_frames = max(max_frames, get_frame_count(vf))


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
            recap_step     = ('int', (), -1),
            is_expert_intervention = 'bool'
        ),
        fields = dict(
            images = ('float', (c, 1, h, w)), # matching (c, num_images, h, w) pattern
            text = ('int', (32,)), # dummy text len
            internal = ('float', (32,)), # matches dim_joint_state
            actions = ('float', (16, 6)), # dummy trajectory
            reward = 'float',
            returns = ('float', (), float('nan')),
            value = ('float', (), float('nan')),
            advantages = ('float', (), float('nan')),
            advantage_ids = ('int', (), -1),
            invalidated = 'bool',
            expert_segment = 'bool'
        )
    )

    print(f"Initializing ReplayBuffer from {len(video_files)} videos...")
    CONVERSION_STATUS["total"] = len(video_files)

    for i, video_path in enumerate(video_files):
        CONVERSION_STATUS["current_video"] = video_path.name
        CONVERSION_STATUS["progress"] = i
        
        VIDEO_TO_EPISODE[video_path.name] = i
        
        if FAST_MOCK:
            # Skip video reading and populate with minimal dummy data
            # Use small dimensions to speed up everything
            with REPLAY_BUFFER.one_episode(task_id = torch.tensor(-1)):
                # Just store one dummy frame per video to satisfy the buffer
                # In a real demo we might want more, but for "extremely fast" one is enough
                REPLAY_BUFFER.store(
                    images = torch.randn(c, 1, h, w),
                    text = torch.zeros(32, dtype=torch.long),
                    internal = torch.randn(32),
                    actions = torch.randn(16, 6),
                    reward = 0.0
                )
            continue

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
                
                # mock other fields
                mock_text = torch.randint(0, 100, (32,))
                mock_internal = torch.randn(32)
                mock_actions = torch.randn(16, 6)
                
                REPLAY_BUFFER.store(
                    images = frame_tensor,
                    text = mock_text,
                    internal = mock_internal,
                    actions = mock_actions,
                    reward = 0.0
                )
        cap.release()
    
    CONVERSION_STATUS["is_converting"] = False
    CONVERSION_STATUS["progress"] = len(video_files)
    print("ReplayBuffer initialization complete.")

@app.get("/api/status")
async def get_status():
    return CONVERSION_STATUS

@app.get("/api/training/status")
async def get_training_status():
    return TRAINING_STATE

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
    
    dirs_to_scan = VIDEO_DIRS if VIDEO_DIRS else []
    
    for vdir in dirs_to_scan:
        if not vdir.exists(): continue
        for file in vdir.iterdir():
            if file.suffix.lower() in extensions and file.stat().st_size > 0:
                frames = get_frame_count(file)
                if frames > 0:  # Only include videos with valid frames
                    videos.append(VideoInfo(
                        filename=file.name,
                        frames=frames,
                        url=f"/videos/{file.name}",
                        folder=vdir.name
                    ))
    
    # sort by filename
    videos.sort(key=lambda x: x.filename)
    return videos

@app.get("/videos/{filename}")
async def serve_video(filename: str):
    if not VIDEO_DIRS:
        return {"error": "Video directories not set"}
    
    video_path = get_video_path(filename)
    if video_path:
        return FileResponse(video_path)
            
    return {"error": "Video not found"}


@app.get("/api/video/{filename}/frames")
async def get_video_frames(filename: str):
    video_path = None
    video_path = get_video_path(filename)
    if not video_path:
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
        
        # Get returns, values, advantages
        returns = REPLAY_BUFFER.data['returns'][episode_id].tolist()
        value = REPLAY_BUFFER.data['value'][episode_id].tolist()
        advantages = REPLAY_BUFFER.data['advantages'][episode_id].tolist()
        advantage_ids = REPLAY_BUFFER.data['advantage_ids'][episode_id].tolist()

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
            "is_expert_intervention": REPLAY_BUFFER.meta_data['is_expert_intervention'][episode_id].item(),
            "expert_segment": REPLAY_BUFFER.data['expert_segment'][episode_id].tolist(),
            "invalidated": REPLAY_BUFFER.data['invalidated'][episode_id].tolist()
        }
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
    REPLAY_BUFFER.store_meta_datapoint(episode_id, 'is_expert_intervention', False)
    
    # Reset fields
    REPLAY_BUFFER.data['returns'][episode_id] = float('nan')
    REPLAY_BUFFER.data['value'][episode_id] = float('nan')
    REPLAY_BUFFER.data['advantages'][episode_id] = float('nan')
    REPLAY_BUFFER.data['advantage_ids'][episode_id] = -1
    REPLAY_BUFFER.data['invalidated'][episode_id] = False
    REPLAY_BUFFER.data['expert_segment'][episode_id] = False
    
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
    REPLAY_BUFFER.data['invalidated'][episode_id] = False
    
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

@app.post("/api/label/intervention")
async def label_intervention(req: InterventionRequest):
    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    episode_id = VIDEO_TO_EPISODE.get(req.filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    # Set meta flag
    REPLAY_BUFFER.store_meta_datapoint(episode_id, 'is_expert_intervention', True)
    
    # Set segment: everything up to current timestep is expert controlled
    expert_mask = REPLAY_BUFFER.data['expert_segment'][episode_id]
    expert_mask[:req.timestep + 1] = True
    REPLAY_BUFFER.data['expert_segment'][episode_id] = expert_mask

    # Force advantage_ids to 1 (Positive) for the expert segment
    # This is a core RECAP mechanic: expert interventions are "ground truth" positives
    adv_ids = REPLAY_BUFFER.data['advantage_ids'][episode_id]
    adv_ids[:req.timestep + 1] = 1
    REPLAY_BUFFER.data['advantage_ids'][episode_id] = adv_ids

    REPLAY_BUFFER.flush()
    
    return {
        "status": "ok", 
        "is_expert_intervention": True,
        "expert_segment": expert_mask.tolist(),
        "advantage_ids": adv_ids.tolist()
    }

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
    REPLAY_BUFFER.data['invalidated'][episode_id] = False

    REPLAY_BUFFER.flush()
    
    returns_list = returns.tolist()
    returns_list = [r if not np.isnan(r) else None for r in returns_list]
    
    return {"status": "ok", "returns": returns_list}

async def _calculate_episode_value_internal(episode_id: int, filename: str, max_t: int = None):
    if VALUE_NETWORK is None:
        raise ValueError("Value network not initialized")

    # Get images for this episode
    images = REPLAY_BUFFER.data['images'][episode_id] # (max_timesteps, c, 1, h, w)
    
    video_path = get_video_path(filename)
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
            
            # Use model's expected image size
            target_size = (VALUE_NETWORK.image_size, VALUE_NETWORK.image_size)
            if batch_images.shape[-2:] != target_size:
                batch_images = TF.resize(batch_images, target_size, antialias = True)
            
            batch_values = VALUE_NETWORK(batch_images)
            values.extend(batch_values.cpu().tolist())

    # Store values back to replay buffer
    final_values = torch.full((images.shape[0],), float('nan'))
    final_values[:len(values)] = torch.tensor(values)
    REPLAY_BUFFER.data['value'][episode_id] = final_values.numpy()
    REPLAY_BUFFER.data['invalidated'][episode_id] = False
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

    try:
        # Get actual frame count
        video_path = get_video_path(filename)
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

        print(f"[RECAP] Calculated advantages for {filename} (ID: {episode_id}). Count: {len(advantages)}")
        valid_advs = torch.tensor(advantages)[~torch.isnan(torch.tensor(advantages))]
        if len(valid_advs) > 0:
            print(f"[RECAP] Advantages - Min: {valid_advs.min().item():.4f}, Max: {valid_advs.max().item():.4f}, Mean: {valid_advs.mean().item():.4f}")
        # Binarize Advantage IDs, but respect expert segments
        # 1 if >= cutoff (once global binarization happens), but for now we might just calculate continuous
        # RECAP requires binarized advantages in the buffer.
        # If it's an expert segment, we MUST set it to 1.
        
        expert_mask = REPLAY_BUFFER.data['expert_segment'][episode_id][:num_frames]
        adv_ids = REPLAY_BUFFER.data['advantage_ids'][episode_id]
        
        # We don't have a cutoff here, so we don't binarize regular steps yet.
        # But we DO ensure expert steps are marked.
        adv_ids[:num_frames][expert_mask] = 1
        REPLAY_BUFFER.data['advantage_ids'][episode_id] = adv_ids

        REPLAY_BUFFER.data['invalidated'][episode_id] = False
        REPLAY_BUFFER.flush()

        return {"status": "ok", "advantages": advantages, "value": values.tolist(), "advantage_ids": adv_ids.tolist()}
    except Exception as e:
        print(f"Error calculating advantage for {filename}: {str(e)}")
        return {"error": str(e)}

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

    print(f"[RECAP] Binarized advantages for {filename} (ID: {episode_id}) with cutoff {cutoff:.4f}. Pos: {np.sum(adv_ids == 1)}, Neg: {np.sum(adv_ids == 0)}")

    return {"status": "ok", "advantage_ids": adv_ids.tolist()}

@app.post("/api/episode/invalidate")
async def invalidate_episode_timesteps(req: dict):
    filename = req.get("filename")
    cutoff = req.get("cutoff", 0.0)

    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    episode_id = VIDEO_TO_EPISODE.get(filename)
    if episode_id is None:
        return {"error": "Video not found in buffer"}

    advs = REPLAY_BUFFER.data['advantages'][episode_id]
    
    # Calculate invalidated mask: True if advantage <= cutoff and not NaN
    valid_mask = ~np.isnan(advs)
    invalidated = np.zeros(advs.shape, dtype=bool)
    invalidated[valid_mask] = (advs[valid_mask] <= cutoff)

    REPLAY_BUFFER.data['invalidated'][episode_id] = invalidated
    REPLAY_BUFFER.flush()

    return {"status": "ok", "invalidated": invalidated.tolist()}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = Path(__file__).parent / "web_ui" / "index.html"
    return HTMLResponse(content=index_path.read_text(), status_code=200)

@app.get("/api/recap/state")
async def get_recap_state():
    """Returns the current state of the RECAP workspace for UI introspection."""
    if not RECAP_WORKSPACE or not RECAP_WORKSPACE.exists():
        return {"enabled": False}

    state = {
        "enabled": True,
        "workspace": str(RECAP_WORKSPACE),
        "pretrained": {
            "actor": (RECAP_WORKSPACE / "pretrained-actor.pt").exists(),
            "critic": (RECAP_WORKSPACE / "pretrained-critic.pt").exists()
        },
        "tasks": []
    }

    # Get tasks from config
    extensions = {".mp4", ".webm", ".avi", ".mov"}
    for task_name, task_config in RECAP_CONFIG.get("tasks", {}).items():
        task_dir = RECAP_WORKSPACE / task_name
        task_state = {
            "name": task_name,
            "max_episode_length": task_config.get("max_episode_length", 200),
            "exists": task_dir.exists(),
            "iterations": []
        }

        if task_dir.exists():
            # Find iterations (numbered folders: 0, 1, 2...)
            for iter_dir in sorted(task_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else -1):
                if iter_dir.is_dir() and iter_dir.name.isdigit():
                    iter_id = int(iter_dir.name)
                    # Find data folders (data.0, data.1, ...)
                    data_folders = []
                    for d in sorted(iter_dir.iterdir()):
                        if d.is_dir() and not d.name.isdigit():
                            video_count = len([f for f in d.iterdir() if f.suffix.lower() in extensions and f.stat().st_size > 0])
                            data_folders.append({
                                "id": d.name,
                                "video_count": video_count
                            })

                    task_state["iterations"].append({
                        "id": iter_id,
                        "actor": (iter_dir / "actor.pt").exists(),
                        "critic": (iter_dir / "critic.pt").exists(),
                        "data": data_folders
                    })

        state["tasks"].append(task_state)

    return state

@app.post("/api/recap/pretrain")
async def recap_pretrain():
    """Simulates generalist pretraining with a single gradient step on dummy data."""
    if not RECAP_WORKSPACE:
        return {"error": "RECAP workspace not configured"}

    actor_path = RECAP_WORKSPACE / "pretrained-actor.pt"
    critic_path = RECAP_WORKSPACE / "pretrained-critic.pt"
    
    if actor_path.exists():
        return {"error": "Already pretrained"}
    
    print("Pretraining: performing one gradient step on dummy data...")
    
    # Use mock config for pretraining speed
    config = PI_ZERO_CONFIGS["mock"]
    model = SmallPiZero(**config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    
    # Dummy data
    images = torch.randn(1, 3, config["image_size"], config["image_size"]).to(DEVICE)
    text = torch.zeros(1, 32, dtype = torch.long).to(DEVICE)
    internal = torch.randn(1, 32).to(DEVICE)
    actions = torch.randn(1, 6).to(DEVICE)
    
    # One gradient step
    output = model(images, text, internal, actions)
    loss = output[0] if isinstance(output, tuple) else output
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save dummy weights (simulation)
    torch.save(model.state_dict(), str(actor_path))
    torch.save(model.state_dict(), str(critic_path))
    
    return {"status": "ok"}

@app.post("/api/recap/specialize")
async def recap_specialize(req: dict):
    """Creates iteration 0 (SFT) for a specific task."""
    if not RECAP_WORKSPACE:
        return {"error": "RECAP workspace not configured"}
    
    task_name = req.get("task_name")
    if not task_name:
        return {"error": "Missing task_name"}
    
    # Check that pretrained weights exist
    if not (RECAP_WORKSPACE / "pretrained-actor.pt").exists():
        return {"error": "Must pretrain first"}
    
    # Create iteration 0 (SFT)
    iter_dir = RECAP_WORKSPACE / task_name / "0"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate creating specialized weights
    torch.save({"simulated": True, "task": task_name, "iteration": 0}, str(iter_dir / "actor.pt"))
    torch.save({"simulated": True, "task": task_name, "iteration": 0}, str(iter_dir / "critic.pt"))
    
    return {"status": "ok"}

@app.post("/api/recap/collect")
async def recap_collect(req: dict):
    """Simulates data collection - creates a new data folder with sample videos."""
    if not RECAP_WORKSPACE:
        return {"error": "RECAP workspace not configured"}
    
    task_name = req.get("task_name")
    iter_id = req.get("iter_id", 0)
    
    if not task_name:
        return {"error": "Missing task_name"}
    
    iter_dir = RECAP_WORKSPACE / task_name / str(iter_id)
    if not iter_dir.exists():
        return {"error": f"Iteration {iter_id} does not exist for {task_name}"}
    
    # Find next data folder index
    existing_data = list(iter_dir.glob("data.*"))
    next_idx = len(existing_data)
    
    data_dir = iter_dir / f"data.{next_idx}"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Import and use simulation engine to generate sample trajectories
    try:
        from .recap_sim_engine import generate_trajectories
        generate_trajectories(data_dir, num_episodes=2, steps=20)
        print(f"RECAP Collect: Simulated 2 episodes in {data_dir}")
    except (ImportError, ValueError) as e:
        print(f"RECAP Collect: Simulation failed or skipped: {e}")
        try:
             import pi_zero_pytorch.recap_sim_engine as rse
             rse.generate_trajectories(data_dir, num_episodes=2, steps=20)
             print(f"RECAP Collect: Simulated 2 episodes via absolute import in {data_dir}")
        except:
             pass
    
    return {"status": "ok", "data_folder": f"data.{next_idx}"}

@app.post("/api/recap/iterate")
async def recap_iterate(req: dict):
    """Advances to the next iteration after finetuning on collected data."""
    if not RECAP_WORKSPACE:
        return {"error": "RECAP workspace not configured"}
    
    task_name = req.get("task_name")
    iter_id = req.get("iter_id", 0)
    
    if not task_name:
        return {"error": "Missing task_name"}
    
    current_iter_dir = RECAP_WORKSPACE / task_name / str(iter_id)
    if not current_iter_dir.exists():
        return {"error": f"Iteration {iter_id} does not exist"}
    
    # Check that data was collected
    data_folders = list(current_iter_dir.glob("data.*"))
    if not data_folders:
        return {"error": "No data collected for this iteration"}
    
    # Create next iteration
    next_iter_id = iter_id + 1
    next_iter_dir = RECAP_WORKSPACE / task_name / str(next_iter_id)
    next_iter_dir.mkdir(parents=True, exist_ok=True)
    
    # If policy fine-tuning just finished, it might have saved weights in a 'policy_finetuned' dir
    # we move it to the next iteration
    finetuned_actor = RECAP_WORKSPACE / "policy_finetuned" / "actor.pt"
    if finetuned_actor.exists():
        shutil.move(str(finetuned_actor), str(next_iter_dir / "actor.pt"))
        print(f"Moved finetuned actor to {next_iter_dir}")
    else:
        # Simulate creating updated weights if not existing
        torch.save({"simulated": True, "task": task_name, "iteration": next_iter_id}, str(next_iter_dir / "actor.pt"))

    torch.save({"simulated": True, "task": task_name, "iteration": next_iter_id}, str(next_iter_dir / "critic.pt"))
    
    return {"status": "ok", "new_iteration": next_iter_id}

@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Just keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def _broadcast_training_update(loop):
    asyncio.run_coroutine_threadsafe(
        manager.broadcast(json.dumps({
            "type": "training_update",
            "state": TRAINING_STATE
        })),
        loop
    )

def train_value_network_thread(config_name: str, loop):
    global VALUE_NETWORK, TRAINING_STATE
    
    # 1. Prepare data
    if config_name == "mock":
        # Extremely fast mock data
        images = torch.randn(1, 3, 32, 32)
        returns = torch.randn(1)
        dataset = torch.utils.data.TensorDataset(images, returns)
    else:
        all_images = []
        all_returns = []
        for i in range(len(REPLAY_BUFFER)):
            returns = REPLAY_BUFFER.data['returns'][i]
            valid_mask = ~np.isnan(returns)
            if valid_mask.any():
                images = REPLAY_BUFFER.data['images'][i][valid_mask]
                all_images.append(torch.from_numpy(images))
                all_returns.append(torch.from_numpy(returns[valid_mask]))
        
        if not all_images:
            TRAINING_STATE["is_training"] = False
            _broadcast_training_update(loop)
            return

        dataset = torch.utils.data.TensorDataset(torch.cat(all_images), torch.cat(all_returns))
    batch_size = 16
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

    # 2. Initialize model
    config = VALUE_NETWORK_CONFIGS.get(config_name, VALUE_NETWORK_CONFIGS["small"])
    model = SmallValueNetwork(**config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    model.train()

    num_epochs = 10
    total_steps = len(loader) * num_epochs if config_name != "mock" else 1
    TRAINING_STATE.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": num_epochs if config_name != "mock" else 1,
        "current_step": 0,
        "total_steps": total_steps,
        "last_loss": 0.0
    })
    _broadcast_training_update(loop)

    # 3. Training loop
    target_size = (config.get('image_size', 224), config.get('image_size', 224))
    
    for epoch in range(num_epochs):
        TRAINING_STATE["current_epoch"] = epoch + 1
        for i, (images, returns) in enumerate(loader):
            images = images.to(DEVICE).squeeze(2)
            if images.shape[-2:] != target_size:
                images = TF.resize(images, target_size, antialias = True)
            
            returns = returns.to(DEVICE)
            
            values, logits = model(images, return_value_and_logits = True)
            loss = model.to_value.loss_fn(logits, returns).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            TRAINING_STATE["current_step"] += 1
            TRAINING_STATE["last_loss"] = float(loss.item())
            
            # For mock config, we only do one gradient step total
            if config_name == "mock":
                print("Mock training: finishing after one gradient step.")
                break
            
            if i % 5 == 0:
                _broadcast_training_update(loop)
        
        if config_name == "mock":
            break
            
        _broadcast_training_update(loop)

    # 4. Finalize
    VALUE_NETWORK = model
    
    # Save the model
    if RECAP_WORKSPACE:
        networks_dir = RECAP_WORKSPACE / "value_networks"
        networks_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{config_name}_{timestamp}.pt"
        model_path = networks_dir / model_filename
        
        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
            "config_name": config_name,
            "epochs": num_epochs,
            "final_loss": TRAINING_STATE["last_loss"],
            "timestamp": timestamp
        }, str(model_path))
        print(f"Value network saved to {model_path}")

    TRAINING_STATE["is_training"] = False
    _broadcast_training_update(loop)

@app.get("/api/value/networks/list")
async def list_value_networks():
    if not RECAP_WORKSPACE:
        return []
    
    networks_dir = RECAP_WORKSPACE / "value_networks"
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

@app.post("/api/value/networks/load")
async def load_value_network(req: dict):
    global VALUE_NETWORK
    filename = req.get("filename")
    if not filename or not RECAP_WORKSPACE:
        return {"error": "Missing filename or RECAP_WORKSPACE"}
    
    model_path = RECAP_WORKSPACE / "value_networks" / filename
    if not model_path.exists():
        return {"error": "Model file not found"}
    
    try:
        checkpoint = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
        config = checkpoint["config"]
        model = SmallValueNetwork(**config).to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        VALUE_NETWORK = model
        return {"status": "ok", "config_name": checkpoint.get("config_name")}
    except Exception as e:
        return {"error": str(e)}

def train_policy_network_thread(config_name: str, loop):
    global TRAINING_STATE
    
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
        for i in range(len(REPLAY_BUFFER)):
            advantage_ids = REPLAY_BUFFER.data['advantage_ids'][i]
            valid_mask = advantage_ids != -1
            if valid_mask.any():
                all_images.append(torch.from_numpy(REPLAY_BUFFER.data['images'][i][valid_mask]))
                all_text.append(torch.from_numpy(REPLAY_BUFFER.data['text'][i][valid_mask]))
                all_internal.append(torch.from_numpy(REPLAY_BUFFER.data['internal'][i][valid_mask]))
                all_actions.append(torch.from_numpy(REPLAY_BUFFER.data['actions'][i][valid_mask]))
                all_advantage_ids.append(torch.from_numpy(advantage_ids[valid_mask]))
        
        if not all_images:
            print("No valid data for policy training")
            TRAINING_STATE["is_training"] = False
            _broadcast_training_update(loop)
            return

        dataset = torch.utils.data.TensorDataset(
            torch.cat(all_images),
            torch.cat(all_text),
            torch.cat(all_internal),
            torch.cat(all_actions),
            torch.cat(all_advantage_ids)
        )
    
    batch_size = 4
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

    # 2. Initialize model
    config = PI_ZERO_CONFIGS.get(config_name, PI_ZERO_CONFIGS["small"])
    model = SmallPiZero(**config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4) # lower lr for finetuning
    model.train()

    num_epochs = 1
    total_steps = len(loader) * num_epochs if config_name != "mock" else 1
    TRAINING_STATE.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": num_epochs if config_name != "mock" else 1,
        "current_step": 0,
        "total_steps": total_steps,
        "last_loss": 0.0
    })
    _broadcast_training_update(loop)

    # 3. Training loop
    target_size = (config.get('image_size', 32), config.get('image_size', 32))
    
    print(f"Starting policy fine-tuning for {num_epochs} epoch...")
    for epoch in range(num_epochs):
        TRAINING_STATE["current_epoch"] = epoch + 1
        for i, (images, text, internal, actions, adv_ids) in enumerate(loader):
            images = images.to(DEVICE).squeeze(2)
            if images.shape[-2:] != target_size:
                images = TF.resize(images, target_size, antialias = True)
            
            text = text.to(DEVICE)
            internal = internal.to(DEVICE)
            actions = actions.to(DEVICE)
            adv_ids = adv_ids.to(DEVICE)
            
            # Conditioned on advantage_ids
            output = model(images, text, internal, actions, advantage_ids = adv_ids)
            loss = output[0] if isinstance(output, tuple) else output
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            TRAINING_STATE["current_step"] += 1
            TRAINING_STATE["last_loss"] = float(loss.item())
            
            # For mock config, we only do one gradient step total
            if config_name == "mock":
                print("Mock finetuning: finishing after one gradient step.")
                break
            
            if i % 2 == 0:
                _broadcast_training_update(loop)
        
        if config_name == "mock":
            break
            
        _broadcast_training_update(loop)

    # 4. Finalize
    if RECAP_WORKSPACE:
        policy_dir = RECAP_WORKSPACE / "policy_finetuned"
        policy_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = policy_dir / "actor.pt"
        torch.save(model.pizero.state_dict(), str(model_path))
        print(f"Finetuned policy saved to {model_path}")

    TRAINING_STATE["is_training"] = False
    _broadcast_training_update(loop)

@app.post("/api/recap/finetune")
async def start_policy_finetune(req: dict):
    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    if TRAINING_STATE["is_training"]:
        return {"error": "Training already in progress"}
        
    config_name = req.get("config", "mock") # use mock for speed in e2e
    loop = asyncio.get_event_loop()
    threading.Thread(target=train_policy_network_thread, args=(config_name, loop), daemon=True).start()
    
    return {"status": "ok"}

@app.post("/api/value/train")
async def start_training(req: dict):
    if REPLAY_BUFFER is None:
        return {"error": "ReplayBuffer not initialized"}
    
    if TRAINING_STATE["is_training"]:
        return {"error": "Training already in progress"}
        
    config_name = req.get("config", "small")
    loop = asyncio.get_event_loop()
    threading.Thread(target=train_value_network_thread, args=(config_name, loop), daemon=True).start()
    
    return {"status": "ok"}

@app.post("/api/recap/simulate_collection")
async def simulate_collection_api(req: dict):
    """Simulates collecting a new batch of data for a task/iteration."""
    task_name = req.get("task_name")
    iter_id = req.get("iter_id", 0)
    
    if not task_name:
        return {"error": "Missing task_name"}
    
    # Create target directory
    timestamp = int(time.time())
    data_id = f"data.batch_{timestamp}"
    target_dir = RECAP_WORKSPACE / task_name / str(iter_id) / data_id
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy 2 random videos from video-rollout as mock data
    try:
        from .recap_sim_engine import generate_trajectories
        generate_trajectories(target_dir, num_episodes=2)
    except:
        try:
            from pi_zero_pytorch.recap_sim_engine import generate_trajectories
            generate_trajectories(target_dir, num_episodes=2)
        except Exception as e:
             print(f"Simulate collection API: failed to generate trajectories: {e}")
    
    return {"status": "ok", "task_name": task_name, "iter_id": iter_id, "data_id": data_id}

@app.post("/api/recap/load_data")
async def recap_load_data(req: dict):
    """Mounts a specific data folder to view in the labeller."""
    global VIDEO_DIRS, REPLAY_BUFFER, VIDEO_TO_EPISODE, CONVERSION_STATUS
    
    task_name = req.get("task_name")
    iter_id = req.get("iter_id")
    data_id = req.get("data_id")
    
    if not all([task_name, iter_id is not None, data_id]):
        return {"error": "Missing required parameters"}
    
    target_dir = RECAP_WORKSPACE / task_name / str(iter_id) / data_id
    if not target_dir.exists():
        return {"error": f"Data directory {target_dir} does not exist"}
    
    # Reset current state
    VIDEO_DIRS = [target_dir]
    REPLAY_BUFFER = None
    VIDEO_TO_EPISODE = {}
    
    # Start conversion in background
    threading.Thread(target=init_replay_buffer, args=(VIDEO_DIRS,), daemon=True).start()
    
    return {"status": "ok", "video_dir": str(target_dir)}

@click.command()
@click.option('--port', default=8000, help='Port to run the server on.')
@click.option('--folder', 'folders', multiple=True, help='Path to video directory for standalone mode.')
@click.option('--recap-workspace', default=None, help='Path to RECAP algorithm workspace folder.')
def main(port, folders, recap_workspace):
    global VIDEO_DIRS, RECAP_WORKSPACE

    # Initialize RECAP workspace
    if recap_workspace:
        RECAP_WORKSPACE = Path(recap_workspace)
        RECAP_WORKSPACE.mkdir(parents=True, exist_ok=True)
        print(f"RECAP workspace enabled: {RECAP_WORKSPACE}")
    else:
        RECAP_WORKSPACE = None

    if folders:
        VIDEO_DIRS = [Path(f) for f in folders]
        valid_dirs = []
        for vdir in VIDEO_DIRS:
            if not vdir.exists():
                 print(f"Warning: Folder {vdir} does not exist")
            else:
                valid_dirs.append(vdir)
        
        if not valid_dirs:
            print("Error: No valid folders provided")
            return

        VIDEO_DIRS = valid_dirs
        print(f"Standalone mode: loading videos from {len(VIDEO_DIRS)} folders")
        # Initialize buffer immediately for standalone mode
        threading.Thread(target=init_replay_buffer, args=(VIDEO_DIRS,), daemon=True).start()
    else:
        # VIDEO_DIRS will be set dynamically in RECAP mode
        VIDEO_DIRS = []
    
    # Initialize CONVERSION_STATUS as not converting
    global CONVERSION_STATUS
    CONVERSION_STATUS = {
        "is_converting": False,
        "progress": 0,
        "total": 0,
        "current_video": ""
    }

    # Initialize Value Network
    global VALUE_NETWORK
    print(f"Initializing SmallValueNetwork on {DEVICE}...")
    VALUE_NETWORK = SmallValueNetwork().to(DEVICE)
    
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
