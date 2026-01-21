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
import numpy as np
import shutil
from typing import List
from memmap_replay_buffer import ReplayBuffer

app = FastAPI()

# will be set via CLI
VIDEO_DIR = Path(".")
CACHE_DIR = Path(".cache/frames")
REPLAY_BUFFER = None
VIDEO_TO_EPISODE = {}

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
    global REPLAY_BUFFER, VIDEO_TO_EPISODE
    
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
            reward = 'float'
        )
    )

    print(f"Initializing ReplayBuffer from {len(video_files)} videos...")

    for i, video_path in enumerate(video_files):
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
    
    print("ReplayBuffer initialization complete.")

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
    for filename, episode_id in VIDEO_TO_EPISODE.items():
        task_completed = REPLAY_BUFFER.meta_data['task_completed'][episode_id].item()
        marked_timestep = REPLAY_BUFFER.meta_data['marked_timestep'][episode_id].item()
        
        if task_completed != -1:
            result[filename] = {
                "task_completed": task_completed,
                "marked_timestep": marked_timestep
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
    
    # Optional: also clear reward at the marked timestep if we wanted to be thorough
    # but since task_completed is -1, it's effectively reset.
    
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
    
    REPLAY_BUFFER.flush()
    return {"status": "ok"}

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

    # Initialize ReplayBuffer
    init_replay_buffer(VIDEO_DIR)

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
