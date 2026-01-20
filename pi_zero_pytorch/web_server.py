import os
import cv2
import click
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List
from pydantic import BaseModel

app = FastAPI()

# will be set via CLI
VIDEO_DIR = Path(".")
CACHE_DIR = Path(".cache/frames")

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
