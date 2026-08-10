import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import WebSocket


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
            except Exception:
                pass


def default_device() -> torch.device:
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def default_conversion_status() -> Dict[str, Any]:
    return {
        "is_converting": False,
        "progress": 0,
        "total": 0,
        "current_video": ""
    }


def default_training_state() -> Dict[str, Any]:
    return {
        "is_training": False,
        "current_epoch": 0,
        "total_epochs": 0,
        "current_step": 0,
        "total_steps": 0,
        "last_loss": 0.0
    }


@dataclass
class AppState:
    video_dirs: List[Path] = field(default_factory=list)
    cache_dir: Path = field(default_factory=lambda: Path(".cache/frames"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    replay_buffer: Any = None
    video_to_episode: Dict[str, int] = field(default_factory=dict)
    video_to_path: Dict[str, Path] = field(default_factory=dict)
    video_to_proprio: Dict[str, Any] = field(default_factory=dict)
    num_views: int = 1
    value_network: Any = None
    recap_workspace: Optional[Path] = None
    recap_config: Dict[str, Any] = field(default_factory=dict)
    device: torch.device = field(default_factory=default_device)
    fast_mock: bool = field(default_factory=lambda: os.getenv("FAST_MOCK", "false").lower() == "true")
    conversion_status: Dict[str, Any] = field(default_factory=default_conversion_status)
    training_state: Dict[str, Any] = field(default_factory=default_training_state)
    manager: ConnectionManager = field(default_factory=ConnectionManager)
    conversion_lock: threading.Lock = field(default_factory=threading.Lock)


app_state = AppState()
