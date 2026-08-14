import logging
import shutil
import time
from pathlib import Path

import torch

from .errors import ApiError
from .networks import PI_ZERO_CONFIGS, SmallPiZero
from .sim_engine import generate_trajectories
from .state import AppState

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov"}


# helpers


def _require_workspace(state: AppState) -> Path:
    if not state.recap_workspace:
        raise ApiError("RECAP workspace not configured", status_code=409)
    return state.recap_workspace


def _require_task(state: AppState, task_name: str) -> Path:
    if not task_name:
        raise ApiError("Missing task_name")
    return _require_workspace(state) / task_name


def _count_videos(directory: Path) -> int:
    return len([
        f for f in directory.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS and f.stat().st_size > 0
    ])


def _save_simulated_weights(iter_dir: Path, task_name: str, iteration: int):
    for name in ("actor.pt", "critic.pt"):
        torch.save({"simulated": True, "task": task_name, "iteration": iteration}, str(iter_dir / name))


def _generate_episodes(data_dir: Path, **kwargs):
    try:
        generate_trajectories(data_dir, num_episodes=2, **kwargs)
        logger.info("simulated 2 episodes in %s", data_dir)
    except Exception:
        logger.exception("trajectory generation failed for %s", data_dir)


# workspace introspection


def get_recap_state(state: AppState):
    """Returns the current state of the RECAP workspace for UI introspection."""
    if not state.recap_workspace or not state.recap_workspace.exists():
        return {"enabled": False}

    workspace = state.recap_workspace
    result = {
        "enabled": True,
        "workspace": str(workspace),
        "pretrained": {
            "actor": (workspace / "pretrained-actor.pt").exists(),
            "critic": (workspace / "pretrained-critic.pt").exists()
        },
        "pretrained_data": None,
        "tasks": []
    }

    pretrained_data_dir = workspace / "pretrained_data"
    if pretrained_data_dir.exists():
        result["pretrained_data"] = {"video_count": _count_videos(pretrained_data_dir)}

    for task_name, task_config in state.recap_config.get("tasks", {}).items():
        task_dir = workspace / task_name
        task_state = {
            "name": task_name,
            "max_episode_length": task_config.get("max_episode_length", 200),
            "exists": task_dir.exists(),
            "iterations": []
        }

        if task_dir.exists():
            # Iterations are numbered folders (0, 1, 2...) holding data folders (data.*)
            for iter_dir in sorted(task_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else -1):
                if not (iter_dir.is_dir() and iter_dir.name.isdigit()):
                    continue

                data_folders = [
                    {"id": d.name, "video_count": _count_videos(d)}
                    for d in sorted(iter_dir.iterdir())
                    if d.is_dir() and not d.name.isdigit()
                ]

                task_state["iterations"].append({
                    "id": int(iter_dir.name),
                    "actor": (iter_dir / "actor.pt").exists(),
                    "critic": (iter_dir / "critic.pt").exists(),
                    "data": data_folders
                })

        result["tasks"].append(task_state)

    return result


# lifecycle steps


def recap_pretrain(state: AppState):
    """Simulates generalist pretraining with a single gradient step on dummy data."""
    workspace = _require_workspace(state)

    actor_path = workspace / "pretrained-actor.pt"
    critic_path = workspace / "pretrained-critic.pt"
    if actor_path.exists():
        raise ApiError("Already pretrained", status_code=409)

    logger.info("pretraining: performing one gradient step on dummy data")

    config = PI_ZERO_CONFIGS["mock"]
    model = SmallPiZero(**config).to(state.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    images = torch.randn(1, 3, config["image_size"], config["image_size"]).to(state.device)
    text = torch.zeros(1, 32, dtype=torch.long).to(state.device)
    internal = torch.randn(1, 32).to(state.device)
    actions = torch.randn(1, 6).to(state.device)

    output = model(images, text, internal, actions)
    loss = output[0] if isinstance(output, tuple) else output
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), str(actor_path))
    torch.save(model.state_dict(), str(critic_path))

    return {"status": "ok"}


def recap_specialize(state: AppState, task_name: str):
    """Creates iteration 0 (SFT) for a specific task."""
    task_dir = _require_task(state, task_name)

    if not (state.recap_workspace / "pretrained-actor.pt").exists():
        raise ApiError("Must pretrain first", status_code=409)

    iter_dir = task_dir / "0"
    iter_dir.mkdir(parents=True, exist_ok=True)
    _save_simulated_weights(iter_dir, task_name, 0)

    return {"status": "ok"}


def recap_collect(state: AppState, task_name: str, iter_id: int):
    """Simulates data collection - creates the next data folder with sample videos."""
    task_dir = _require_task(state, task_name)

    iter_dir = task_dir / str(iter_id)
    if not iter_dir.exists():
        raise ApiError(f"Iteration {iter_id} does not exist for {task_name}", status_code=404)

    next_idx = len(list(iter_dir.glob("data.*")))
    data_dir = iter_dir / f"data.{next_idx}"
    data_dir.mkdir(parents=True, exist_ok=True)

    _generate_episodes(data_dir, steps=20)

    return {"status": "ok", "data_folder": f"data.{next_idx}"}


def simulate_collection(state: AppState, task_name: str, iter_id: int):
    """Simulates collecting a new timestamped batch of data for a task/iteration."""
    task_dir = _require_task(state, task_name)

    data_id = f"data.batch_{int(time.time())}"
    data_dir = task_dir / str(iter_id) / data_id
    data_dir.mkdir(parents=True, exist_ok=True)

    _generate_episodes(data_dir)

    return {"status": "ok", "task_name": task_name, "iter_id": iter_id, "data_id": data_id}


def recap_iterate(state: AppState, task_name: str, iter_id: int):
    """Advances to the next iteration after finetuning on collected data."""
    task_dir = _require_task(state, task_name)

    current_iter_dir = task_dir / str(iter_id)
    if not current_iter_dir.exists():
        raise ApiError(f"Iteration {iter_id} does not exist", status_code=404)

    if not list(current_iter_dir.glob("data.*")):
        raise ApiError("No data collected for this iteration", status_code=409)

    next_iter_id = iter_id + 1
    next_iter_dir = task_dir / str(next_iter_id)
    next_iter_dir.mkdir(parents=True, exist_ok=True)

    _save_simulated_weights(next_iter_dir, task_name, next_iter_id)

    # If policy finetuning just finished, promote its weights to the new iteration
    finetuned_actor = state.recap_workspace / "policy_finetuned" / "actor.pt"
    if finetuned_actor.exists():
        shutil.move(str(finetuned_actor), str(next_iter_dir / "actor.pt"))
        logger.info("moved finetuned actor to %s", next_iter_dir)

    return {"status": "ok", "new_iteration": next_iter_id}
