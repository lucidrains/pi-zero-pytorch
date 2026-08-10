import shutil

import torch

from .networks import PI_ZERO_CONFIGS, SmallPiZero
from .sim_engine import generate_trajectories
from .state import AppState

VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov"}


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

    # Check for pretrained_data directory
    pretrained_data_dir = workspace / "pretrained_data"
    if pretrained_data_dir.exists():
        video_count = len([f for f in pretrained_data_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS and f.stat().st_size > 0])
        result["pretrained_data"] = {"video_count": video_count}

    for task_name, task_config in state.recap_config.get("tasks", {}).items():
        task_dir = workspace / task_name
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
                            video_count = len([f for f in d.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS and f.stat().st_size > 0])
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

        result["tasks"].append(task_state)

    return result


def recap_pretrain(state: AppState):
    """Simulates generalist pretraining with a single gradient step on dummy data."""
    if not state.recap_workspace:
        return {"error": "RECAP workspace not configured"}

    actor_path = state.recap_workspace / "pretrained-actor.pt"
    critic_path = state.recap_workspace / "pretrained-critic.pt"

    if actor_path.exists():
        return {"error": "Already pretrained"}

    print("Pretraining: performing one gradient step on dummy data...")

    # Use mock config for pretraining speed
    config = PI_ZERO_CONFIGS["mock"]
    model = SmallPiZero(**config).to(state.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Dummy data
    images = torch.randn(1, 3, config["image_size"], config["image_size"]).to(state.device)
    text = torch.zeros(1, 32, dtype=torch.long).to(state.device)
    internal = torch.randn(1, 32).to(state.device)
    actions = torch.randn(1, 6).to(state.device)

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


def recap_specialize(state: AppState, task_name: str):
    """Creates iteration 0 (SFT) for a specific task."""
    if not state.recap_workspace:
        return {"error": "RECAP workspace not configured"}

    if not task_name:
        return {"error": "Missing task_name"}

    # Check that pretrained weights exist
    if not (state.recap_workspace / "pretrained-actor.pt").exists():
        return {"error": "Must pretrain first"}

    # Create iteration 0 (SFT)
    iter_dir = state.recap_workspace / task_name / "0"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Simulate creating specialized weights
    torch.save({"simulated": True, "task": task_name, "iteration": 0}, str(iter_dir / "actor.pt"))
    torch.save({"simulated": True, "task": task_name, "iteration": 0}, str(iter_dir / "critic.pt"))

    return {"status": "ok"}


def recap_collect(state: AppState, task_name: str, iter_id: int):
    """Simulates data collection - creates a new data folder with sample videos."""
    if not state.recap_workspace:
        return {"error": "RECAP workspace not configured"}

    if not task_name:
        return {"error": "Missing task_name"}

    iter_dir = state.recap_workspace / task_name / str(iter_id)
    if not iter_dir.exists():
        return {"error": f"Iteration {iter_id} does not exist for {task_name}"}

    # Find next data folder index
    existing_data = list(iter_dir.glob("data.*"))
    next_idx = len(existing_data)

    data_dir = iter_dir / f"data.{next_idx}"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use simulation engine to generate sample trajectories
    try:
        generate_trajectories(data_dir, num_episodes=2, steps=20)
        print(f"RECAP Collect: Simulated 2 episodes in {data_dir}")
    except Exception as e:
        print(f"RECAP Collect: Simulation failed or skipped: {e}")

    return {"status": "ok", "data_folder": f"data.{next_idx}"}


def recap_iterate(state: AppState, task_name: str, iter_id: int):
    """Advances to the next iteration after finetuning on collected data."""
    if not state.recap_workspace:
        return {"error": "RECAP workspace not configured"}

    if not task_name:
        return {"error": "Missing task_name"}

    current_iter_dir = state.recap_workspace / task_name / str(iter_id)
    if not current_iter_dir.exists():
        return {"error": f"Iteration {iter_id} does not exist"}

    # Check that data was collected
    data_folders = list(current_iter_dir.glob("data.*"))
    if not data_folders:
        return {"error": "No data collected for this iteration"}

    # Create next iteration
    next_iter_id = iter_id + 1
    next_iter_dir = state.recap_workspace / task_name / str(next_iter_id)
    next_iter_dir.mkdir(parents=True, exist_ok=True)

    # If policy fine-tuning just finished, it might have saved weights in a 'policy_finetuned' dir
    # we move it to the next iteration
    finetuned_actor = state.recap_workspace / "policy_finetuned" / "actor.pt"
    if finetuned_actor.exists():
        shutil.move(str(finetuned_actor), str(next_iter_dir / "actor.pt"))
        print(f"Moved finetuned actor to {next_iter_dir}")
    else:
        # Simulate creating updated weights if not existing
        torch.save({"simulated": True, "task": task_name, "iteration": next_iter_id}, str(next_iter_dir / "actor.pt"))

    torch.save({"simulated": True, "task": task_name, "iteration": next_iter_id}, str(next_iter_dir / "critic.pt"))

    return {"status": "ok", "new_iteration": next_iter_id}


def simulate_collection(state: AppState, task_name: str, iter_id: int):
    """Simulates collecting a new batch of data for a task/iteration."""
    if not state.recap_workspace:
        return {"error": "RECAP workspace not configured"}

    if not task_name:
        return {"error": "Missing task_name"}

    # Create target directory
    import time
    timestamp = int(time.time())
    data_id = f"data.batch_{timestamp}"
    target_dir = state.recap_workspace / task_name / str(iter_id) / data_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy 2 random videos from video-rollout as mock data
    try:
        generate_trajectories(target_dir, num_episodes=2)
    except Exception as e:
        print(f"Simulate collection API: failed to generate trajectories: {e}")

    return {"status": "ok", "task_name": task_name, "iter_id": iter_id, "data_id": data_id}
