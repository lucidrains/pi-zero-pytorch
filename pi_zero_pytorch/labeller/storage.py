import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from memmap_replay_buffer import ReplayBuffer

from .state import AppState

VIDEO_EXTENSIONS = [".mp4", ".webm", ".avi", ".mov"]
PROPRIO_KEYS = ['proprio', 'joint_state', 'qpos', 'robot_state', 'state']


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


def extract_frames(video_path: Path, cache_path: Path):
    if cache_path.exists() and any(cache_path.glob("*.jpg")):
        return

    cache_path.mkdir(parents=True, exist_ok=True)

    output_pattern = str(cache_path / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-q:v", "2",
        output_pattern,
        "-hide_banner", "-loglevel", "warning"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(cache_path / f"frame_{count:04d}.jpg"), frame)
            count += 1
        cap.release()


def get_video_path(state: AppState, filename: str) -> Path:
    # Check our explicit mapping first
    if filename in state.video_to_path:
        return state.video_to_path[filename]

    # Check if it's in a subfolder (for RECAP)
    if "/" in filename or "\\" in filename:
        return Path(filename)

    # Fallback to the first video directory if available
    if state.video_dirs:
        return state.video_dirs[0] / filename

    return state.data_dir / filename


def _scan_episodes(state: AppState, video_dirs: List[Path]):
    """Discover episodes (multi-view grouped videos) under the given directories."""
    episodes = {}
    for data_dir in video_dirs:
        print(f"[RECAP] Scanning {data_dir} for videos...")
        found_in_dir = False
        for ext in VIDEO_EXTENSIONS:
            for f in data_dir.rglob(f"*{ext}"):
                found_in_dir = True
                if f.parent.name.startswith("data."):
                    parts = f.name.split('.')
                    if len(parts) >= 3 and parts[-2].isdigit():
                        ep_name = ".".join(parts[:-2])
                        view_idx = int(parts[-2])
                    else:
                        ep_name = f.stem
                        view_idx = 0
                else:
                    ep_name = f.parent.name
                    try:
                        view_idx = int(f.stem)
                    except Exception:
                        view_idx = 0

                if ep_name not in episodes:
                    episodes[ep_name] = {}
                episodes[ep_name][view_idx] = f
                print(f"[RECAP] Found video: {f} -> ep: {ep_name}, view: {view_idx}")

        if not found_in_dir:
            print(f"[RECAP] WARNING: No videos found in {data_dir}")
    return episodes


def _scan_proprio(state: AppState, video_dirs: List[Path]) -> int:
    """Load proprioception data from .npz files; returns the detected proprio dim."""
    proprio_dim = 0
    print("[RECAP] Initializing proprio_dim tracking...")
    for data_dir in video_dirs:
        for npz_file in data_dir.rglob("*.npz"):
            try:
                data = np.load(npz_file, allow_pickle=True)
                ep_name = npz_file.stem  # Use filename as episode name

                # Find a matching proprioception key
                for key in PROPRIO_KEYS:
                    if key in data:
                        proprio_array = data[key]
                        if proprio_array.ndim >= 1:
                            state.video_to_proprio[ep_name] = proprio_array.tolist()
                            if proprio_array.ndim == 1:
                                proprio_dim = max(proprio_dim, 1)
                            else:
                                proprio_dim = max(proprio_dim, proprio_array.shape[-1])
                            print(f"[RECAP] Loaded proprio for {ep_name}: shape={proprio_array.shape}")
                        break
            except Exception as e:
                print(f"[RECAP] Failed to load proprio from {npz_file}: {e}")
    print(f"[RECAP] Final detected proprio_dim: {proprio_dim}")
    return proprio_dim


def _store_episode(state: AppState, episodes, ep_name: str, ep_proprio, num_views: int, proprio_dim: int, max_frames: int, h, w, c):
    view_paths = [episodes[ep_name].get(v) for v in range(num_views)]
    view_paths = [p if p else view_paths[0] for p in view_paths]

    if state.fast_mock:
        start_mock = time.time()
        with state.replay_buffer.one_episode(task_id=torch.tensor(-1)):
            for t_idx in range(32):
                store_kwargs = dict(
                    images=torch.randn(c, num_views, h, w),
                    text=torch.zeros(32, dtype=torch.long),
                    internal=torch.randn(32),
                    actions=torch.randn(16, 6),
                    reward=0.0
                )
                if ep_proprio is not None:
                    # Use actual proprio if available (broadcast/slice as needed)
                    if t_idx < ep_proprio.shape[0]:
                        store_kwargs['proprio'] = ep_proprio[t_idx]
                    else:
                        store_kwargs['proprio'] = torch.zeros(proprio_dim)
                elif proprio_dim > 0:
                    store_kwargs['proprio'] = torch.zeros(proprio_dim)

                state.replay_buffer.store(**store_kwargs)
        print(f"[RECAP] FAST_MOCK: episode {ep_name} done in {time.time() - start_mock:.4f}s")
        return

    caps = [cv2.VideoCapture(str(p)) for p in view_paths]
    with state.replay_buffer.one_episode(task_id=torch.tensor(-1)):
        t_idx = 0
        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frames.append(frame_tensor)
            if len(frames) < len(caps):
                break

            store_kwargs = dict(
                images=torch.stack(frames, dim=1),
                text=torch.randint(0, 100, (32,)),
                internal=torch.randn(32),
                actions=torch.randn(16, 6),
                reward=0.0
            )

            if ep_proprio is not None:
                if t_idx < ep_proprio.shape[0]:
                    store_kwargs['proprio'] = ep_proprio[t_idx]
                else:
                    store_kwargs['proprio'] = ep_proprio[-1]  # repeat last
            elif proprio_dim > 0:
                store_kwargs['proprio'] = torch.zeros(proprio_dim)

            state.replay_buffer.store(**store_kwargs)
            t_idx += 1
    for cap in caps:
        cap.release()


def init_replay_buffer(state: AppState, video_dirs: Union[List[Path], Path]):
    """Build the ReplayBuffer from video files; runs in a background thread."""
    state.video_to_episode.clear()
    state.video_to_path.clear()
    state.video_to_proprio.clear()

    print(f"[RECAP] init_replay_buffer started with {video_dirs}")
    state.conversion_status["is_converting"] = True
    state.conversion_status["progress"] = 0

    try:
        tmp_buffer_dir = Path("tmp/replay_buffer")
        if tmp_buffer_dir.exists():
            shutil.rmtree(tmp_buffer_dir)
        tmp_buffer_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(video_dirs, Path):
            video_dirs = [video_dirs]

        episodes = _scan_episodes(state, video_dirs)
        episode_names = sorted(episodes.keys())

        if not episode_names:
            print("No valid video files found")
            return

        num_views = 0
        for ep_name in episode_names:
            num_views = max(num_views, max(episodes[ep_name].keys()) + 1)

        num_views = max(1, num_views)
        state.num_views = num_views
        print(f"[RECAP] Detected {len(episode_names)} episodes with {num_views} view(s).")

        proprio_dim = _scan_proprio(state, video_dirs)

        h, w, c, max_frames = None, None, None, 0
        for ep_name in episode_names:
            vf = episodes[ep_name][0]
            cap = cv2.VideoCapture(str(vf))
            ret, frame = cap.read()
            cap.release()
            if ret:
                h, w, c = frame.shape
                break

        if h is None:
            print("Could not read any video files")
            return

        for ep_name in episode_names:
            for vf in episodes[ep_name].values():
                max_frames = max(max_frames, get_frame_count(vf))

        fields = dict(
            images=('float', (c, num_views, h, w)),
            text=('int', (32,)),
            internal=('float', (32,)),
            actions=('float', (16, 6)),
            reward='float',
            returns=('float', (), float('nan')),
            value=('float', (), float('nan')),
            advantages=('float', (), float('nan')),
            advantage_ids=('int', (), -1),
            expert_segment='bool'
        )

        if proprio_dim > 0:
            fields['proprio'] = ('float', (proprio_dim,))

        state.replay_buffer = ReplayBuffer(
            str(tmp_buffer_dir),
            max_episodes=len(episode_names),
            max_timesteps=max_frames,
            meta_fields=dict(
                task_id=('int', (), -1),
                fail='bool',
                task_completed=('int', (), -1),
                marked_timestep=('int', (), -1),
                invalidated='bool',
                recap_step=('int', (), -1),
                is_expert_intervention='bool'
            ),
            fields=fields
        )

        state.conversion_status["total"] = len(episode_names)

        for i, ep_name in enumerate(episode_names):
            state.conversion_status["current_video"] = ep_name
            state.conversion_status["progress"] = i

            state.video_to_episode[ep_name] = i
            for view_idx, p in episodes[ep_name].items():
                state.video_to_path[p.name] = p
            state.video_to_path[ep_name] = episodes[ep_name][0]

            # Fetch proprio for this episode
            ep_proprio = state.video_to_proprio.get(ep_name)
            if ep_proprio:
                ep_proprio = torch.tensor(ep_proprio)

            _store_episode(state, episodes, ep_name, ep_proprio, num_views, proprio_dim, max_frames, h, w, c)
    except Exception as e:
        print(f"[RECAP] ERROR in init_replay_buffer: {e}")
        import traceback
        traceback.print_exc()
    finally:
        state.conversion_status["is_converting"] = False
        state.conversion_status["progress"] = len(state.video_to_episode) if state.replay_buffer is not None else 0
        print("ReplayBuffer initialization complete.")
