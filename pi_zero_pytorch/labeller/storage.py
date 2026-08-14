import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
from memmap_replay_buffer import ReplayBuffer

from .state import AppState

logger = logging.getLogger(__name__)

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
    # A `.complete` marker guarantees the extraction finished; otherwise a
    # truncated cache (interrupted run) is rebuilt from scratch.
    complete_marker = cache_path / ".complete"
    if complete_marker.exists() and any(cache_path.glob("*.jpg")):
        return

    shutil.rmtree(cache_path, ignore_errors=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    output_pattern = str(cache_path / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        # Previews don't need full resolution or near-lossless quality;
        # capping at 480px wide cuts extraction time and cache size several-fold.
        "-vf", "scale=min(480\\,iw):-2",
        "-q:v", "5",
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

    if any(cache_path.glob("*.jpg")):
        complete_marker.touch()


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


def _episode_from_name(filename: str):
    """Split a video filename into (episode_name, view_idx).

    Supports:
      - episode_0.0.mp4 / episode_0.1.mp4  -> ("episode_0", 0) / ("episode_0", 1)
      - episode_0.mp4                      -> ("episode_0", 0)
      - data.0/episode_0.0.mp4             -> ("episode_0", 0)  (data.* folder)
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return stem, 0


def _scan_episodes(video_dirs: List[Path]):
    """Discover episodes (multi-view grouped videos) under the given directories."""
    episodes = {}
    for data_dir in video_dirs:
        logger.info("scanning %s for videos", data_dir)
        found_in_dir = False
        for ext in VIDEO_EXTENSIONS:
            for f in sorted(data_dir.rglob(f"*{ext}")):
                found_in_dir = True
                ep_name, view_idx = _episode_from_name(f.name)
                episodes.setdefault(ep_name, {})[view_idx] = f
                logger.debug("found video %s -> episode %s, view %d", f, ep_name, view_idx)

        if not found_in_dir:
            logger.warning("no videos found in %s", data_dir)
    return episodes


def _scan_proprio(state: AppState, video_dirs: List[Path]) -> int:
    """Load proprioception data from .npz files; returns the detected proprio dim."""
    proprio_dim = 0
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
                            logger.info("loaded proprio for %s: shape=%s", ep_name, proprio_array.shape)
                        break
            except Exception:
                logger.exception("failed to load proprio from %s", npz_file)
    logger.info("detected proprio_dim: %d", proprio_dim)
    return proprio_dim


def _store_episode(state: AppState, episodes, ep_name: str, ep_proprio, num_views: int, proprio_dim: int, h, w, c):
    view_paths = [episodes[ep_name].get(v) for v in range(num_views)]
    view_paths = [p if p else view_paths[0] for p in view_paths]

    if state.fast_mock:
        start_mock = time.time()
        with state.replay_buffer.one_episode(task_id=torch.tensor(-1)):
            for t_idx in range(32):
                store_kwargs = dict(
                    images=torch.randint(0, 256, (c, num_views, h, w), dtype=torch.uint8),
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
        logger.info("fast_mock: episode %s stored in %.4fs", ep_name, time.time() - start_mock)
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
                # Store uint8 (0-255) instead of float32: 4x smaller buffers,
                # faster writes; normalized to [0,1] at training time.
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
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

    logger.info("init_replay_buffer started with %s", video_dirs)
    state.conversion_status["is_converting"] = True
    state.conversion_status["progress"] = 0

    try:
        buffer_dir = state.buffer_dir
        if buffer_dir.exists():
            shutil.rmtree(buffer_dir)
        buffer_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(video_dirs, Path):
            video_dirs = [video_dirs]

        episodes = _scan_episodes(video_dirs)
        episode_names = sorted(episodes.keys())

        if not episode_names:
            logger.warning("no valid video files found")
            return

        num_views = 0
        for ep_name in episode_names:
            num_views = max(num_views, max(episodes[ep_name].keys()) + 1)

        num_views = max(1, num_views)
        state.num_views = num_views
        logger.info("detected %d episodes with %d view(s)", len(episode_names), num_views)

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
            logger.warning("could not read any video files")
            return

        for ep_name in episode_names:
            for vf in episodes[ep_name].values():
                max_frames = max(max_frames, get_frame_count(vf))

        fields = dict(
            images=('uint8', (c, num_views, h, w)),
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
            str(buffer_dir),
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

            _store_episode(state, episodes, ep_name, ep_proprio, num_views, proprio_dim, h, w, c)
    except Exception:
        logger.exception("init_replay_buffer failed")
    finally:
        state.conversion_status["is_converting"] = False
        state.conversion_status["progress"] = len(state.video_to_episode) if state.replay_buffer is not None else 0
        logger.info("ReplayBuffer initialization complete")
