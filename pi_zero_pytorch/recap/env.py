from __future__ import annotations

from abc import ABC, abstractmethod
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    import gymnasium as gym
except ImportError:
    gym = None


class BaseRecapEnv(ABC):
    """
    Abstract Base Class for RECAP Environment Rollouts.
    """
    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, Dict]:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        pass

    @abstractmethod
    def render(self) -> List[np.ndarray]:
        """Returns list of RGB numpy arrays (one for each viewpoint)."""
        pass

    @abstractmethod
    def close(self):
        pass


class GymRecapEnv(BaseRecapEnv):
    """
    Gymnasium Environment wrapper producing multi-view visual observations
    and sidecar trajectories for RECAP VLA/VAM training.
    """
    def __init__(self, env_name: str = "LunarLander-v3", num_views: int = 2):
        if gym is None:
            raise ImportError("gymnasium package is required for GymRecapEnv")
        self.env_name = env_name
        self.num_views = num_views
        self.env = gym.make(env_name, render_mode="rgb_array")

    def reset(self) -> Tuple[np.ndarray, Dict]:
        return self.env.reset()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        return self.env.step(action)

    def render(self) -> List[np.ndarray]:
        frame_0 = self.env.render()
        frames = [frame_0]
        if self.num_views > 1:
            # Simulate a second camera viewpoint (e.g. horizontal flip or alternate camera angle)
            frame_1 = np.flip(frame_0, axis=1).copy()
            frames.append(frame_1)
            for v in range(2, self.num_views):
                frames.append(frame_0.copy())
        return frames

    def close(self):
        self.env.close()

    def rollout_episode(
        self,
        policy: Optional[torch.nn.Module] = None,
        max_steps: int = 128,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[List[List[np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes a closed-loop policy rollout in the environment.
        Returns:
            multi_view_frames: list of length num_views containing frames per timestep
            observations: (T, obs_dim)
            actions: (T, action_dim)
            rewards: (T,)
        """
        obs, info = self.reset()
        done = False
        step_count = 0

        all_views_frames: List[List[np.ndarray]] = [[] for _ in range(self.num_views)]
        observations_list = []
        actions_list = []
        rewards_list = []

        while not done and step_count < max_steps:
            frames = self.render()
            for v_idx, f in enumerate(frames[:self.num_views]):
                all_views_frames[v_idx].append(f)

            observations_list.append(obs.copy())

            if policy is not None:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    if hasattr(policy, 'act'):
                        action = policy.act(obs_t)
                    else:
                        out = policy(obs_t)
                        action = out.argmax(dim=-1).item() if out.ndim > 1 else out.item()
            else:
                action = self.env.action_space.sample()

            actions_list.append(action)
            obs, reward, terminated, truncated, info = self.step(action)
            rewards_list.append(reward)
            done = terminated or truncated
            step_count += 1

        obs_arr = np.array(observations_list, dtype=np.float32)
        act_arr = np.array(actions_list)
        rew_arr = np.array(rewards_list, dtype=np.float32)

        return all_views_frames, obs_arr, act_arr, rew_arr


def save_rollout_videos(
    output_dir: Path,
    episode_idx: int,
    multi_view_frames: List[List[np.ndarray]],
    fps: float = 30.0
):
    """
    Saves multi-view rollout frames as .mp4 files with episode indexing.
    Format: episode_{episode_idx}.{view_idx}.mp4
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save viewpoints metadata if not present
    meta_path = output_dir / "viewpoints.json"
    if not meta_path.exists():
        viewpoints = {
            "viewpoints": {str(i): f"camera_view_{i}" for i in range(len(multi_view_frames))}
        }
        with open(meta_path, "w") as f:
            json.dump(viewpoints, f, indent=2)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    for v_idx, frames in enumerate(multi_view_frames):
        if not frames:
            continue
        h, w, c = frames[0].shape
        video_path = output_dir / f"episode_{episode_idx}.{v_idx}.mp4"
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
