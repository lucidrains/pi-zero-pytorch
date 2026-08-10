from pi_zero_pytorch.mock import create_mock_replay_buffer


def generate_trajectories(folder, num_episodes=2, steps=128):
    """
    Shim to generate mock trajectories for RECAP testing/demo.
    Useful when the real simulation engine is not available.
    """
    import shutil
    from pathlib import Path

    # Copy all mock videos to simulate a multi-view rollout
    source_dir = Path("video-rollout")
    if source_dir.exists():
        Path(folder).mkdir(parents=True, exist_ok=True)
        # Copy up to num_episodes * 2 (for 2 views each)
        vids = sorted(list(source_dir.glob("episode_*.mp4")))
        for vid in vids[:num_episodes * 2]:
            shutil.copy(vid, Path(folder) / vid.name)

    return create_mock_replay_buffer(
        folder=folder,
        max_episodes=num_episodes,
        max_timesteps=steps,
        num_episodes=num_episodes,
        cleanup_if_exists=False  # don't cleanup, we just copied videos
    )
