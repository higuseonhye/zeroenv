"""
Visualization — training curves, agent replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def plot_training_curves(
    history: dict[str, list],
    save_path: str | Path | None = None,
    window: int = 10,
) -> None:
    """Plot reward, loss, epsilon curves. Requires matplotlib."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    rewards = history.get("episode_rewards", [])
    losses = history.get("episode_losses", [])
    epsilons = history.get("episode_epsilons", [])

    def smooth(x: list, w: int) -> np.ndarray:
        if len(x) < w:
            return np.array(x)
        return np.convolve(x, np.ones(w) / w, mode="valid")

    if rewards:
        ax = axes[0]
        ax.plot(rewards, alpha=0.3, color="blue")
        if len(rewards) >= window:
            ax.plot(range(window - 1, len(rewards)), smooth(rewards, window), color="blue", linewidth=2)
        ax.set_title("Episode Reward")
        ax.set_xlabel("Episode")

    if losses and any(l > 0 for l in losses):
        ax = axes[1]
        ax.plot(losses, alpha=0.3, color="orange")
        if len(losses) >= window:
            ax.plot(range(window - 1, len(losses)), smooth(losses, window), color="orange", linewidth=2)
        ax.set_title("Loss")
        ax.set_xlabel("Episode")

    if epsilons:
        ax = axes[2]
        ax.plot(epsilons, color="green")
        ax.set_title("Epsilon")
        ax.set_xlabel("Episode")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def replay_agent(
    env_factory,
    agent: Any,
    n_episodes: int = 3,
    max_steps: int = 100,
    save_dir: str | Path | None = None,
    render_mode: str = "rgb_array",
) -> list[np.ndarray]:
    """Run trained agent and collect frames. Returns list of frame arrays."""
    frames = []
    env = env_factory()
    if hasattr(env, "render_mode"):
        env.render_mode = render_mode

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if hasattr(env, "render") and render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            if terminated or truncated:
                break

    env.close()
    return frames
