"""
Training loop — episode runner, logging, checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agents.base import BaseAgent


def run_training_ppo(
    env_factory: Callable[[], Any],
    agent: Any,
    n_episodes: int = 500,
    max_steps: int = 200,
    checkpoint_dir: str = "checkpoints",
    log_interval: int = 10,
    save_interval: int = 100,
    seed: int | None = None,
) -> dict[str, list]:
    """PPO training: collect rollout, then learn."""
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)
    env = env_factory()
    rollout_steps = getattr(agent, "rollout_steps", 2048)

    episode_rewards: list[float] = []
    episode_losses: list[float] = []
    total_steps = 0

    obs, info = env.reset(seed=seed)
    agent.reset()
    ep_reward = 0.0
    ep_count = 0
    last_done = False

    while ep_count < n_episodes:
        for _ in range(rollout_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(reward, done)
            ep_reward += reward
            total_steps += 1

            if done:
                ep_count += 1
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, info = env.reset(seed=seed + ep_count if seed else None)
                last_done = True
            else:
                obs = next_obs
                last_done = False

        metrics = agent.learn_from_rollout(obs, last_done)
        episode_losses.append(metrics.get("loss", 0.0))

        if ep_count > 0 and ep_count % log_interval == 0:
            recent = episode_rewards[-log_interval:]
            print(f"Episode {ep_count}/{n_episodes} | Reward: {np.mean(recent):.2f} | Loss: {episode_losses[-1]:.4f}")

        if ep_count > 0 and ep_count % save_interval == 0 and hasattr(agent, "save"):
            path = Path(checkpoint_dir) / f"ppo_checkpoint_ep{ep_count}.pt"
            agent.save(str(path))
            print(f"Saved {path}")

    env.close()
    return {
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "episode_epsilons": [0.0] * len(episode_rewards),
    }


def run_training(
    env_factory: Callable[[], Any],
    agent: BaseAgent,
    n_episodes: int = 500,
    max_steps: int = 200,
    checkpoint_dir: str = "checkpoints",
    log_interval: int = 10,
    save_interval: int = 100,
    seed: int | None = None,
    curriculum_factory: Callable[[int], Any] | None = None,
) -> dict[str, list]:
    """
    Run training loop.
    curriculum_factory: If provided, create new env each episode (for Curriculum Learning).
    Returns dict with episode_rewards, losses, epsilons.
    """
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)
    env = None

    episode_rewards: list[float] = []
    episode_losses: list[float] = []
    episode_epsilons: list[float] = []

    for ep in range(n_episodes):
        if curriculum_factory is not None:
            if env is not None:
                env.close()
            env = curriculum_factory(ep)
        elif env is None:
            env = env_factory()

        obs, info = env.reset(seed=seed)
        agent.reset()
        total_reward = 0.0
        ep_losses: list[float] = []

        for step in range(max_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.observe(next_obs, reward, terminated, truncated, info)

            total_reward += reward
            metrics = agent.learn()
            if metrics and "loss" in metrics:
                ep_losses.append(metrics["loss"])

            if terminated or truncated:
                break
            obs = next_obs

        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        if hasattr(agent, "epsilon"):
            episode_epsilons.append(agent.epsilon)
        else:
            episode_epsilons.append(0.0)

        if (ep + 1) % log_interval == 0:
            avg_r = np.mean(episode_rewards[-log_interval:])
            avg_l = np.mean(episode_losses[-log_interval:]) if episode_losses[-1] else 0
            print(f"Episode {ep+1}/{n_episodes} | Reward: {avg_r:.2f} | Loss: {avg_l:.4f} | ε: {episode_epsilons[-1]:.3f}")

        if (ep + 1) % save_interval == 0 and hasattr(agent, "save"):
            path = Path(checkpoint_dir) / f"checkpoint_ep{ep+1}.pt"
            agent.save(str(path))
            print(f"Saved {path}")

    if env is not None:
        env.close()
    return {
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "episode_epsilons": episode_epsilons,
    }
