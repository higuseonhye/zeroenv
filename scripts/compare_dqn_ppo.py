"""
Compare DQN vs PPO on GridWorld.
Run: python scripts/compare_dqn_ppo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from envs.gridworld import GridWorldEnv
from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent
from training.runner import run_training, run_training_ppo
from visualization.renderer import plot_training_curves


def make_env():
    return GridWorldEnv(grid_size=5, obstacles=[(1, 1), (2, 2)], render_mode=None)


def main():
    Path("checkpoints/dqn").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/ppo").mkdir(parents=True, exist_ok=True)

    obs_dim = 4
    n_actions = 4
    n_episodes = 300
    seed = 42

    print("=" * 50)
    print("Training DQN...")
    print("=" * 50)
    dqn = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, buffer_size=5000, batch_size=32)
    dqn_history = run_training(
        env_factory=make_env,
        agent=dqn,
        n_episodes=n_episodes,
        max_steps=100,
        checkpoint_dir="checkpoints/dqn",
        log_interval=20,
        save_interval=200,
        seed=seed,
    )
    dqn.save("checkpoints/dqn/final.pt")

    print("\n" + "=" * 50)
    print("Training PPO...")
    print("=" * 50)
    ppo = PPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        rollout_steps=128,
        batch_size=32,
        n_epochs=4,
    )
    ppo_history = run_training_ppo(
        env_factory=make_env,
        agent=ppo,
        n_episodes=n_episodes,
        max_steps=100,
        checkpoint_dir="checkpoints/ppo",
        log_interval=20,
        save_interval=200,
        seed=seed,
    )
    ppo.save("checkpoints/ppo/final.pt")

    print("\n" + "=" * 50)
    print("Results (last 50 episodes)")
    print("=" * 50)
    dqn_recent = np.mean(dqn_history["episode_rewards"][-50:])
    ppo_recent = np.mean(ppo_history["episode_rewards"][-50:])
    print(f"DQN mean reward: {dqn_recent:.2f}")
    print(f"PPO mean reward: {ppo_recent:.2f}")

    plot_training_curves(dqn_history, save_path="checkpoints/dqn_curves.png")
    plot_training_curves(ppo_history, save_path="checkpoints/ppo_curves.png")
    print("\nSaved training curves to checkpoints/")


if __name__ == "__main__":
    main()
