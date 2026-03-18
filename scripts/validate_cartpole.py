"""
Validate DQN on CartPole (Gymnasium).
Run: python scripts/validate_cartpole.py
Target: mean reward > 100 over 10 eval episodes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np

from algorithms.dqn import DQNAgent
from training.runner import run_training


def main():
    env = gym.make("CartPole-v1")
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    env.close()

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=100,
    )

    print("Training DQN on CartPole for 300 episodes...")
    history = run_training(
        env_factory=lambda: gym.make("CartPole-v1"),
        agent=agent,
        n_episodes=300,
        max_steps=500,
        log_interval=50,
        save_interval=300,
        seed=42,
    )

    # Eval
    agent.epsilon = 0.0
    env = gym.make("CartPole-v1")
    rewards = []
    for ep in range(10):
        obs, _ = env.reset(seed=42 + ep)
        total = 0.0
        for _ in range(500):
            action = agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        rewards.append(total)
    env.close()

    mean_r = np.mean(rewards)
    print(f"\nEval (10 episodes): mean reward = {mean_r:.1f}")
    if mean_r >= 100:
        print("PASS: DQN works on CartPole")
    else:
        print("Note: May need more episodes. CartPole-v1 is solved at 195 avg reward.")


if __name__ == "__main__":
    main()
