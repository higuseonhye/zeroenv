"""
ZeroEnv CLI — train and evaluate agents.
Usage:
  python main.py train --env gridworld --algo dqn --episodes 500
  python main.py train --env cartpole --algo dqn --episodes 300
  python main.py eval --env gridworld --checkpoint checkpoints/checkpoint_ep500.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from envs.gridworld import GridWorldEnv
from envs.factories import make_gridworld_curriculum, make_gridworld_procedural
from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent
from training.runner import run_training, run_training_ppo


def make_gridworld():
    return GridWorldEnv(grid_size=5, obstacles=[(1, 1), (2, 2)], render_mode=None)


def make_cartpole():
    return gym.make("CartPole-v1")


def get_env_info(env_name: str) -> tuple[int, int]:
    if env_name == "gridworld":
        env = make_gridworld()
    else:
        env = make_cartpole()
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    env.close()
    return obs_dim, n_actions


def train(args):
    obs_dim, n_actions = get_env_info(args.env)
    env_factory = make_gridworld if args.env == "gridworld" else make_cartpole

    if args.algo == "dqn":
        agent = DQNAgent(
            obs_dim=obs_dim,
            n_actions=n_actions,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update=100,
        )
        run_fn = run_training
    else:
        agent = PPOAgent(
            obs_dim=obs_dim,
            n_actions=n_actions,
            rollout_steps=256 if args.env == "gridworld" else 2048,
            batch_size=64,
        )
        run_fn = run_training_ppo

    run_kwargs = dict(
        env_factory=env_factory,
        agent=agent,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    if args.algo == "dqn" and getattr(args, "curriculum", False) and args.env == "gridworld":
        run_kwargs["curriculum_factory"] = lambda ep: make_gridworld_curriculum(ep, render_mode=None)

    history = run_fn(**run_kwargs)

    # Save final model and history
    final_path = Path(args.checkpoint_dir) / "final.pt"
    agent.save(str(final_path))
    np.save(Path(args.checkpoint_dir) / "history.npy", history, allow_pickle=True)
    print(f"Saved {final_path}")
    return history


def eval_agent(args):
    obs_dim, n_actions = get_env_info(args.env)
    # Assume DQN for eval (PPO eval would need separate logic)
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)
    agent.load(args.checkpoint)
    agent.epsilon = 0.0  # deterministic

    env_factory = make_gridworld if args.env == "gridworld" else make_cartpole
    env = env_factory()

    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed)
        total = 0.0
        for _ in range(500):
            action = agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        rewards.append(total)
        print(f"Episode {ep+1}: reward={total:.2f}")

    env.close()
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--env", choices=["gridworld", "cartpole"], default="gridworld")
    p_train.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    p_train.add_argument("--episodes", type=int, default=500)
    p_train.add_argument("--max-steps", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--gamma", type=float, default=0.99)
    p_train.add_argument("--checkpoint-dir", default="checkpoints")
    p_train.add_argument("--log-interval", type=int, default=10)
    p_train.add_argument("--save-interval", type=int, default=100)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--curriculum", action="store_true", help="Curriculum learning (gridworld only): easy→hard")

    # eval
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--env", choices=["gridworld", "cartpole"], default="gridworld")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    else:
        eval_agent(args)


if __name__ == "__main__":
    main()
