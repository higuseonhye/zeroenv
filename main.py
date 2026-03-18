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
from visualization.renderer import plot_training_curves
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
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    final_path = Path(args.checkpoint_dir) / "final.pt"
    agent.save(str(final_path))
    np.save(Path(args.checkpoint_dir) / "history.npy", history, allow_pickle=True)
    print(f"Saved {final_path}")

    if getattr(args, "plot", False):
        plot_path = Path(args.checkpoint_dir) / "training_curves.png"
        plot_training_curves(history, save_path=str(plot_path))
    return history


def eval_agent(args):
    obs_dim, n_actions = get_env_info(args.env)
    algo = getattr(args, "algo", "dqn")

    if algo == "dqn":
        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(args.checkpoint)
        agent.epsilon = 0.0
    else:
        agent = PPOAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(args.checkpoint)

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
    p_train.add_argument("--plot", action="store_true", help="Save training curves after training")

    # eval
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--env", choices=["gridworld", "cartpole"], default="gridworld")
    p_eval.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--seed", type=int, default=None)

    # compare
    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--episodes", type=int, default=200)
    p_compare.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_agent(args)
    elif args.cmd == "compare":
        compare_agents(args)


def compare_agents(args):
    """Train DQN and PPO, compare results, save curves."""
    from pathlib import Path
    Path("checkpoints/dqn").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/ppo").mkdir(parents=True, exist_ok=True)

    obs_dim, n_actions = 4, 4
    n_episodes = args.episodes
    seed = args.seed

    def make_env():
        return GridWorldEnv(grid_size=5, obstacles=[(1, 1), (2, 2)], render_mode=None)

    print("=" * 50)
    print("Training DQN...")
    print("=" * 50)
    dqn = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, buffer_size=5000, batch_size=32)
    dqn_history = run_training(make_env, dqn, n_episodes=n_episodes, max_steps=100,
        checkpoint_dir="checkpoints/dqn", log_interval=20, save_interval=200, seed=seed)
    dqn.save("checkpoints/dqn/final.pt")
    np.save(Path("checkpoints/dqn") / "history.npy", dqn_history, allow_pickle=True)

    print("\n" + "=" * 50)
    print("Training PPO...")
    print("=" * 50)
    ppo = PPOAgent(obs_dim=obs_dim, n_actions=n_actions, rollout_steps=128, batch_size=32, n_epochs=4)
    ppo_history = run_training_ppo(make_env, ppo, n_episodes=n_episodes, max_steps=100,
        checkpoint_dir="checkpoints/ppo", log_interval=20, save_interval=200, seed=seed)
    ppo.save("checkpoints/ppo/final.pt")
    np.save(Path("checkpoints/ppo") / "history.npy", ppo_history, allow_pickle=True)

    print("\n" + "=" * 50)
    print("Results (last 50 episodes)")
    print("=" * 50)
    dqn_recent = np.mean(dqn_history["episode_rewards"][-50:]) if len(dqn_history["episode_rewards"]) >= 50 else np.mean(dqn_history["episode_rewards"])
    ppo_recent = np.mean(ppo_history["episode_rewards"][-50:]) if len(ppo_history["episode_rewards"]) >= 50 else np.mean(ppo_history["episode_rewards"])
    print(f"DQN mean reward: {dqn_recent:.2f}")
    print(f"PPO mean reward: {ppo_recent:.2f}")

    plot_training_curves(dqn_history, save_path="checkpoints/dqn_curves.png")
    plot_training_curves(ppo_history, save_path="checkpoints/ppo_curves.png")
    # Combined comparison plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    dr = np.array(dqn_history["episode_rewards"])
    pr = np.array(ppo_history["episode_rewards"])
    ax.plot(dr, alpha=0.5, color="blue", label="DQN")
    ax.plot(pr, alpha=0.5, color="orange", label="PPO")
    w = 10
    if len(dr) >= w:
        s = np.convolve(dr, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(dr)), s, color="blue", linewidth=2)
    if len(pr) >= w:
        s = np.convolve(pr, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(pr)), s, color="orange", linewidth=2)
    ax.set_title("DQN vs PPO — Episode Reward")
    ax.set_xlabel("Episode")
    ax.legend()
    plt.tight_layout()
    plt.savefig("checkpoints/compare_curves.png", dpi=150)
    plt.close()
    print("\nSaved training curves to checkpoints/ (dqn_curves, ppo_curves, compare_curves)")


if __name__ == "__main__":
    main()
