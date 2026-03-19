"""
ZeroEnv Dashboard — Streamlit web UI for GridWorld.
Run: streamlit run app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np

from envs.gridworld import GridWorldEnv

st.set_page_config(page_title="ZeroEnv", page_icon="🎮", layout="wide")


def load_history(path):
    """Load history from .npy file. Returns (dict or None, error_msg or None)."""
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / path).resolve()
    if not p.exists():
        return None, f"File not found: {p}"
    try:
        data = np.load(str(p), allow_pickle=True)
        out = data.item() if hasattr(data, "item") else dict(data)
        if not isinstance(out, dict):
            return None, f"Not a dict: {type(out)}"
        return out, None
    except Exception as e:
        return None, str(e)


def history_to_dataframe(history):
    """Convert history dict to DataFrame for st.line_chart. Uses Streamlit native chart (no matplotlib)."""
    import pandas as pd
    rewards = history.get("episode_rewards", [])
    losses = history.get("episode_losses", [])
    epsilons = history.get("episode_epsilons", [])
    n = max(len(rewards), len(losses), len(epsilons))
    if n == 0:
        return None
    def pad(x, size):
        return (x + [x[-1] if x else 0] * (size - len(x)))[:size] if len(x) < size else x[:size]
    data = {"episode": list(range(n)), "reward": pad(rewards, n)}
    if losses and any(l > 0 for l in losses):
        data["loss"] = pad(losses, n)
    if epsilons:
        data["epsilon"] = pad(epsilons, n)
    return pd.DataFrame(data)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stMetric { background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%); 
                padding: 1rem; border-radius: 8px; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%); }
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)


def parse_obstacles(text: str, max_row: int, max_col: int) -> list[tuple[int, int]]:
    """Parse 'r,c r,c' or 'r,c' format into list of (row,col) tuples."""
    obstacles = []
    for part in text.strip().split():
        part = part.strip()
        if not part:
            continue
        try:
            r, c = map(int, part.replace(",", " ").split())
            if 0 <= r < max_row and 0 <= c < max_col:
                obstacles.append((r, c))
        except (ValueError, AttributeError):
            pass
    return obstacles


def init_session():
    if "env" not in st.session_state:
        st.session_state.env = None
    if "obs" not in st.session_state:
        st.session_state.obs = None
    if "info" not in st.session_state:
        st.session_state.info = None
    if "total_reward" not in st.session_state:
        st.session_state.total_reward = 0.0
    if "step_count" not in st.session_state:
        st.session_state.step_count = 0
    if "episode_done" not in st.session_state:
        st.session_state.episode_done = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "train_running" not in st.session_state:
        st.session_state.train_running = False


def create_env(grid_size: int, obstacles_text: str, goal: tuple[int, int], start: tuple[int, int]):
    obstacles = parse_obstacles(obstacles_text, grid_size, grid_size)
    return GridWorldEnv(
        grid_size=grid_size,
        obstacles=obstacles if obstacles else None,
        goal=goal,
        start=start,
        render_mode="rgb_array",
    )


def main():
    init_session()

    st.title("🎮 ZeroEnv — RL Dashboard")
    st.caption("GridWorld, training curves, live training, DQN vs PPO comparison")

    tab_grid, tab_curves, tab_compare, tab_train = st.tabs([
        "GridWorld", "📈 Learning Curves", "⚖️ DQN vs PPO", "▶️ Live Training"
    ])

    with st.sidebar:
        st.header("⚙️ Environment Config")
        grid_size = st.slider("Grid size", 3, 12, 5)
        obstacles_text = st.text_area(
            "Obstacles (row,col per line)",
            value="1,1\n2,2",
            height=100,
            help="One per line, e.g. 1,1 or 2 2",
        )
        goal_r = st.number_input("Goal row", 0, grid_size - 1, grid_size - 1)
        goal_c = st.number_input("Goal col", 0, grid_size - 1, grid_size - 1)
        start_r = st.number_input("Start row", 0, grid_size - 1, 0)
        start_c = st.number_input("Start col", 0, grid_size - 1, 0)

        if st.button("🔄 Apply & Reset"):
            try:
                env = create_env(
                    grid_size,
                    obstacles_text,
                    (goal_r, goal_c),
                    (start_r, start_c),
                )
                st.session_state.env = env
                obs, info = env.reset(seed=42)
                st.session_state.obs = obs
                st.session_state.info = info
                st.session_state.total_reward = 0.0
                st.session_state.step_count = 0
                st.session_state.episode_done = False
                st.session_state.history = []
                st.experimental_rerun()
            except Exception as e:
                st.error(str(e))

        st.markdown("---")
        st.header("🎯 Controls")
        action_names = ["⬆️ Up", "➡️ Right", "⬇️ Down", "⬅️ Left"]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset") and st.session_state.env:
                obs, info = st.session_state.env.reset(seed=42)
                st.session_state.obs = obs
                st.session_state.info = info
                st.session_state.total_reward = 0.0
                st.session_state.step_count = 0
                st.session_state.episode_done = False
                st.experimental_rerun()

        with col2:
            run_episode = st.button("▶️ Run Episode")

        st.markdown("---")
        st.header("🤖 Trained Agent")
        checkpoint_path = st.text_input("Checkpoint path", value="checkpoints/final.pt", help="e.g. checkpoints/final.pt")
        agent_algo = st.selectbox("Algorithm", ["dqn", "ppo"], help="Algorithm used to train the checkpoint")
        run_trained = st.button("▶️ Run Trained Agent")

    # Auto-create env if not exists
    if st.session_state.env is None:
        try:
            env = create_env(grid_size, obstacles_text, (goal_r, goal_c), (start_r, start_c))
            st.session_state.env = env
            obs, info = env.reset(seed=42)
            st.session_state.obs = obs
            st.session_state.info = info
        except Exception as e:
            st.error(f"Invalid config: {e}")
            st.stop()

    env = st.session_state.env
    obs = st.session_state.obs
    info = st.session_state.info

    # Run episode (trained agent)
    if run_trained:
        try:
            if agent_algo == "dqn":
                from algorithms.dqn import DQNAgent
                agent = DQNAgent(obs_dim=4, n_actions=4)
                agent.load(checkpoint_path)
                agent.epsilon = 0.0
            else:
                from algorithms.ppo import PPOAgent
                agent = PPOAgent(obs_dim=4, n_actions=4)
                agent.load(checkpoint_path)
            st.session_state.env = create_env(grid_size, obstacles_text, (goal_r, goal_c), (start_r, start_c))
            obs, info = st.session_state.env.reset(seed=42)
            st.session_state.obs = obs
            st.session_state.total_reward = 0.0
            st.session_state.step_count = 0
            st.session_state.history = []
            for _ in range(100):
                action = agent.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = st.session_state.env.step(action)
                st.session_state.total_reward += reward
                st.session_state.step_count += 1
                st.session_state.history.append((obs.copy(), reward, action))
                if terminated or truncated:
                    break
            st.session_state.obs = obs
            st.session_state.episode_done = True
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to load agent: {e}")

    # Run episode (random agent)
    if run_episode and not st.session_state.episode_done:
        with st.spinner("Running episode..."):
            obs, info = env.reset(seed=np.random.randint(0, 10000))
            st.session_state.obs = obs
            st.session_state.info = info
            st.session_state.total_reward = 0.0
            st.session_state.step_count = 0
            st.session_state.history = []

            max_steps = 100
            for _ in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                st.session_state.total_reward += reward
                st.session_state.step_count += 1
                st.session_state.history.append((obs.copy(), reward, action))
                if terminated or truncated:
                    st.session_state.episode_done = True
                    st.session_state.obs = obs
                    break
                st.session_state.obs = obs
        st.experimental_rerun()

    # --- Tab: GridWorld ---
    with tab_grid:
        col_viz, col_info = st.columns([2, 1])

        with col_viz:
            if obs is not None:
                arr = env.render()
                st.image(arr, caption="GridWorld state")

        with col_info:
            st.metric("Total reward", f"{st.session_state.total_reward:.2f}")
            st.metric("Steps", st.session_state.step_count)
            if obs is not None:
                st.write("**Agent:**", f"({obs[0]}, {obs[1]})")
                st.write("**Goal:**", f"({obs[2]}, {obs[3]})")
                st.write("**Status:**", "✅ Done" if st.session_state.episode_done else "🔄 In progress")
            st.markdown("---")
            st.subheader("Step")
            if not st.session_state.episode_done:
                action = st.selectbox("Action", [0, 1, 2, 3], format_func=lambda x: ["Up", "Right", "Down", "Left"][x])
                if st.button("Step"):
                    obs, reward, terminated, truncated, info = env.step(action)
                    st.session_state.obs = obs
                    st.session_state.info = info
                    st.session_state.total_reward += reward
                    st.session_state.step_count += 1
                    if terminated or truncated:
                        st.session_state.episode_done = True
                    st.experimental_rerun()
            else:
                st.info("Episode ended. Reset or Run new episode.")

        if st.session_state.history:
            st.markdown("---")
            st.subheader("📜 Episode replay")
            n = len(st.session_state.history)
            step_slider = st.slider("Replay step", 0, n - 1, n - 1 if st.session_state.episode_done else 0)
            replay_obs, replay_r, replay_a = st.session_state.history[step_slider]
            env._agent_pos = (replay_obs[0], replay_obs[1])
            arr = env.render()
            env._agent_pos = (obs[0], obs[1])
            st.image(arr, width=320, caption=f"Step {step_slider}: reward={replay_r:.2f}")

    # --- Tab: Learning Curves ---
    with tab_curves:
        st.subheader("📈 Learning Curves")
        hist_path = st.text_input("History path", value="checkpoints/history.npy", key="hist_path")
        if st.button("Load", key="load_hist"):
            h, err = load_history(hist_path)
            st.session_state.loaded_history = h
            st.session_state.load_hist_error = err if err else None
        if st.session_state.get("load_hist_error"):
            st.error(st.session_state.load_hist_error)
        if "loaded_history" in st.session_state and st.session_state.loaded_history:
            h = st.session_state.loaded_history
            df = history_to_dataframe(h)
            if df is not None and len(df) > 0:
                st.line_chart(df.set_index("episode"))
            else:
                st.warning("No data to display.")
        else:
            st.info("Enter path (e.g. checkpoints/history.npy) and click Load")

    # --- Tab: DQN vs PPO Comparison ---
    with tab_compare:
        st.subheader("⚖️ DQN vs PPO Comparison")
        c1, c2 = st.columns(2)
        with c1:
            dqn_path = st.text_input("DQN history", value="checkpoints/dqn/history.npy", key="dqn_path")
        with c2:
            ppo_path = st.text_input("PPO history", value="checkpoints/ppo/history.npy", key="ppo_path")
        if st.button("Compare", key="compare_btn"):
            dqn_h, dqn_err = load_history(dqn_path.strip())
            ppo_h, ppo_err = load_history(ppo_path.strip())
            if dqn_h and ppo_h:
                import pandas as pd
                dr = dqn_h.get("episode_rewards", [])
                pr = ppo_h.get("episode_rewards", [])
                n = max(len(dr), len(pr)) or 1
                dr_pad = (dr + [dr[-1]] * (n - len(dr)))[:n] if dr else [0] * n
                pr_pad = (pr + [pr[-1]] * (n - len(pr)))[:n] if pr else [0] * n
                df = pd.DataFrame({"episode": range(n), "DQN": dr_pad, "PPO": pr_pad})
                st.session_state.compare_df = df
                st.session_state.compare_error = None
            else:
                st.session_state.compare_df = None
                errs = [e for e in [dqn_err, ppo_err] if e]
                st.session_state.compare_error = " | ".join(errs) if errs else "Failed to load"
        if st.session_state.get("compare_error"):
            st.error(st.session_state.compare_error)
        if "compare_df" in st.session_state and st.session_state.compare_df is not None:
            st.line_chart(st.session_state.compare_df.set_index("episode"))

    # --- Tab: Live Training ---
    with tab_train:
        st.subheader("▶️ Live Training")
        t1, t2 = st.columns(2)
        with t1:
            train_algo = st.selectbox("Algorithm", ["dqn", "ppo"], key="train_algo")
            train_episodes = st.number_input("Episodes", 10, 500, 100, key="train_ep")
        with t2:
            train_max_steps = st.number_input("Max steps/episode", 50, 500, 100, key="train_ms")
        if st.button("Start Training", key="train_start"):
            st.session_state.train_running = True
        if st.session_state.get("train_running"):
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()
            status_placeholder = st.empty()
            try:
                from envs.gridworld import GridWorldEnv
                from algorithms.dqn import DQNAgent
                from algorithms.ppo import PPOAgent
                from training.runner import run_training, run_training_ppo

                def make_env():
                    return GridWorldEnv(grid_size=5, obstacles=[(1, 1), (2, 2)], render_mode=None)

                obs_dim, n_actions = 4, 4
                rewards = []
                if train_algo == "dqn":
                    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, buffer_size=2000, batch_size=32)
                    for ep in range(train_episodes):
                        env = make_env()
                        obs, _ = env.reset(seed=42 + ep)
                        agent.reset()
                        total = 0.0
                        for _ in range(train_max_steps):
                            action = agent.act(obs)
                            next_obs, r, term, trunc, _ = env.step(action)
                            agent.observe(next_obs, r, term, trunc, {})
                            total += r
                            agent.learn()
                            if term or trunc:
                                break
                            obs = next_obs
                        env.close()
                        rewards.append(total)
                        progress_bar.progress((ep + 1) / train_episodes)
                        chart_placeholder.line_chart(__import__("pandas").DataFrame({"reward": rewards}))
                        status_placeholder.caption(f"Episode {ep+1}/{train_episodes} | Reward: {total:.2f}")
                else:
                    agent = PPOAgent(obs_dim=obs_dim, n_actions=n_actions, rollout_steps=64, batch_size=32)
                    hist = run_training_ppo(make_env, agent, n_episodes=train_episodes, max_steps=train_max_steps,
                        checkpoint_dir="checkpoints", seed=42)
                    rewards = hist["episode_rewards"]
                    progress_bar.progress(1.0)
                    chart_placeholder.line_chart(__import__("pandas").DataFrame({"reward": rewards}))
                    status_placeholder.caption(f"PPO training complete ({len(rewards)} episodes)")
                Path("checkpoints").mkdir(exist_ok=True)
                agent.save("checkpoints/final.pt")
                np.save(Path("checkpoints") / "history.npy", {"episode_rewards": rewards, "episode_losses": [], "episode_epsilons": []}, allow_pickle=True)
                st.session_state.train_running = False
                st.success(f"Training complete! Last 10 episodes avg: {np.mean(rewards[-10:]):.2f}")
            except Exception as e:
                st.error(str(e))
                st.session_state.train_running = False


if __name__ == "__main__":
    main()
