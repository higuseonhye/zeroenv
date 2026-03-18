"""
ZeroEnv Dashboard — Streamlit web UI for GridWorld.
Run: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import numpy as np

from envs.gridworld import GridWorldEnv

st.set_page_config(page_title="ZeroEnv", page_icon="🎮", layout="wide")

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

    st.title("🎮 ZeroEnv — GridWorld")
    st.caption("Reinforcement learning environment playground")

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

    # Main content
    col_viz, col_info = st.columns([2, 1])

    with col_viz:
        if obs is not None:
            # Render current state
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

    # Episode replay (if we have history)
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Episode replay")
        n = len(st.session_state.history)
        step_slider = st.slider("Replay step", 0, n - 1, n - 1 if st.session_state.episode_done else 0)
        replay_obs, replay_r, replay_a = st.session_state.history[step_slider]
        # Temporarily set agent pos for render
        env._agent_pos = (replay_obs[0], replay_obs[1])
        arr = env.render()
        env._agent_pos = (obs[0], obs[1])  # restore
        st.image(arr, width=320, caption=f"Step {step_slider}: reward={replay_r:.2f}")


if __name__ == "__main__":
    main()
