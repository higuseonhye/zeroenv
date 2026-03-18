"""Quick sanity check for GridWorld environment."""

from envs.gridworld import GridWorldEnv

env = GridWorldEnv(grid_size=5, obstacles=[(2, 2)], render_mode="rgb_array")
obs, info = env.reset(seed=42)
print("Start:", obs[:2], "Goal:", obs[2:])

# Path avoiding obstacle at (2,2): right 4, down 4
for action in [1, 1, 1, 1, 2, 2, 2, 2]:
    obs, r, term, trunc, _ = env.step(action)
    print(f"Action {action}: pos={obs[:2]}, reward={r}")
    if term:
        print("Reached goal!")
        break
