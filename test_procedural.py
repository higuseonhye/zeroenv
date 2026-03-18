"""Procedural GridWorld 테스트."""
from envs.factories import make_gridworld_procedural

env = make_gridworld_procedural(seed=42)
print("Obstacles:", env.obstacles)
print("Grid size:", env.rows, "x", env.cols)

obs, _ = env.reset()
print("Start:", obs[:2], "Goal:", obs[2:])
