"""
Environment factories — Curriculum, Procedural, environment swap.
Phase 1 extension: FAQ perspectives reflected.
"""

from __future__ import annotations

import numpy as np

from envs.gridworld import GridWorldEnv


def make_gridworld_basic(**kwargs) -> GridWorldEnv:
    """Basic GridWorld. Same behavior as original."""
    return GridWorldEnv(
        grid_size=5,
        obstacles=[(1, 1), (2, 2)],
        **kwargs,
    )


def make_gridworld_curriculum(episode: int, **kwargs) -> GridWorldEnv:
    """
    Curriculum: difficulty increases with episode count.
    grid_size 3 → 5 → 7 → ... gradual expansion.
    """
    size = 3 + min(episode // 100, 7)  # 3~10
    obstacles = [(size // 2, size // 2)] if size >= 3 else []
    return GridWorldEnv(
        grid_size=size,
        obstacles=obstacles,
        **kwargs,
    )


def make_gridworld_procedural(seed: int, grid_size: int = 5, obstacle_count: int = 3, **kwargs) -> GridWorldEnv:
    """
    Procedural: different map per seed.
    Different obstacle layout each time → learn in diverse environments → generalization.
    """
    rng = np.random.default_rng(seed)
    rows = cols = grid_size
    obstacles = []
    start, goal = (0, 0), (rows - 1, cols - 1)
    while len(obstacles) < obstacle_count:
        r, c = int(rng.integers(0, rows)), int(rng.integers(0, cols))
        pos = (r, c)
        if pos != start and pos != goal and pos not in obstacles:
            obstacles.append(pos)
    return GridWorldEnv(
        grid_size=grid_size,
        obstacles=obstacles,
        **kwargs,
    )
