"""
Environment factories — Curriculum, Procedural, 환경 교체.
Phase 1 확장: FAQ 관점 반영.
"""

from __future__ import annotations

import numpy as np

from envs.gridworld import GridWorldEnv


def make_gridworld_basic(**kwargs) -> GridWorldEnv:
    """기본 GridWorld. 기존 동작과 동일."""
    return GridWorldEnv(
        grid_size=5,
        obstacles=[(1, 1), (2, 2)],
        **kwargs,
    )


def make_gridworld_curriculum(episode: int, **kwargs) -> GridWorldEnv:
    """
    Curriculum: 에피소드 수에 따라 난이도 상승.
    grid_size 3 → 5 → 7 → ... 점진적 확대.
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
    Procedural: seed마다 다른 맵 생성.
    매번 다른 장애물 배치 → 다양한 환경에서 학습 → 일반화.
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
