"""
Custom GridWorld environment — Gymnasium-compatible.
Configurable grid size, obstacles, goal position.
Reward: +1 goal, -0.01 step penalty, -1 wall/obstacle.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
from typing import Any

# Cell types for internal grid representation
EMPTY = 0
AGENT = 1
GOAL = 2
OBSTACLE = 3


class GridWorldEnv(gym.Env):
    """
    A configurable GridWorld environment compatible with Gymnasium.
    
    Actions: 0=up, 1=right, 2=down, 3=left
    Observation: [agent_row, agent_col, goal_row, goal_col]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: tuple[int, int] | int = 5,
        obstacles: list[tuple[int, int]] | None = None,
        goal: tuple[int, int] | None = None,
        start: tuple[int, int] | None = None,
        render_mode: str | None = None,
        max_steps: int | None = None,
    ):
        """
        Args:
            grid_size: (rows, cols) or single int for square grid. Default 5x5.
            obstacles: List of (row, col) obstacle positions.
            goal: (row, col) goal position. Default: bottom-right.
            start: (row, col) start position. Default: top-left.
            render_mode: "human" or "rgb_array".
            max_steps: Max steps per episode. None = unlimited.
        """
        super().__init__()

        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.grid_size = grid_size
        self.rows, self.cols = grid_size

        self.obstacles = set(obstacles or [])
        self._goal = goal if goal is not None else (self.rows - 1, self.cols - 1)
        self._start = start if start is not None else (0, 0)
        self.max_steps = max_steps if max_steps is not None else self.rows * self.cols * 4

        # Validate
        self._validate_positions()

        # Spaces
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.rows, self.cols) - 1,
            shape=(4,),
            dtype=np.int32,
        )

        self.render_mode = render_mode
        self._window = None
        self._clock = None

        # State
        self._agent_pos: tuple[int, int] = (0, 0)
        self._step_count = 0

    def _validate_positions(self) -> None:
        """Ensure start, goal, and obstacles are valid and non-overlapping."""
        r, c = self.rows, self.cols
        valid = lambda p: 0 <= p[0] < r and 0 <= p[1] < c

        if not valid(self._start):
            raise ValueError(f"Invalid start {self._start} for grid {self.grid_size}")
        if not valid(self._goal):
            raise ValueError(f"Invalid goal {self._goal} for grid {self.grid_size}")
        for obs in self.obstacles:
            if not valid(obs):
                raise ValueError(f"Invalid obstacle {obs} for grid {self.grid_size}")

        if self._start == self._goal:
            raise ValueError("Start and goal cannot be the same")
        if self._start in self.obstacles:
            raise ValueError("Start cannot be on obstacle")
        if self._goal in self.obstacles:
            raise ValueError("Goal cannot be on obstacle")

    def _get_obs(self) -> ObsType:
        return np.array(
            [self._agent_pos[0], self._agent_pos[1], self._goal[0], self._goal[1]],
            dtype=np.int32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {"agent_pos": self._agent_pos, "goal": self._goal}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self._agent_pos = self._start
        self._step_count = 0
        return self._get_obs(), self._get_info()

    def _move(self, action: int) -> tuple[int, int]:
        """Return new (row, col) after action. Does not validate."""
        dr = [-1, 0, 1, 0]   # up, right, down, left
        dc = [0, 1, 0, -1]
        r, c = self._agent_pos
        return r + dr[action], c + dc[action]

    def step(
        self, action: int
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        new_r, new_c = self._move(action)
        reward = -0.01  # step penalty
        terminated = False
        truncated = False

        # Out of bounds or obstacle
        if (
            new_r < 0
            or new_r >= self.rows
            or new_c < 0
            or new_c >= self.cols
            or (new_r, new_c) in self.obstacles
        ):
            reward = -1.0
            # Stay in place
        else:
            self._agent_pos = (new_r, new_c)
            if self._agent_pos == self._goal:
                reward = 1.0
                terminated = True

        self._step_count += 1
        if self._step_count >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "human":
            return self._render_human()
        return None

    def _render_rgb_array(self) -> np.ndarray:
        """Return (H, W, 3) RGB array. Cell size 64px."""
        cell_size = 64
        h, w = self.rows * cell_size, self.cols * cell_size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Colors (R, G, B)
        colors = {
            EMPTY: (240, 240, 240),
            AGENT: (70, 130, 180),   # steel blue
            GOAL: (34, 139, 34),     # forest green
            OBSTACLE: (128, 128, 128),
        }

        for row in range(self.rows):
            for col in range(self.cols):
                y1, y2 = row * cell_size, (row + 1) * cell_size
                x1, x2 = col * cell_size, (col + 1) * cell_size

                if (row, col) == self._agent_pos:
                    cell = AGENT
                elif (row, col) == self._goal:
                    cell = GOAL
                elif (row, col) in self.obstacles:
                    cell = OBSTACLE
                else:
                    cell = EMPTY

                img[y1:y2, x1:x2] = colors[cell]

        # Grid lines (internal boundaries only to avoid index errors)
        for i in range(1, self.rows):
            img[i * cell_size, :] = 0
        for j in range(1, self.cols):
            img[:, j * cell_size] = 0

        return img

    def _render_human(self) -> None:
        """Render in a Pygame window."""
        try:
            import pygame
        except ImportError:
            raise ImportError("Pygame required for human render mode. pip install pygame")

        rgb = self._render_rgb_array()
        h, w = rgb.shape[0], rgb.shape[1]

        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode((w, h))
            self._clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        self._window.blit(surf, (0, 0))
        pygame.event.pump()
        self._clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self) -> None:
        if self._window is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._window = None
            self._clock = None
