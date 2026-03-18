"""
DQN (Deep Q-Network) from scratch.
Replay buffer, Target network, Epsilon-greedy.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import BaseAgent


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """Q-network: obs -> Q-values per action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    """DQN agent with replay buffer, target network, epsilon-greedy."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 100,
        device: str | None = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQN(obs_dim, n_actions).to(self.device)
        self.target_net = DQN(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self._step_count = 0
        self._last_obs: np.ndarray | None = None
        self._last_action: int | None = None

    def observe(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        next_obs = np.asarray(obs, dtype=np.float32).flatten()
        done = terminated or truncated

        if self._last_obs is not None and self._last_action is not None:
            self.buffer.push(self._last_obs, self._last_action, reward, next_obs, done)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_flat = np.asarray(obs, dtype=np.float32).flatten()
        obs_batch = obs_flat.reshape(1, -1)

        if deterministic or random.random() >= self.epsilon:
            with torch.no_grad():
                x = torch.FloatTensor(obs_batch).to(self.device)
                q = self.q_net(x)
                action = q.argmax(dim=-1).item()
        else:
            action = random.randint(0, self.n_actions - 1)

        self._last_obs = obs_flat.copy()
        self._last_action = action
        return action

    def learn(self) -> dict[str, float] | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def save(self, path: str) -> None:
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data["q_net"])
        self.target_net.load_state_dict(data["target_net"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        if "epsilon" in data:
            self.epsilon = data["epsilon"]

    def reset(self) -> None:
        self._last_obs = None
        self._last_action = None
