"""
PPO (Proximal Policy Optimization) from scratch.
Actor-Critic with clipped objective.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import BaseAgent


class ActorCritic(nn.Module):
    """Shared backbone, separate policy and value heads."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> tuple[int, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(-1)


class PPOBuffer:
    """Rollout buffer for PPO (on-policy)."""

    def __init__(self, obs_dim: int, max_len: int):
        self.obs = np.zeros((max_len, obs_dim), dtype=np.float32)
        self.actions = np.zeros(max_len, dtype=np.int64)
        self.rewards = np.zeros(max_len, dtype=np.float32)
        self.log_probs = np.zeros(max_len, dtype=np.float32)
        self.values = np.zeros(max_len, dtype=np.float32)
        self.dones = np.zeros(max_len, dtype=np.float32)
        self.ptr = 0
        self.max_len = max_len

    def push(self, obs: np.ndarray, action: int, reward: float, log_prob: float, value: float, done: bool):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_len

    def get(self) -> dict[str, np.ndarray]:
        n = self.ptr if self.ptr > 0 else self.max_len
        return {
            "obs": self.obs[:n].copy(),
            "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(),
            "log_probs": self.log_probs[:n].copy(),
            "values": self.values[:n].copy(),
            "dones": self.dones[:n].copy(),
        }

    def clear(self):
        self.ptr = 0


class PPOAgent(BaseAgent):
    """PPO agent with clipped objective."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        device: str | None = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = PPOBuffer(obs_dim, rollout_steps)

        self._step_count = 0
        self._last_obs: np.ndarray | None = None
        self._last_action: int = 0
        self._last_log_prob: torch.Tensor | None = None
        self._last_value: torch.Tensor | None = None

    def observe(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        pass  # PPO collects in rollout, not per-step

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_flat = np.asarray(obs, dtype=np.float32).flatten()
        x = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.net.get_action(x, deterministic=deterministic)

        self._last_obs = obs_flat
        self._last_action = action
        self._last_log_prob = log_prob
        self._last_value = value
        return action

    def store_transition(self, reward: float, done: bool) -> None:
        if self._last_obs is not None and self._last_log_prob is not None and self._last_value is not None:
            self.buffer.push(
                self._last_obs,
                self._last_action,
                reward,
                self._last_log_prob.item(),
                self._last_value.item(),
                done,
            )

    def learn(self) -> dict[str, float] | None:
        return None  # PPO learns at end of rollout

    def learn_from_rollout(self, last_obs: np.ndarray, last_done: bool) -> dict[str, float]:
        """Compute GAE, then PPO update."""
        data = self.buffer.get()
        if len(data["obs"]) == 0:
            return {}

        obs = torch.FloatTensor(data["obs"]).to(self.device)
        actions = torch.LongTensor(data["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(data["log_probs"]).to(self.device)
        rewards = data["rewards"]
        dones = data["dones"]

        # GAE
        with torch.no_grad():
            _, values = self.net(obs)
            values = values.squeeze(-1).cpu().numpy()
            last_val = 0.0
            if not last_done:
                x = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
                _, last_val_t = self.net(x)
                last_val = last_val_t.item()
            advantages = np.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_val = last_val
                else:
                    next_val = values[t + 1]
                delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * last_gae * (1 - dones[t])
            returns = advantages + values

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        n_batches = (len(obs) + self.batch_size - 1) // self.batch_size
        indices = np.arange(len(obs))

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                logits, values = self.net(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                values = values.squeeze(-1)

                ratio = (new_log_prob - mb_old_log).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_ret)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.buffer.clear()
        return {"loss": total_loss / (n_batches * self.n_epochs)}

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def reset(self) -> None:
        self._last_obs = None
        self._last_action = 0
        self._last_log_prob = None
        self._last_value = None
