# ZeroEnv

Reinforcement learning framework from scratch.  
Phase 1–5: Environment → Agent (DQN/PPO) → Training → Visualization → Algorithm Comparison.

## Quick Start

```bash
pip install -r requirements.txt

# Verify environment
python test_env.py

# Train DQN on GridWorld
python main.py train --env gridworld --algo dqn --episodes 200

# Train PPO
python main.py train --env gridworld --algo ppo --episodes 200

# Compare DQN vs PPO
python main.py compare --episodes 200

# Evaluate trained agent
python main.py eval --env gridworld --checkpoint checkpoints/final.pt --algo dqn

# Streamlit dashboard
streamlit run app.py
```

## Project Structure

```
zeroenv/
├── envs/           # GridWorld, factories (curriculum, procedural)
├── agents/         # Base agent interface
├── algorithms/     # DQN, PPO
├── training/       # Training loops (DQN, PPO)
├── visualization/  # Training curves, agent replay
├── scripts/        # Validation (CartPole, compare)
├── main.py         # CLI: train, eval, compare
├── app.py          # Streamlit dashboard
├── LEARNING.md     # Phase별 학습 가이드
└── docs/           # PHASE1_LEARNING, FAQ
```

## Phase Summary

| Phase | 내용 |
|-------|------|
| 1 | GridWorld, Gymnasium API, 보상 설계 |
| 2 | DQN (Replay, Target, ε-greedy), Agent base |
| 3 | Training loop, 로깅, 체크포인트 |
| 4 | Streamlit 대시보드, 학습 곡선, 에이전트 재생 |
| 5 | PPO (Actor-Critic, GAE, clipped), DQN vs PPO 비교 |

## 학습 포인트

→ **[LEARNING.md](LEARNING.md)** — Phase별 핵심 개념 (MDP, Q-learning, Policy Gradient, PPO)  
→ **[docs/PHASE1_LEARNING.md](docs/PHASE1_LEARNING.md)** — Phase 1 상세 & FAQ

## Dependencies

- numpy, gymnasium, torch, matplotlib, streamlit
