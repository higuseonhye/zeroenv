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
├── LEARNING.md     # Phase-by-phase learning guide
└── docs/           # PHASE1_LEARNING, FAQ
```

## Phase Summary

| Phase | Content |
|-------|---------|
| 1 | GridWorld, Gymnasium API, reward design |
| 2 | DQN (Replay, Target, ε-greedy), Agent base |
| 3 | Training loop, logging, checkpoints |
| 4 | Streamlit dashboard, learning curves, agent replay |
| 5 | PPO (Actor-Critic, GAE, clipped), DQN vs PPO comparison |

## Learning Points

→ **[LEARNING.md](LEARNING.md)** — Phase-by-phase core concepts (MDP, Q-learning, Policy Gradient, PPO)  
→ **[docs/PHASE1_LEARNING.md](docs/PHASE1_LEARNING.md)** — Phase 1 details & FAQ

## Dependencies

- numpy, gymnasium, torch, matplotlib, streamlit
