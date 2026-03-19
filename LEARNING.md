# ZeroEnv — RL Learning Guide

Phase-by-phase implementation summary and core concepts·theory.

---

## Phase 1 — Environment

### Implementation
- **GridWorld**: 5×5 grid, obstacles, goal
- **Gymnasium interface**: `reset()`, `step()`, `render()`
- **Rewards**: +1(goal), -0.01(step), -1(wall/obstacle)

### Core Concepts

#### 1. Markov Decision Process (MDP)
Basic model for representing the environment mathematically.

- **S (State space)**: All possible states. In GridWorld: `(row, col)` combinations
- **A (Action space)**: Possible actions. Discrete(4) = up, right, down, left
- **P(s'|s,a)**: Transition probability. Probability of reaching s' from s with action a
- **R(s,a,s')**: Reward function
- **γ (gamma)**: Discount factor. Current value of future reward (0~1)

**Markov property**: Next state depends only on **current state and action**. Past history is unnecessary.
→ Thanks to this property, optimal decisions can be made knowing only the state.

#### 2. Reward Design (Reward Shaping)
- **Sparse reward**: +1 only on goal reach → learning can be difficult
- **Dense reward**: Small reward each step → easier learning but may guide wrong behavior
- **GridWorld**: Balances goal(+1) + step(-0.01) + collision(-1)

#### 3. Gymnasium API
- **reset(seed, options)**: Episode start, reproducible initialization
- **step(action)**: Returns `(obs, reward, terminated, truncated, info)`
- **terminated**: End within MDP (e.g. goal reached)
- **truncated**: End by external factor (e.g. time limit)

Reason for distinguishing `terminated` and `truncated`: **Bootstrapping** treats them differently for whether to use future value.

---

## Phase 2 — Agent + Algorithm (DQN)

### Implementation
- **Agent base class**: `observe`, `act`, `learn`
- **DQN**: Replay buffer, Target network, Epsilon-greedy
- **CartPole validation**: Verify algorithm correctness

### Core Concepts

#### 1. Value Function
- **V(s)**: Expected cumulative reward when starting from state s and following policy π
- **Q(s,a)**: Expected cumulative reward when taking action a in state s, then following π

```
V^π(s) = E[ Σ γ^t · r_t | s_0=s, π ]
Q^π(s,a) = E[ Σ γ^t · r_t | s_0=s, a_0=a, π ]
```

**Bellman equation**:
```
Q(s,a) = r + γ · max_a' Q(s', a')
```

#### 2. Q-Learning (Off-policy)
- **Goal**: Estimate optimal Q*
- **Update**: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`
- **Off-policy**: Behavior policy (ε-greedy) differs from learning policy (max)

#### 3. DQN (Deep Q-Network)
Approximate Q with **neural network** instead of table: `Q(s,a; θ)`

- **Replay Buffer**: Store past (s,a,r,s') to break **correlation** and improve sample efficiency via **reuse**
- **Target Network**: Use `Q(s',a'; θ_target)`. If θ keeps changing, target is unstable → periodically copy θ to θ_target
- **Epsilon-greedy**: ε probability random action (exploration), 1-ε optimal action (exploitation). Decrease ε over time.

#### 4. Exploration vs Exploitation
- **Exploitation**: Choose best known action so far
- **Exploration**: Try new actions
- Balance is important. ε-greedy is simple but effective.

---

## Phase 3 — Training Loop

### Implementation
- Episode execution loop
- Logging: reward, loss, epsilon per episode
- Checkpoint save/load

### Core Concepts

#### 1. Episode
- One run from `reset()` until `terminated` or `truncated`
- Environment is initialized each episode

#### 2. Training Loop Structure
```
for episode in range(N):
    obs, info = env.reset()
    while not done:
        action = agent.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        agent.learn(...)
        done = term or trunc
```

#### 3. Importance of Logging
- **Episode reward**: Policy performance metric
- **Loss**: Q learning progress
- **Epsilon**: Exploration level. Approaches 0 when converged

#### 4. Checkpoints
- Resume after interruption
- Keep best model
- Save with hyperparameters for reproducibility

---

## Phase 4 — Visualization

### Implementation
- **Streamlit dashboard**: GridWorld environment config, manual/trained agent execution
- **rgb_array render**: Grid visualization as image without Pygame
- **Matplotlib learning curves**: reward, loss, epsilon
- **Agent replay**: Load checkpoint and replay DQN/PPO

### Core Concepts

#### 1. Role of Visualization
- **Debugging**: Verify environment·agent behavior
- **Analysis**: Judge convergence from learning curves
- **Explanation**: Intuitive understanding of policy behavior

#### 2. Learning Curve Interpretation
- **Reward rise**: Policy improving
- **Oscillation**: Learning rate may be high, or exploration too much
- **Plateau**: May have converged, or stuck in local optimum

---

## Phase 5 — Algorithm Expansion (PPO)

### Implementation
- PPO from scratch
- DQN vs PPO comparison

### Core Concepts

#### 1. Policy Gradient
- **Value-based (DQN)**: Learn Q → indirectly derive policy
- **Policy-based (PPO)**: Directly learn policy π(a|s)

```
∇J(θ) = E[ ∇log π(a|s;θ) · A(s,a) ]
```
A(s,a): Advantage. How much better this action is than average.

#### 2. PPO (Proximal Policy Optimization)
- **Clipped objective**: Keep policy update small for stability
- **On-policy**: Use only data collected by current policy
- **Actor-Critic**: Actor(policy) + Critic(value) learned together

#### 3. DQN vs PPO Comparison
| | DQN | PPO |
|---|-----|-----|
| Action space | Discrete | Discrete + Continuous |
| Sample efficiency | High via Replay | Relatively lower (on-policy) |
| Stability | Needs Target net etc. | Relatively stable with clipping |
| Application | Atari, GridWorld | Robot control, games, etc. |

---

## Overall Learning Order Summary

1. **Phase 1**: MDP, rewards, environment interface
2. **Phase 2**: Value function, Q-learning, DQN (Replay, Target, ε-greedy)
3. **Phase 3**: Episode, training loop, logging
4. **Phase 4**: Visualization and analysis
5. **Phase 5**: Policy gradient, PPO, algorithm comparison
