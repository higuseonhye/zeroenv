# Phase 1 Learning Points — Environment

Summary of **principles to understand** and **how to apply them to other projects** from an RL learning perspective.  
Reflects perspectives covered in the FAQ.

---

## 1. Why Build the Environment First?

The core RL loop:

```
Agent receives observation (obs) → chooses action → Environment responds (reward, next state)
```

**You cannot design an Agent until the Environment is defined.**  
Observation space, action space, and reward structure all come from the Environment.

→ **When starting a new RL project, always begin with Environment definition.**

---

## 2. MDP — Representing the Environment Mathematically

### 2.1 Why MDP?

To solve an RL problem, you must be able to express "what to optimize" in mathematical form.  
MDP is the standard framework that defines that formulation.

| Element | GridWorld Example | Applying to Other Projects |
|---------|-------------------|----------------------------|
| **State S** | (row, col) | Game: screen pixels, Robot: joint angles |
| **Action A** | {up, right, down, left} | Discrete or Continuous |
| **Transition P** | Deterministic: s' = s + Δ(s,a) | Can be stochastic |
| **Reward R** | +1, -0.01, -1 | Design for your goal |
| **γ (gamma)** | 0.99 | How much to value the future |

### 2.2 Meaning of the Markov Property

**The next state depends only on the current state and action.**

- Distinguish decision memory vs learning memory. (→ FAQ Q1)
- With full observability, everything is in the current state. With partial observability, past (memory) is needed.

→ **When designing a new environment:** First check "Is this problem Markov?"

---

## 3. Reward Design — Reward Shaping

### 3.1 Rewards Determine Learning

The Agent seeks to maximize **cumulative reward**.  
If the reward structure is wrong, it learns unintended behavior.

### 3.2 Basic Design (GridWorld)

```
reward_goal: +1      → "Go here"
reward_step: -0.01   → "Finish quickly"
reward_obstacle: -1  → "Don't do this"
```

→ **When applying:** Use + for goal behavior, - for danger/inefficiency, consider step cost.

### 3.3 Extended Perspectives (→ FAQ Q6)

| Perspective | Summary |
|-------------|---------|
| **Future·Delayed gratification** | γ≈1 + sparse reward to guide toward "distant future" |
| **Quantum jump** | 99 failures + 1 success → large reward. Step function |
| **Diverse rewards** | Design goal, process, curiosity, step cost, failure separately. Minus is not "punishment" but information/cost |
| **Unintended rewards** | Reward hacking: if agent gets reward via unintended path, it will choose that path |

---

## 4. Gymnasium API — Why a Standard Interface?

### 4.1 reset / step / render

```python
obs, info = env.reset(seed=42)
obs, reward, term, trunc, info = env.step(action)
frame = env.render()
```

### 4.2 terminated vs truncated (→ FAQ Q4)

| | terminated | truncated |
|---|------------|-----------|
| **Meaning** | Natural termination within MDP | External limit (timeout, etc.) |
| **Bootstrapping** | Next state value = 0 | Next state value usable |

→ **When applying:** Distinguish terminated/truncated when handling `done` in TD learning.

### 4.3 Why Use Standards

- No need to change algorithm code for each environment.
- **Environments are swappable components.** (→ FAQ Q7) You can replace the grid with CartPole, graphs, etc.

---

## 5. Environment Design·Modification (→ FAQ Q7)

### 5.1 What It Means to Change the Environment

- **Structure**: grid size, obstacles, goal position
- **Rules**: possible actions, transition dynamics
- **Difficulty**: easy map → hard map
- **Environment swap**: grid → CartPole, graph, images, etc.

### 5.2 Implementation (Phase 1 Extension)

| Method | Implementation |
|--------|----------------|
| **Configurable rewards** | `reward_goal`, `reward_step`, `reward_obstacle` params |
| **Curriculum** | `make_gridworld_curriculum(episode)` — grid_size 3→10 gradual |
| **Procedural** | `make_gridworld_procedural(seed)` — different map per seed |
| **Environment swap** | `env = gym.make("CartPole-v1")` etc. to swap env |

```python
# Example
from envs.factories import make_gridworld_curriculum, make_gridworld_procedural

# Curriculum
env = make_gridworld_curriculum(episode=0)   # 3×3
env = make_gridworld_curriculum(episode=200) # 5×5

# Procedural
env = make_gridworld_procedural(seed=42)

# When training
python main.py train --env gridworld --curriculum
```

---

## 6. Implementation Checklist (For Other Projects)

- [ ] **State space**: What is observed? shape, dtype, range?
- [ ] **Action space**: Discrete / Box / MultiDiscrete?
- [ ] **Rewards**: Match the goal? Sparse/dense balance? Explicit reward_goal, reward_step?
- [ ] **Markov**: Is current state sufficient?
- [ ] **terminated vs truncated**: Both needed?
- [ ] **seed**: Reproducible via `reset(seed=)`?
- [ ] **Environment swap**: Following Gymnasium API allows easy swap?
- [ ] **Curriculum/Procedural**: Implement via factory if needed?

---

## 7. Phase 1 Summary

1. **Environment-first**: Environment definition is the starting point of an RL project.
2. **MDP**: Formulate the problem with state, action, transition, reward, γ.
3. **Reward design**: Can design rewards for goal, danger, process, journey satisfaction.
4. **Standard API**: Gymnasium enables algorithm reuse and environment swapping.
5. **terminated/truncated**: Distinguish for bootstrapping and learning stability.
6. **Environment design**: Curriculum, Procedural, Parameterized. Changing the environment is also a strategy when stuck.
7. **Environment swap**: Can replace the grid with a completely different environment.

---

## References

- **FAQ**: [docs/FAQ_PHASE1.md](FAQ_PHASE1.md) — Markov vs memory, cumulative reward, discount γ, future·unintended rewards, Quantum jump, diverse rewards, environment design·swap
- **Code**: `envs/gridworld.py`, `envs/factories.py`
