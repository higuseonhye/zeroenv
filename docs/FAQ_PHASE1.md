# Phase 1 FAQ — Markov, Reward Design

Frequently asked questions and answers during RL learning.

---

## Q1. Markov — "Isn't knowing past history (observation/action sequence) important in AI? Memory. Why is RL different?"

> **Terminology**: "Past history" is more accurately "past record". In RL we use terms like **observation history**, **action history**. "History" already implies past, so "past history" is redundant.

### Divide "memory" into two types

| Type | Meaning | Example in RL |
|------|---------|---------------|
| **Decision memory** | Past needed to decide "what to do now" | Unnecessary if current observation contains everything |
| **Learning memory** | Store past experience for learning | Replay buffer, experience replay |

RL does use **learning memory** (DQN's replay buffer is a prime example). Markov refers to the **decision** side.

---

### Difference between other AI and RL

**LLM·Sequence models**: "State" is the **entire context so far**. To predict the next token, `[previous tokens]` are needed. Here "state = past sequence", so memory (context) is essential.

**RL MDP**: "State" is the **current snapshot from the environment**. The environment gives "information needed now" at once, e.g. "(position, velocity, goal position)". If this single frame contains everything needed for optimal decisions, past trajectory is unnecessary.

→ Difference: **Who defines state, and how.** LLM: state=past, RL MDP: state=environment's current summary.

---

### Concrete examples

**Chess**: Current board alone is enough to find the best move. "In what order this position was reached" doesn't matter. → Markov, memory unnecessary.

**Poker**: Opponent's cards are hidden. Must remember which cards were played to estimate probabilities. → Partial observability, memory needed.

**GridWorld**: Receives (agent position, goal position) each step. "Came from left or from above" doesn't affect next action choice. → Markov.

---

### When does RL use memory?

1. **Fully observable MDP**: observation = state. All info needed "at this moment" is present. → No memory needed at decision time.
2. **Partially observable POMDP**: Observation alone is insufficient. Must aggregate past observations to estimate "true state". → **Decision memory** like LSTM hidden state, belief state needed.
3. **Learning phase**: Store past (s,a,r,s') for reuse (replay buffer, etc.). → **Learning memory** is always used.

---

### In one line

**Markov = "Can make the best decision with only the information at this moment."**  
If the environment summarizes "this moment" well enough, memory isn't needed. If the summary is insufficient (partial observability), fill it with memory.

---

## Q2. Cumulative Reward — "Isn't it proportional to step count? Like always fighting on a tilted playing field?"

> **Terminology**: **Step count** (episode length) is more accurate than "time". In RL, cumulative reward varies by **how many steps** were taken, not "real time".

### What's correct

The longer the episode (more steps), the more reward accumulates. With -0.01 per step: 100 steps = -1.0, 10 steps = -0.1. Saying "proportional to step count" is correct in that sense.

### But this isn't "unfair" — it's by design

We want **fast goal attainment**. So we design cumulative reward to worsen with more steps.

- Reach in 8 steps: +1 - 0.08 = 0.92
- Reach in 50 steps: +1 - 0.5 = 0.5

Same rules for everyone, but **efficient paths** get higher reward.

### Role of discount γ

We actually use **discounted cumulative reward**:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

With γ < 1, **near future** matters more. Distant future reward shrinks toward 0. So "more steps = always better" is false — **when** you get the reward matters.

### "Tilted playing field" analogy

- All agents evaluated by the **same rules**, so it's "fair".
- But **reward design determines learning direction**, so we design which actions are favored via rewards.
- Step penalty → field tilted toward "finish quickly"
- No penalty → field tilted toward "no rush"

So we **intentionally design a tilted reward structure**.

---

## Q3. Observation vs State — Environment gives "observation", but theory uses "state". What's the difference?

### Definitions

- **State**: Environment's **true full information**. Core concept of MDP theory. Agent may not see it.
- **Observation**: What the agent **actually receives**. The `obs` returned by `env.step()`.

### Relationship

| Situation | Relationship | Example |
|-----------|--------------|---------|
| **Full observability** | observation = state | GridWorld: obs is (position, goal) → that is the state |
| **Partial observability** | observation ⊂ state | Poker: only own cards visible, opponent cards are part of state but not observed |

With full observability, **use obs directly as state.** GridWorld, CartPole do this.  
With partial observability, state can't be known from observation alone; must aggregate history to "estimate state".

### In one line

**Observation = what the agent sees, State = environment's full information.** With full observability they're the same.

---

## Q4. terminated vs truncated — Why separate them?

### Meaning

- **terminated**: Ends naturally **within** the MDP. Goal reached, death, etc.
- **truncated**: Cut off by **external** factor. max_steps exceeded, time limit, etc.

### Difference in Bootstrapping (important for Phase 2 DQN)

When using "next state value" in TD learning:

| | Next state value |
|--|------------------|
| **terminated** | 0 (truly ended, no future) |
| **truncated** | Q(s',a') usable (episode could continue) |

Treating `done = terminated or truncated` and "next value = 0" for both would incorrectly learn 0 for truncated cases. **Distinguishing goal reached vs timeout** is needed for correct learning.

### In one line

**terminated = truly ended, truncated = forced stop.** Whether to use 0 for next value differs.

---

## Q5. Discount γ — With "γ=0.99", how much is a reward 100 steps later worth?

### What the question means

In RL, **cumulative reward** (how "good" the agent is) is not a simple sum:

```
Total value = (current reward) + (next step reward)×γ + (next next reward)×γ² + ...
```

**γ (gamma)** controls "how much to discount future rewards".  
The question asks concretely:

> **"A reward of 1 received 100 steps from now — how much is it worth at the current time?"**

So "reward 100 steps later" = reward at **t+100**,  
"worth" = how much that reward contributes to cumulative value when γ is applied.

### Why discount the future?

- **Current** reward 1 counts as 1.
- **Future** reward is weighted less than 1. (γ < 1)
- So "near reward" matters more than "distant reward".

→ A mechanism to make the agent care more about **near future** than far future.

### Formula

**Current value** of reward r received k steps later = γ^k × r

### With γ=0.99

| Steps | γ^k | Meaning |
|-------|-----|---------|
| 0 | 1.0 | Current reward = 100% |
| 10 | 0.90 | 10 steps later = 90% |
| 50 | 0.61 | 50 steps later = 61% |
| 100 | 0.37 | 100 steps later = 37% |
| 200 | 0.13 | 200 steps later = 13% |

Example: +1 received 100 steps later contributes 0.37 at "now".

### Interpretation

- Smaller γ → focus on **short term**. γ=0.9: 10 steps later already 35% decay.
- γ close to 1 → consider **long term**. γ=0.99: 100 steps later still 37%.
- γ=1: all future rewards equal. With infinite episodes the sum can diverge, so usually γ<1.

### In one line

**γ^k = "current value ratio of reward k steps later".** With γ=0.99, reward 100 steps later is 37% of current value.

---

## Q6. Larger future reward? Unintended reward?

### "Nothing now, but big reward far in the future" — like life

Yes. **Designing for larger future** also makes sense.

**γ > 1?**  
Theoretically possible, but with long episodes the sum **diverges**. So we usually use γ ∈ (0, 1].

**Instead, do this:**
- **γ close to 1** (0.99, 0.999): discount future less → distant future reward still matters
- **Sparse reward**: large reward only on goal. Design like "0 now, +1 once later"
- **Reward design**: put "short-term loss, long-term gain" into the reward structure

→ Like life's "delayed gratification", RL can use γ≈1 + sparse reward to guide "action for distant future".

### "Getting reward in unintended situations"

The agent **maximizes cumulative reward**. So:

- Not the path we **intended**, but
- A path we **missed** that gives reward
- The agent will choose that path.

This is **Reward Hacking**. Examples:
- Goal: "reach goal" → bug lets +1 by stepping on a specific cell
- Goal: "high score" → repeat screen bug to raise score only

**Why does it happen?**  
When the reward function doesn't perfectly reflect "the behavior we want".

**Countermeasures:**
- Review reward design carefully
- Simulate whether unintended reward paths exist
- If needed, use **IRL** or **preference learning** to learn human intent

→ **Reward design determines "what the agent learns."** Design so unintended optima don't arise.

### "99 failures but 1 success gets big reward" — quantum jump, step function

Like human growth, you can put **sudden breakthrough**, **step-wise jump** into rewards.

**Structure examples:**
- 99 failures (reward 0 or small negative) → 100th success **+100**
- "Almost there" +0.5, "fully done" +100 → step function
- By level: level 1 +1, level 2 +10, level 3 +100 → quantum jump

**Meaning:**
- **Sparse + large reward**: flat until one success, then big jump
- **Reward for persistence**: little penalty for failure, large reward for success → "don't give up, keep trying"
- **Growth curve**: initial plateau → breakthrough → next plateau → …

**Implementation:**
- Success reward = base × (1 + bonus) or fixed per level
- "First success bonus": extra reward only on first achievement
- Step function: `if progress >= threshold_k: reward = level_k_reward`

**Caution:**
- Sparse reward → **rare learning signal**; without good exploration, success may never be found
- **Curriculum learning** (easy first), **Reward shaping** (intermediate reward), **Curiosity** etc. needed to "find the path to success"

→ **You can design "sudden breakthrough" into rewards.** But exploration to reach that moment must come first.

### "Different kinds of reward along the way" — minus has meaning, enjoy the journey

Yes. **Multiple reward types** at once is common design.

**Diverse reward types:**
- **Goal reward**: reach goal +1 (ultimate success)
- **Process reward**: discover new area +0.1, find clue +0.05 (small joy)
- **Step cost**: -0.01 (cost of movement — not "bad", just cost)
- **Failure reward**: wall collision -1 (information — "this is not it")
- **Curiosity reward**: first visit +ε (reward for exploration itself)

**When "minus" has meaning:**
- **-0.01 step**: cost of "using time/energy". Natural cost of moving toward the goal.
- **-1 collision**: learning signal "this path is wrong". Failure is also information and part of growth.
- Negative number doesn't mean "bad" — it **informs the result** of the action.

**"Enjoy the journey":**
- **Goal only**: "result matters" → journey can feel like suffering
- **Intermediate rewards**: "discover something new", "getting closer", "fun of exploration" → **process itself has meaning**
- **Intrinsic reward** (curiosity, diversity): small reward for "new experience" regardless of goal → enjoy the journey

**Implementation:**
- Each term designed separately, then summed. Balance "goal vs process" with weights.

→ **You can put small joys of the journey, meaning of failure, fun of exploration into rewards.** Minus is not "punishment" but **information and cost**.

---

## Q7. Designing·Changing the Environment Itself — Does it matter in RL?

Yes. **Environment design** is one of RL's three axes (agent, environment, reward). Changing **environment structure** is as important as changing rewards.

### What changing the environment means

- **Structure**: grid size, obstacle count·position, goal position
- **Rules**: what actions are possible, how transitions work
- **Difficulty**: easy map → hard map
- **New map/level generation**: learn in different environment each time

### How RL reflects this

| Method | Meaning |
|--------|---------|
| **Curriculum Learning** | Easy environment first → gradually harder. 3×3 grid → 5×5 → 10×10 |
| **Procedural Generation** | Random obstacles·goal each episode. Learn in diverse environments → **generalization** |
| **Parameterized Environment** | Take `grid_size`, `n_obstacles` etc. as args to tune environment |
| **Environment redesign** | When learning stalls, consider "environment too hard", simplify map or add intermediate goals |

### Why it matters

- **Agent only**: learns to do better in the same environment
- **Environment**: changes "what environment to learn in" → learning efficiency, generalization, difficulty control change
- **Life analogy**: "change only yourself" vs "change the environment (work, relationships, space)". Both are needed.

### Implementation example (GridWorld)

```python
# Curriculum: difficulty rises with episode count
def make_env(episode: int):
    size = 3 + min(episode // 100, 7)  # 3→10 gradual expansion
    return GridWorldEnv(grid_size=size, obstacles=...)

# Procedural: different map each time
def make_env(seed: int):
    obstacles = random_obstacles(seed)
    return GridWorldEnv(obstacles=obstacles)
```

→ **Environment design is as important as reward design.** When stuck, don't only think "make the agent smarter" — **changing the environment** is also a strategy.

### Can we replace the grid with something completely different?

Yes. **Replace the grid with a different environment entirely.**

- GridWorld → CartPole
- GridWorld → maze, graph, continuous space, image-based game — **completely different structure**

Environments are **swappable components**. As long as `reset()`, `step()`, `observation_space`, `action_space` follow Gymnasium spec, keep agent·training loop and **swap only the environment**.

```python
# Now: GridWorld
env = GridWorldEnv(...)

# Can swap: CartPole
env = gym.make("CartPole-v1")

# Can swap: custom environment
env = MyGraphEnv(...)
```

→ **GridWorld is the Phase 1 example.** You can drop the grid and swap to a completely different environment if needed.

---

## Summary

| Question | Key point |
|----------|-----------|
| **Markov vs memory** | Decision vs learning distinction. LLM: state=past, RL MDP: state=environment summary. Partial observability → fill with memory. |
| **Cumulative reward and step count** | Varying with steps is by design. Discount γ makes near future more important. |
| **Observation vs State** | Observation=what agent sees, State=environment's full info. Full observability → same. |
| **terminated vs truncated** | terminated=truly ended (next value 0), truncated=forced stop (next value usable). Important for bootstrapping. |
| **Discount γ** | γ^k = current value ratio of reward k steps later. γ=0.99 → 100 steps later = 37%. |
| **Future·Unintended reward** | γ≈1 + sparse reward to guide "distant future". Reward hacking: unintended path → agent chooses it. |
| **Quantum jump / step function** | 99 failures + 1 success → large reward. Sparse+large reward guides "breakthrough". Exploration·curriculum must come first. |
| **Diverse rewards, journey satisfaction** | Design goal, process, curiosity, step cost, failure separately. Minus is not "punishment" but information·cost. Intermediate rewards give meaning to the journey. |
| **Environment design·change** | One of RL's three axes. Curriculum, Procedural, Parameterized. When stuck, changing the environment is also a strategy. |
| **Grid → other environment swap** | Yes. Environments are swappable. Match reset/step/space and you can drop the grid for CartPole, graphs, images, etc. |
