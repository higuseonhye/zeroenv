# Phase 1 학습 포인트 — Environment

RL을 공부하는 입장에서 **이해해야 할 원리**와 **다른 프로젝트에 적용하는 방법** 정리.  
FAQ에서 다룬 관점을 반영함.

---

## 1. 왜 Environment를 먼저 만드는가?

강화학습의 핵심 루프:

```
Agent가 관측(obs)을 받음 → 행동(action) 선택 → Environment가 반응(보상, 다음 상태)
```

**Environment가 정의되기 전에는 Agent를 설계할 수 없다.**  
관측 공간, 행동 공간, 보상 구조가 모두 Environment에서 나온다.

→ **새 RL 프로젝트를 시작할 때는 항상 Environment 정의부터.**

---

## 2. MDP — 환경을 수학적으로 표현하기

### 2.1 왜 MDP인가?

RL 문제를 풀려면 "무엇을 최적화하는가"를 수식으로 쓸 수 있어야 한다.  
MDP는 그 수식을 정의하는 표준 틀이다.

| 요소 | GridWorld 예시 | 다른 프로젝트 적용 |
|------|----------------|-------------------|
| **State S** | (row, col) | 게임: 화면 픽셀, 로봇: 관절 각도 |
| **Action A** | {up, right, down, left} | Discrete 또는 Continuous |
| **Transition P** | 확정적: s' = s + Δ(s,a) | 확률적일 수도 있음 |
| **Reward R** | +1, -0.01, -1 | 목표에 맞게 설계 |
| **γ (gamma)** | 0.99 | 미래를 얼마나 중요하게 볼지 |

### 2.2 Markov 성질의 의미

**다음 상태는 현재 상태와 행동에만 의존한다.**

- 결정용 메모리 vs 학습용 메모리 구분. (→ FAQ Q1)
- 완전 관측이면 현재 상태에 다 들어있음. 부분 관측이면 과거(메모리)가 필요.

→ **새 환경을 설계할 때:** "이 문제가 Markov한가?"를 먼저 확인한다.

---

## 3. 보상 설계 — Reward Shaping

### 3.1 보상이 학습을 결정한다

Agent는 **누적 보상**을 최대화하려 한다.  
보상 구조가 잘못되면, 의도와 다른 행동을 학습한다.

### 3.2 기본 설계 (GridWorld)

```
reward_goal: +1      → "여기로 가라"
reward_step: -0.01   → "빨리 끝내라"
reward_obstacle: -1  → "이건 하지 마라"
```

→ **적용 시:** 목표 행동에는 +, 위험/비효율에는 -, 스텝 비용을 고려한다.

### 3.3 확장 관점 (→ FAQ Q6)

| 관점 | 요약 |
|------|------|
| **미래·지연 만족** | γ≈1 + 희소 보상으로 "먼 미래" 유도 |
| **Quantum jump** | 99번 실패 + 1번 성공 시 큰 보상. 계단식 |
| **다양한 보상** | 목표·과정·호기심·스텝비용·실패를 각각 설계. 마이너스는 "벌"이 아니라 정보·비용 |
| **의도치 않은 보상** | Reward hacking: 의도치 않은 경로로 보상 받으면 에이전트가 그걸 택함 |

---

## 4. Gymnasium API — 왜 표준 인터페이스인가?

### 4.1 reset / step / render

```python
obs, info = env.reset(seed=42)
obs, reward, term, trunc, info = env.step(action)
frame = env.render()
```

### 4.2 terminated vs truncated (→ FAQ Q4)

| | terminated | truncated |
|---|------------|-----------|
| **의미** | MDP 내 자연 종료 | MDP 외부 제한 (시간 초과 등) |
| **Bootstrapping** | 다음 상태 가치 = 0 | 다음 상태 가치 사용 가능 |

→ **적용 시:** TD 학습에서 `done` 처리 시 terminated/truncated를 구분한다.

### 4.3 표준을 쓰는 이유

- 알고리즘 코드를 환경에 맞게 바꿀 필요가 없다.
- **환경은 교체 가능한 부품.** (→ FAQ Q7) 그리드를 버리고 CartPole, 그래프 등 완전히 다른 것으로 바꿔도 된다.

---

## 5. 환경 설계·변경 (→ FAQ Q7)

### 5.1 환경을 바꾼다는 것

- **구조**: 그리드 크기, 장애물, 목표 위치
- **규칙**: 가능한 행동, 전이 방식
- **난이도**: 쉬운 맵 → 어려운 맵
- **환경 교체**: 그리드 → CartPole, 그래프, 이미지 등 완전히 다른 것으로 바꾸기

### 5.2 구현 (Phase 1 확장)

| 방법 | 구현 |
|------|------|
| **Configurable rewards** | `reward_goal`, `reward_step`, `reward_obstacle` params |
| **Curriculum** | `make_gridworld_curriculum(episode)` — grid_size 3→10 점진적 |
| **Procedural** | `make_gridworld_procedural(seed)` — seed마다 다른 맵 |
| **환경 교체** | `env = gym.make("CartPole-v1")` 등 다른 env로 교체 |

```python
# 예시
from envs.factories import make_gridworld_curriculum, make_gridworld_procedural

# Curriculum
env = make_gridworld_curriculum(episode=0)   # 3×3
env = make_gridworld_curriculum(episode=200) # 5×5

# Procedural
env = make_gridworld_procedural(seed=42)

# 학습 시
python main.py train --env gridworld --curriculum
```

---

## 6. 구현 시 체크리스트 (다른 프로젝트용)

- [ ] **State space**: 무엇을 관측하는가? shape, dtype, 범위는?
- [ ] **Action space**: Discrete / Box / MultiDiscrete?
- [ ] **보상**: 목표에 맞는가? 희소/밀집 균형은? reward_goal, reward_step 등 명시?
- [ ] **Markov**: 현재 상태만으로 충분한가?
- [ ] **terminated vs truncated**: 둘 다 필요한가?
- [ ] **seed**: `reset(seed=)`로 재현 가능한가?
- [ ] **환경 교체**: Gymnasium API를 따르면 다른 env로 쉽게 교체 가능?
- [ ] **Curriculum/Procedural**: 필요하면 factory로 구현?

---

## 7. Phase 1에서 배운 것 요약

1. **Environment-first**: 환경 정의가 RL 프로젝트의 출발점이다.
2. **MDP**: 상태, 행동, 전이, 보상, γ로 문제를 수식화한다.
3. **보상 설계**: 목표·위험·과정·여정의 행복을 보상으로 설계할 수 있다.
4. **표준 API**: Gymnasium을 따르면 알고리즘 재사용이 쉽고, 환경 교체가 가능하다.
5. **terminated/truncated**: Bootstrapping과 학습 안정성을 위해 구분한다.
6. **환경 설계**: Curriculum, Procedural, Parameterized. 막히면 환경을 바꿔보는 것도 전략이다.
7. **환경 교체**: 그리드를 완전히 다른 환경으로 바꿔도 된다.

---

## 참고

- **FAQ**: [docs/FAQ_PHASE1.md](FAQ_PHASE1.md) — Markov vs 메모리, 누적 보상, 할인 γ, 미래·의도치 않은 보상, Quantum jump, 다양한 보상, 환경 설계·교체
- **코드**: `envs/gridworld.py`, `envs/factories.py`
