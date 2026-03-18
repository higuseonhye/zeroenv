# ZeroEnv — RL 학습 가이드

Phase별 구현 내용과 핵심 개념·이론 정리.

---

## Phase 1 — Environment

### 구현 내용
- **GridWorld**: 5×5 그리드, 장애물, 목표
- **Gymnasium 인터페이스**: `reset()`, `step()`, `render()`
- **보상**: +1(목표), -0.01(스텝), -1(벽/장애물)

### 핵심 개념

#### 1. Markov Decision Process (MDP)
환경을 수학적으로 표현하는 기본 모델.

- **S (State space)**: 가능한 모든 상태. GridWorld에서는 `(row, col)` 조합
- **A (Action space)**: 가능한 행동. Discrete(4) = up, right, down, left
- **P(s'|s,a)**: 전이 확률. 상태 s에서 행동 a를 했을 때 s'로 갈 확률
- **R(s,a,s')**: 보상 함수
- **γ (gamma)**: 할인율. 미래 보상의 현재 가치 (0~1)

**Markov 성질**: 다음 상태는 **현재 상태와 행동**에만 의존. 과거 이력은 불필요.
→ 이 성질 덕분에 상태만 알면 최적 결정을 내릴 수 있음.

#### 2. 보상 설계 (Reward Shaping)
- **희소 보상 (Sparse)**: 목표 도달 시에만 +1 → 학습이 어려울 수 있음
- **밀집 보상 (Dense)**: 스텝마다 작은 보상 → 학습은 쉽지만 잘못된 행동을 유도할 수 있음
- **GridWorld**: 목표(+1) + 스텝(-0.01) + 충돌(-1)로 균형을 맞춤

#### 3. Gymnasium API
- **reset(seed, options)**: 에피소드 시작, 재현 가능한 초기화
- **step(action)**: `(obs, reward, terminated, truncated, info)` 반환
- **terminated**: 목표 도달 등 MDP 내 종료
- **truncated**: 시간 제한 등 MDP 외부 요인으로 인한 종료

`terminated`와 `truncated`를 구분하는 이유: **Bootstrapping**에서 미래 가치를 사용할지 여부를 다르게 처리하기 위함.

---

## Phase 2 — Agent + Algorithm (DQN)

### 구현 내용
- **Agent base class**: `observe`, `act`, `learn`
- **DQN**: Replay buffer, Target network, Epsilon-greedy
- **CartPole 검증**: 알고리즘 정확성 확인

### 핵심 개념

#### 1. Value Function (가치 함수)
- **V(s)**: 상태 s에서 시작해 정책 π를 따를 때 기대 누적 보상
- **Q(s,a)**: 상태 s에서 행동 a를 하고, 이후 π를 따를 때 기대 누적 보상

```
V^π(s) = E[ Σ γ^t · r_t | s_0=s, π ]
Q^π(s,a) = E[ Σ γ^t · r_t | s_0=s, a_0=a, π ]
```

**Bellman 방정식**:
```
Q(s,a) = r + γ · max_a' Q(s', a')
```

#### 2. Q-Learning (Off-policy)
- **목표**: 최적 Q*를 추정
- **업데이트**: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`
- **Off-policy**: 행동 정책(ε-greedy)과 학습 정책(max)이 다름

#### 3. DQN (Deep Q-Network)
Q를 테이블이 아닌 **신경망**으로 근사: `Q(s,a; θ)`

- **Replay Buffer**: 과거 (s,a,r,s')를 저장해 **상관관계**를 깨고, **재사용**으로 샘플 효율 향상
- **Target Network**: `Q(s',a'; θ_target)` 사용. θ가 계속 변하면 목표가 흔들려 학습 불안정 → θ_target을 주기적으로 θ로 복사
- **Epsilon-greedy**: ε 확률로 랜덤 행동(탐험), 1-ε로 최적 행동(활용). ε를 시간에 따라 감소시킴.

#### 4. Exploration vs Exploitation
- **Exploitation**: 현재까지 알려진 최선의 행동 선택
- **Exploration**: 새로운 행동 시도
- 둘의 균형이 중요. ε-greedy는 단순하지만 효과적인 방법.

---

## Phase 3 — Training Loop

### 구현 내용
- 에피소드 실행 루프
- 로깅: reward, loss, epsilon per episode
- 체크포인트 저장/로드

### 핵심 개념

#### 1. Episode (에피소드)
- `reset()`부터 `terminated` 또는 `truncated`까지의 한 번의 시행
- 에피소드마다 환경이 초기화됨

#### 2. Training Loop 구조
```
for episode in range(N):
    obs, info = env.reset()
    while not done:
        action = agent.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        agent.learn(...)
        done = term or trunc
```

#### 3. 로깅의 중요성
- **Episode reward**: 정책 성능 지표
- **Loss**: Q 학습 진행 상황
- **Epsilon**: 탐험 정도. 수렴 시 0에 가까워짐

#### 4. 체크포인트
- 학습 중단 후 재개
- 최고 성능 모델 보관
- 하이퍼파라미터와 함께 저장하면 재현 가능

---

## Phase 4 — Visualization

### 구현 내용
- **Streamlit 대시보드**: GridWorld 환경 설정, 수동/학습된 에이전트 실행
- **rgb_array 렌더**: Pygame 없이 이미지로 그리드 시각화
- **Matplotlib 학습 곡선**: reward, loss, epsilon
- **에이전트 재생**: 체크포인트 로드 후 DQN/PPO 재생

### 핵심 개념

#### 1. 시각화의 역할
- **디버깅**: 환경·에이전트 동작 확인
- **분석**: 학습 곡선으로 수렴 여부 판단
- **설명**: 정책이 어떤 행동을 하는지 직관적으로 이해

#### 2. 학습 곡선 해석
- **Reward 상승**: 정책이 개선됨
- **진동**: 학습률이 높거나, 탐험이 많을 수 있음
- **Plateau**: 수렴했거나, local optimum에 갇혔을 수 있음

---

## Phase 5 — Algorithm Expansion (PPO)

### 구현 내용
- PPO from scratch
- DQN vs PPO 비교

### 핵심 개념

#### 1. Policy Gradient (정책 경사)
- **Value-based (DQN)**: Q를 학습 → 간접적으로 정책 유도
- **Policy-based (PPO)**: 정책 π(a|s)를 직접 학습

```
∇J(θ) = E[ ∇log π(a|s;θ) · A(s,a) ]
```
A(s,a): Advantage. 해당 행동이 평균보다 얼마나 좋은지.

#### 2. PPO (Proximal Policy Optimization)
- **Clipped objective**: 정책 업데이트를 작게 유지해 안정성 확보
- **On-policy**: 현재 정책으로 수집한 데이터만 사용
- **Actor-Critic**: Actor(정책) + Critic(가치) 동시 학습

#### 3. DQN vs PPO 비교
| | DQN | PPO |
|---|-----|-----|
| 행동 공간 | Discrete | Discrete + Continuous |
| 샘플 효율 | Replay로 높음 | On-policy로 상대적으로 낮음 |
| 안정성 | Target net 등 필요 | Clipping으로 비교적 안정 |
| 적용 | Atari, GridWorld | 로봇 제어, 게임 등 |

---

## 전체 학습 순서 요약

1. **Phase 1**: MDP, 보상, 환경 인터페이스
2. **Phase 2**: 가치 함수, Q-learning, DQN (Replay, Target, ε-greedy)
3. **Phase 3**: 에피소드, 학습 루프, 로깅
4. **Phase 4**: 시각화와 분석
5. **Phase 5**: 정책 경사, PPO, 알고리즘 비교
