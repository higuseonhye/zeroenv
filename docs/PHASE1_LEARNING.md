# Phase 1 학습 포인트 — Environment

RL을 공부하는 입장에서 **이해해야 할 원리**와 **다른 프로젝트에 적용하는 방법** 정리.

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
| **State S** | (row, col) | 게임: 화면 픽셀, 로봇: 관절 각도, 추천: 사용자 벡터 |
| **Action A** | {up, right, down, left} | Discrete 또는 Continuous |
| **Transition P** | 확정적: s' = s + Δ(s,a) | 확률적일 수도 있음 |
| **Reward R** | +1, -0.01, -1 | 목표에 맞게 설계 |
| **γ (gamma)** | 0.99 | 미래를 얼마나 중요하게 볼지 |

### 2.2 Markov 성질의 의미

**다음 상태는 현재 상태와 행동에만 의존한다.**

- 과거 경로를 저장할 필요가 없다.
- 현재 상태만 알면 최적 결정을 내릴 수 있다.
- 이 성질이 깨지면 (부분 관측, 히스토리 의존) POMDP 등으로 확장해야 한다.

→ **새 환경을 설계할 때:** "이 문제가 Markov한가?"를 먼저 확인한다.

---

## 3. 보상 설계 — Reward Shaping

### 3.1 보상이 학습을 결정한다

Agent는 **누적 보상**을 최대화하려 한다.  
보상 구조가 잘못되면, 의도와 다른 행동을 학습한다.

### 3.2 희소 vs 밀집

| 유형 | 장점 | 단점 |
|-----|------|------|
| **희소** (목표만 +1) | 의도가 명확 | 학습 신호가 드물어 학습이 어려움 |
| **밀집** (매 스텝 보상) | 학습 신호가 많음 | 잘못된 보상이 잘못된 행동을 유도 |

### 3.3 GridWorld 설계 원리

```
+1  : 목표 도달     → "여기로 가라"
-0.01: 매 스텝     → "빨리 끝내라"
-1  : 벽/장애물    → "이건 하지 마라"
```

→ **적용 시:** 목표 행동에는 +, 위험/비효율에는 -, 그리고 스텝 비용을 고려한다.

---

## 4. Gymnasium API — 왜 표준 인터페이스인가?

### 4.1 reset / step / render

```python
obs, info = env.reset(seed=42)           # 에피소드 시작
obs, reward, term, trunc, info = env.step(action)  # 한 스텝
frame = env.render()                     # 시각화 (선택)
```

- **reset**: 에피소드마다 동일한 초기화, `seed`로 재현성 확보
- **step**: (obs, reward, terminated, truncated, info) 5-tuple
- **render**: human(창) 또는 rgb_array(이미지)

### 4.2 terminated vs truncated

| | terminated | truncated |
|---|------------|-----------|
| **의미** | MDP 내 자연 종료 (목표 도달 등) | MDP 외부 제한 (시간 초과 등) |
| **Bootstrapping** | 다음 상태 가치 = 0 (끝났으므로) | 다음 상태 가치 사용 가능 |
| **예시** | 목표 도달, 사망 | max_steps 초과 |

→ **적용 시:** TD 학습에서 `done` 처리 시 terminated/truncated를 구분하면 더 정확하다.

### 4.3 표준을 쓰는 이유

- 알고리즘 코드를 환경에 맞게 바꿀 필요가 없다.
- 다른 사람이 만든 알고리즘을 그대로 쓸 수 있다.
- CartPole, Atari 등 수많은 환경과 호환된다.

→ **새 환경을 만들 때:** Gymnasium API를 따르면 재사용성이 크게 올라간다.

---

## 5. 구현 시 체크리스트 (다른 프로젝트용)

새 RL 환경을 만들 때 확인할 것:

- [ ] **State space**: 무엇을 관측하는가? shape, dtype, 범위는?
- [ ] **Action space**: Discrete / Box / MultiDiscrete?
- [ ] **보상**: 목표에 맞는가? 희소/밀집 균형은?
- [ ] **Markov**: 현재 상태만으로 충분한가?
- [ ] **terminated vs truncated**: 둘 다 필요한가?
- [ ] **seed**: `reset(seed=)`로 재현 가능한가?
- [ ] **render**: 디버깅·논문용 시각화가 필요한가?

---

## 6. Phase 1에서 배운 것 요약

1. **Environment-first**: 환경 정의가 RL 프로젝트의 출발점이다.
2. **MDP**: 상태, 행동, 전이, 보상, γ로 문제를 수식화한다.
3. **보상 설계**: 목표와 위험을 보상으로 명확히 표현한다.
4. **표준 API**: Gymnasium을 따르면 알고리즘 재사용이 쉬워진다.
5. **terminated/truncated**: Bootstrapping과 학습 안정성을 위해 구분한다.
