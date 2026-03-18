# ZeroEnv

Reinforcement learning framework from scratch.

## Phase 1 — Environment (v0.1.0-phase1)

### Quick Start

```bash
pip install gymnasium numpy

# Verify
python test_env.py
```

### GridWorld

- 5×5 그리드, 장애물, 목표/시작 위치 설정
- Gymnasium API: `reset`, `step`, `render`
- 보상: +1(목표), -0.01(스텝), -1(벽/장애물)

### 학습 포인트

→ **[docs/PHASE1_LEARNING.md](docs/PHASE1_LEARNING.md)** — 원리 이해 & 다른 프로젝트 적용법

### Project Structure (Phase 1)

```
zeroenv/
├── envs/gridworld.py   # GridWorld 환경
├── test_env.py         # 검증 스크립트
├── docs/PHASE1_LEARNING.md
└── requirements.txt
```
