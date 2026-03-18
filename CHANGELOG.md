# Changelog

## [0.3.0-phase2-5] - Phase 2–5 완성

### Added
- **Phase 2**: `agents/base.py`, `algorithms/dqn.py` (Replay, Target, ε-greedy)
- **Phase 3**: `training/runner.py` (run_training, run_training_ppo), curriculum_factory
- **Phase 4**: `visualization/renderer.py` (plot_training_curves, replay_agent), Streamlit app (GridWorld, trained agent replay)
- **Phase 5**: `algorithms/ppo.py` (Actor-Critic, GAE, clipped objective)
- **main.py**: `train`, `eval`, `compare` commands (DQN vs PPO)
- **scripts/**: validate_cartpole.py, compare_dqn_ppo.py
- **LEARNING.md**: Phase 2–5 학습 가이드

### Changed
- app.py: DQN/PPO 알고리즘 선택, 체크포인트 로드
- compare: history.npy 저장, compare_curves.png (DQN vs PPO)

---

## [0.2.0-phase1-update] - Phase 1 확장 (FAQ 관점 반영)

### Added
- **Configurable rewards**: `reward_goal`, `reward_step`, `reward_obstacle` params
- **envs/factories.py**: `make_gridworld_curriculum`, `make_gridworld_procedural`
- **Curriculum learning**: `--curriculum` flag, `curriculum_factory` in run_training
- **docs/PHASE1_LEARNING.md**: FAQ 관점 반영, 보상·환경 설계·교체 정리

### Changed
- GridWorld: 보상 값 파라미터화 (기본값 유지, backward compatible)
- training/runner: optional `curriculum_factory` 지원

### Phase 2-5 영향
- 없음. observation_space, action_space, reset, step 인터페이스 동일.

---

## [0.1.0-phase1] - Phase 1: Environment

### Added
- **GridWorld** (`envs/gridworld.py`)
  - 5×5 기본 그리드, 장애물, 목표/시작 위치 설정
  - Gymnasium API: reset, step, render (human / rgb_array)
  - 보상: +1(목표), -0.01(스텝), -1(벽/장애물)
- **test_env.py**: 환경 동작 검증
- **docs/PHASE1_LEARNING.md**: Phase 1 학습 포인트 (원리·적용법)

### Dependencies
- gymnasium, numpy
