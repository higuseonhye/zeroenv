# Changelog

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
