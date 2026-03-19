# Changelog

## [0.3.0-phase2-5] - Phase 2–5 Complete

### Added
- **Phase 2**: `agents/base.py`, `algorithms/dqn.py` (Replay, Target, ε-greedy)
- **Phase 3**: `training/runner.py` (run_training, run_training_ppo), curriculum_factory
- **Phase 4**: `visualization/renderer.py` (plot_training_curves, replay_agent), Streamlit app (GridWorld, trained agent replay)
- **Phase 5**: `algorithms/ppo.py` (Actor-Critic, GAE, clipped objective)
- **main.py**: `train`, `eval`, `compare` commands (DQN vs PPO)
- **scripts/**: validate_cartpole.py, compare_dqn_ppo.py
- **LEARNING.md**: Phase 2–5 learning guide

### Changed
- app.py: DQN/PPO algorithm selection, checkpoint load
- compare: history.npy save, compare_curves.png (DQN vs PPO)

---

## [0.2.0-phase1-update] - Phase 1 Extension (FAQ perspectives reflected)

### Added
- **Configurable rewards**: `reward_goal`, `reward_step`, `reward_obstacle` params
- **envs/factories.py**: `make_gridworld_curriculum`, `make_gridworld_procedural`
- **Curriculum learning**: `--curriculum` flag, `curriculum_factory` in run_training
- **docs/PHASE1_LEARNING.md**: FAQ perspectives reflected, reward·environment design·swap summary

### Changed
- GridWorld: reward values parameterized (defaults preserved, backward compatible)
- training/runner: optional `curriculum_factory` support

### Phase 2-5 Impact
- None. observation_space, action_space, reset, step interfaces unchanged.

---

## [0.1.0-phase1] - Phase 1: Environment

### Added
- **GridWorld** (`envs/gridworld.py`)
  - 5×5 default grid, obstacles, goal/start position config
  - Gymnasium API: reset, step, render (human / rgb_array)
  - Rewards: +1(goal), -0.01(step), -1(wall/obstacle)
- **test_env.py**: Environment behavior verification
- **docs/PHASE1_LEARNING.md**: Phase 1 learning points (principles·application)

### Dependencies
- gymnasium, numpy
