# Phase 7 MPC And DRL Readiness

Date: 2026-04-10

Phase 7 adds the missing control-facing abstraction layer on top of the simulator and model stack: a generic MPC runtime interface, a Gymnasium-style environment wrapper, state-correction hooks, and reusable closed-loop metrics.

## What Landed

Control runtime:

- `dte/control/mpc_interface.py`
  - `ProcessMPCInterface`
  - deterministic rollout through the simulator or a loaded `DigitalTwin`
  - current state estimate management
  - measurement assimilation hook integration
  - generic objective and constraint hooks
  - random-shooting optimisation for controller prototypes

RL environment:

- `dte/control/rl_env.py`
  - `ProcessControlEnv`
  - Gymnasium-style `reset()` and `step()` signatures
  - minimal `BoxSpace` objects for action and observation spaces
  - configurable disturbance schedules
  - reward computed from control-oriented metrics

State correction:

- `dte/control/state_correction.py`
  - exponential measurement filtering
  - state assimilation against a prior estimate
  - optional clipping to typed state bounds
  - latent refresh through the `DigitalTwin` encoder when a model is attached
  - one-step latent prediction helper

Evaluation metrics:

- `dte/evaluation/control_metrics.py`
  - tracking cost
  - control effort cost
  - constraint violation summaries
  - disturbance sensitivity
  - mismatch robustness
  - aggregate closed-loop metrics

Exports:

- `dte/control/__init__.py` now exposes the new control interfaces alongside the existing MPC and PID classes.

## Backward Compatibility

The existing control path was left intact:

- `dte/control/mpc.py`
- `scripts/run_mpc.py`

No existing command-line interface was removed or renamed. The new Phase 7 layer is additive.

## Example Usage

### Generic MPC-facing rollout

```python
runtime = ProcessMPCInterface(spec, simulator, model=None)
runtime.reset(initial_state=spec.default_initial_state_array())

best = runtime.optimize_random_shooting(
    target_state=spec.default_initial_state_array(),
    horizon=12,
    n_candidates=16,
)
```

### RL-style environment loop

```python
env = ProcessControlEnv(spec, simulator)
obs, info = env.reset(seed=0)

action = env.action_space.sample(seed=1)
obs, reward, terminated, truncated, info = env.step(action)
```

### Measurement correction

```python
hook = StateCorrectionHook(spec, model=model)
update = hook.correct(
    prior_state=prior_state,
    measurement=measurement,
    control=control,
)
```

## Verification

- `JAX_PLATFORMS=cpu pytest tests/test_phase7_control.py -q`
- `python -m py_compile dte/control/__init__.py dte/control/mpc_interface.py dte/control/rl_env.py dte/control/state_correction.py dte/evaluation/control_metrics.py tests/test_phase7_control.py`

The test slice covers:

- one state-correction path with latent refresh
- one generic MPC optimisation loop
- one Gymnasium-style environment rollout
- disturbance sensitivity and mismatch robustness metrics

## Current Scope

This is intentionally a lightweight readiness layer, not a full control framework.

What is covered now:

- generic rollout/evaluate hooks for control algorithms
- a process environment wrapper suitable for RL experiments
- state correction hooks for online use
- reusable closed-loop metrics

What is still intentionally thin:

- no bundled RL training algorithm or replay buffer stack
- optimisation in `ProcessMPCInterface` is random-shooting, not a full advanced solver family
- the environment wrapper is Gymnasium-style without taking a hard dependency on `gymnasium`
