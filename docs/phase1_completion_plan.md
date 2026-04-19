# Phase 1 Completion Plan

This document defines the concrete plan to close `Phase 1: Unit Foundation V1`
from [docs/full_stack_convergence_program.md](/home/ismayil/digital-twin-engine/docs/full_stack_convergence_program.md:1).

It is based on the current canonical run, not on hypothetical failure modes.

## Current Status

Canonical workspace:

- [outputs/unit_foundation_phase1_run_v11/summary.json](/home/ismayil/digital-twin-engine/outputs/unit_foundation_phase1_run_v11/summary.json:1)

Current gate state:

- `shared_checkpoint_trains_reproducibly`: pass
- `control_response_fidelity_is_measured`: pass
- `control_gate_completed`: pass
- `rollout_stability_on_held_out_variants`: fail
- `transfer_beats_scratch_on_targets`: fail

Current red metrics:

- rollout outlier: `cstr_fast_kinetics` rollout RMSE `20.4004`
- rollout outlier: `heat_exchanger` rollout RMSE `10.1438`
- transfer fail: `two_tank_high_throughput` warm-start rollout RMSE `0.8689` vs scratch `0.1274`

Primary evidence:

- [eval/summary.json](/home/ismayil/digital-twin-engine/outputs/unit_foundation_phase1_run_v11/outputs/unit_foundation/eval/summary.json:1)
- [transfer_benchmark/summary.json](/home/ismayil/digital-twin-engine/outputs/unit_foundation_phase1_run_v11/outputs/unit_foundation/transfer_benchmark/summary.json:1)
- [control_gate/summary.json](/home/ismayil/digital-twin-engine/outputs/unit_foundation_phase1_run_v11/outputs/unit_foundation/control_gate/summary.json:1)

## Definition Of Done

Phase 1 is complete only when the canonical baseline run finishes with:

- `rollout_stability_on_held_out_variants: true`
- `transfer_beats_scratch_on_targets: true`
- no changes to the primary workflow beyond the canonical path:
  - `scripts/run_unit_foundation_baseline.py`
  - `scripts/train_universal.py`
  - `scripts/evaluate_universal.py`
  - `configs/training_universal_phase1_regime.yaml`
  - `configs/generation_phase1_regime.yaml`

No new permanent debug script, alternate benchmark script, or parallel config tree
should be introduced to close this phase.

## Execution Order

### 1. Fix Shared Checkpoint Rollout Outliers First

The foundation checkpoint is still weak on:

- `cstr_fast_kinetics`
- `heat_exchanger`

That means the repo is not only failing transfer. It is also failing the
foundation-quality gate on the main checkpoint itself.

Required work:

- tune the canonical regime training config, not a separate permanent config
- improve long-horizon behavior for the two failing families before touching
  customer-facing work
- promote only the winning settings back into the canonical config surface

Allowed levers:

- regime sampling weights
- multi-step loss emphasis
- family- or subtype-targeted loss terms
- normalization / regularization settings in the universal trainer
- shared model capacity only if the current width is the real limiter

Immediate target:

- reduce `cstr_fast_kinetics` rollout RMSE below `10`
- reduce `heat_exchanger` rollout RMSE below `10`

Implementation rule:

- iterate using temporary workspace-local config copies under `outputs/`
- once the winning change is clear, land it in `configs/training_universal_phase1_regime.yaml`
- do not keep multiple active Phase 1 configs in the repo

### 2. Make Transfer Policy Explicit Instead Of One-Size-Fits-All

Current transfer failures show that one warm-start policy is not good enough for
every family.

The canonical transfer benchmark should support a small explicit transfer policy
surface per target family or subtype, inside the same baseline runner.

Required policy dimensions:

- trainable surface:
  - `full`
  - `adapters`
- normalization tuning:
  - on
  - off
- warm-start optimizer schedule:
  - few-shot schedule derived from target batch count
  - optional family-specific override when clearly justified
- source-row initialization policy:
  - exact-name row reuse
  - nearest-family fallback reuse

Targets that must be solved:

- `two_tank_high_throughput`

Success rule per target:

- warm-start total loss must be less than or equal to scratch
- warm-start rollout RMSE must be less than or equal to scratch

Implementation rule:

- add the policy surface inside `scripts/run_unit_foundation_baseline.py`
- do not create a second transfer benchmark script
- if a policy works only for one family, encode it as a canonical target-policy rule,
  not as a one-off manual command

### 3. Use Targeted Canonical Reruns, Not Full Blind Reruns

Iteration speed matters, but the path must stay canonical.

For rollout closure:

- run training + evaluation through the canonical stack
- inspect [eval/summary.json](/home/ismayil/digital-twin-engine/outputs/unit_foundation_phase1_run/outputs/unit_foundation/eval/summary.json:1)

For transfer closure:

- rerun only the failing targets through the canonical baseline runner using
  `--transfer_targets`
- skip already-solved phases when appropriate:
  - `--skip_generation`
  - `--skip_training`
  - `--skip_evaluation`
  - `--skip_control`

This keeps the iteration loop fast without introducing a second active workflow.

Current canonical closure rule:

- rollout-targeted attempts should use:
  - `--skip_generation`
  - `--skip_transfer`
  - `--skip_control`
- transfer-targeted attempts should use:
  - `--skip_generation`
  - `--skip_training`
  - `--skip_evaluation`
  - `--skip_control`
  - `--transfer_targets two_tank_high_throughput`

### 4. Re-run The Full Canonical Baseline Once The Outliers Are Closed

After the rollout gate and transfer gate pass in targeted reruns:

- run the full canonical baseline end to end again
- do not declare Phase 1 complete from partial reruns alone

Required final command:

```bash
source .venv/bin/activate
python scripts/run_unit_foundation_baseline.py \
  --workspace_dir outputs/unit_foundation_phase1_final \
  --jax_platform cpu
```

Phase 1 closes only when the top-level summary flips to:

- `status: ok`
- `acceptance.accepted: true`

## Concrete Work Plan

### Work Item A: Close Rollout Stability

Files:

- `configs/training_universal_phase1_regime.yaml`
- `dte/training/universal/trainer.py`
- `scripts/evaluate_universal.py`

Steps:

1. identify the minimal training changes that improve `cstr_fast_kinetics` and `heat_exchanger`
2. retrain the shared checkpoint on the canonical regime corpus
3. re-evaluate rollout metrics on the canonical evaluation path
4. stop only when both outliers clear the current gate

### Work Item B: Close Transfer On Reactor And Hydraulic Variants

Files:

- `scripts/run_unit_foundation_baseline.py`
- `dte/calibration/unit_calibration.py`
- `tests/test_unit_foundation_baseline.py`
- `tests/test_unit_calibration.py`

Steps:

1. add explicit target-family transfer policy selection
2. keep `cstr_fast_kinetics` and `heat_exchanger_high_ua` green
3. solve `two_tank_high_throughput`
4. codify the winning policy in the canonical benchmark
5. add regressions so those targets cannot silently fall back again

### Work Item C: Reconfirm Control Gate On The Winning Checkpoint

Files:

- `scripts/run_unit_foundation_baseline.py`
- `dte/control/`
- `tests/test_phase7_control.py`

Steps:

1. rerun control gate on the new accepted foundation checkpoint
2. confirm control metrics remain green after the training and transfer changes

## Exit Checklist

Before marking Phase 1 complete, all of the following must be true:

- canonical eval rollout outliers are gone
- canonical transfer benchmark beats scratch on all held-out targets
- canonical control gate still passes
- no temporary debug configs or scripts remain in the active repo surface
- the final accepted behavior is documented in the canonical docs, not in chat logs
