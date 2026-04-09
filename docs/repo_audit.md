# Repository Audit

Date: 2026-04-09

Scope: `plan.md`, top-level docs, `configs/`, `dte/`, `scripts/`, `app/`, deployment files, and `tests/`.

This repository already contains two distinct modeling paths:

1. A single-system physics-informed latent neural SDE stack centered on `DigitalTwin`.
2. A newer grouped universal shared-checkpoint baseline centered on `UniversalDigitalTwin`.

The codebase is materially farther along than the early assumptions in `plan.md` in a few areas, especially system abstraction and grouped universal modeling. It is still missing the flowsheet, adapter, law-layer, and customer-onboarding layers described in the roadmap.

## Executive Summary

- The unit-level single-system stack is functional and fairly mature: simulator registry, `SystemSpec`, pluggable physics losses, training, evaluation, API, MPC, online adaptation, and few-shot transfer all exist.
- The repo also has a real universal mixed-system research path: padded mixed batches, `system_id`, typed `state_groups`, numeric system descriptors, grouped encode/decode, and a dedicated universal trainer/evaluator.
- The next roadmap blockers are not basic model plumbing. They are semantic structure gaps:
  - channel-level typed state schema
  - richer process-unit descriptors
  - reusable law and constraint layers
  - adapters/calibration
  - flowsheet graph abstractions
- There are also a few concrete implementation debts that should be addressed before major refactors:
  - hidden training-script config overrides
  - hardcoded dimensions in `TrajectoryDataset`
  - missing param normalization stats in generated HDF5 files
  - an undefined-variable bug in the single-system trainer SDE KL branch
  - a broken PID branch in `scripts/run_mpc.py`

## A. Current Architecture Map

### Core single-system path

| Layer | Current implementation |
| --- | --- |
| System abstraction | `dte/simulators/base.py` defines `DecoderConstraint`, `NormalizationSpec`, `StateGroupSpec`, `SystemSpec`, and `ProcessSimulator` |
| System registry | `dte/simulators/registry.py` resolves `SystemSpec` and simulator instances from YAML |
| Simulators | `dte/simulators/cstr.py`, `dte/simulators/heat_exchanger.py`, `dte/simulators/two_tank.py` |
| Encoder | `dte/models/encoder.py` `Encoder` |
| Decoder | `dte/models/decoder.py` `Decoder` plus generic decoder constraints |
| Latent dynamics | `dte/models/latent_sde.py` `LatentDrift`, `LatentDiffusion`, `LatentSDE` |
| Integrated model | `dte/models/digital_twin.py` `DigitalTwin` |
| Dataset loading | `dte/data/dataset.py` `TrajectoryDataset` |
| Generic data generation | `dte/data/generation_generic.py` `GenericDataGenerator` |
| Real data ingestion | `dte/data/real_data.py` `RealDataIngestion` |
| Physics losses | `dte/physics/base.py`, `dte/physics/registry.py`, and per-system modules under `dte/physics/` |
| Loss assembly | `dte/training/losses.py` `LossComputer` |
| Training loop | `dte/training/trainer.py` `Trainer` |
| Transfer learning | `dte/training/transfer.py` `FewShotAdapter`, `zero_shot_eval` |
| Online adaptation | `dte/training/online.py` `OnlineAdapter` |
| Evaluation | `scripts/evaluate.py` |
| Checkpointing | `DigitalTwin.save/load`, plus script-level checkpoint management in `scripts/train.py` |

### Universal shared-checkpoint path

| Layer | Current implementation |
| --- | --- |
| Mixed-system dataset | `dte/data/multi_system_dataset.py` `MultiSystemTrajectoryDataset` |
| System metadata tables | `UniversalSystemMetadata` in `dte/data/multi_system_dataset.py` |
| Universal model | `dte/models/universal_digital_twin.py` `UniversalDigitalTwin` |
| Universal trainer | `dte/training/universal_trainer.py` `UniversalTrainer` |
| Universal training entrypoint | `scripts/train_universal.py` |
| Universal evaluation entrypoint | `scripts/evaluate_universal.py` |

### Model behavior and rollout logic

| Concern | Current implementation |
| --- | --- |
| Initial encoding | `DigitalTwin.encode` and `UniversalDigitalTwin.encode` |
| Deterministic rollout | `DigitalTwin.rollout_latent(..., stochastic=False)` and `UniversalDigitalTwin.rollout_latent` |
| Stochastic rollout | `DigitalTwin.predict`, `DigitalTwin.predict_ensemble`, `LatentSDE.__call__` |
| Decoder-time conditioning | Both single-system and universal decoders condition on latent state, params, and controls |
| Simulator-informed prior | `DigitalTwin.latent_drift` can blend learned drift with a simulator prior |
| Learned solver | Implemented in `LatentSDE` and used through `DigitalTwin.rollout_latent` |
| Neural CDE path | Implemented in both `LatentSDE` and `UniversalDigitalTwin` |
| Self-correction | Implemented in `LatentSDE` and used in the single-system rollout |

### Uncertainty path

| Concern | Current implementation |
| --- | --- |
| Variational encoder | `Encoder.encode` returns `z_mean`, `z_logvar`; `UniversalDigitalTwin` optionally does the same |
| Stochastic latent diffusion | `LatentDiffusion` in `dte/models/latent_sde.py` |
| SDE solver path | `LatentSDE.__call__` with `diffrax` Brownian control term |
| Ensemble inference | `DigitalTwin.predict_ensemble` and `/ensemble` API endpoint |
| Training-time stochasticity gate | `sde_training.enabled` and `sde_training.warmup_steps` in trainer/config |

### Physics path

| Concern | Current implementation |
| --- | --- |
| Physics interface | `dte/physics/base.py` `PhysicsLoss` |
| Registry resolution | `dte/physics/registry.py` |
| CSTR residuals | `dte/physics/cstr.py` |
| Heat exchanger residuals | `dte/physics/heat_exchanger.py` |
| Two-tank residuals | `dte/physics/two_tank.py` |
| Evaluation diagnostics | Registry-provided diagnostic functions consumed by `scripts/evaluate.py` |

### Production and operator-facing surfaces

| Surface | Current implementation |
| --- | --- |
| REST API | `dte/api/models.py`, `dte/api/service.py` |
| Interactive dashboard | `app/dashboard.py` |
| Autoresearch dashboard | `app/agent_dashboard.py` |
| MPC | `dte/control/mpc.py`, `scripts/run_mpc.py` |
| PID baseline | `dte/control/pid.py` |
| Docker/deployment | `Dockerfile`, `docker-compose.yml` |

### Autoresearch and campaign tooling

| Surface | Current implementation |
| --- | --- |
| Bounded harness | `scripts/autoresearch.py` |
| Baseline/result helpers | `dte/autoresearch/workflow.py` |
| Autonomous agent loop | `scripts/agent.py` |
| Multi-campaign runner | `scripts/run_campaigns.py` |
| Backlog comparison runner | `scripts/compare_idea_backlogs.py` |
| LLM patch scrubber | `scripts/scrubber.py` |

## B. Data Model Map

### HDF5 training-data schema

The repo standardizes on this on-disk schema:

| Key | Shape |
| --- | --- |
| `states` | `(N, T, state_dim)` |
| `controls` | `(N, T, control_dim)` |
| `disturbances` | `(N, T, disturbance_dim)` |
| `params` | `(N, param_dim)` |
| `time` | `(N, T)` |
| `normalization/state_mean` | `(state_dim,)` |
| `normalization/state_std` | `(state_dim,)` |
| `normalization/control_mean` | `(control_dim,)` |
| `normalization/control_std` | `(control_dim,)` |
| `normalization/disturbance_mean` | `(disturbance_dim,)` |
| `normalization/disturbance_std` | `(disturbance_dim,)` |

Observed caveat: generated and ingested datasets do not currently write `param_mean` or `param_std`, even though `TrajectoryDataset.normalize_params` and `denormalize_params` expect them.

### Single-system in-memory batch contract

`TrajectoryDataset`:

- Loads HDF5 into physical units.
- Extracts overlapping subsequences of length `seq_len` with stride `stride`.
- Returns per-sample tensors:
  - `states`: `(seq_len, state_dim)`
  - `controls`: `(seq_len, control_dim)`
  - `disturbances`: `(seq_len, disturbance_dim)`
  - `params`: `(param_dim,)`
  - `t`: `(seq_len,)`
- Returns batched tensors from `sample_batch`:
  - `states`: `(batch, seq_len, state_dim)`
  - `controls`: `(batch, seq_len, control_dim)`
  - `disturbances`: `(batch, seq_len, disturbance_dim)`
  - `params`: `(batch, param_dim)`
  - `t`: `(batch, seq_len)`

Important detail: the on-disk key is `time`, but the in-memory subsequence key is `t`.

### Universal mixed-system batch contract

`MultiSystemTrajectoryDataset.sample_batch` pads to max dimensions across systems and returns:

- `states`: `(batch, seq_len, max_state_dim)`
- `controls`: `(batch, seq_len, max_control_dim)`
- `disturbances`: `(batch, seq_len, max_disturbance_dim)`
- `params`: `(batch, max_param_dim)`
- `t`: `(batch, seq_len)`
- `state_mask`: `(batch, max_state_dim)`
- `control_mask`: `(batch, max_control_dim)`
- `disturbance_mask`: `(batch, max_disturbance_dim)`
- `param_mask`: `(batch, max_param_dim)`
- `time_mask`: `(batch, seq_len)`
- `system_id`: `(batch,)`

This is the only path in the repo that currently supports mixed dimensions, padding, and masking.

### How systems are identified today

Single-system path:

- One training run uses one `system_config`.
- The system is identified indirectly by the YAML and the resolved `SystemSpec`.
- No explicit `system_id` appears in single-system batches or checkpoints.

Universal path:

- Every batch sample carries an integer `system_id`.
- The model also consumes numeric `system_descriptor` tables and state-group metadata derived from `SystemSpec`.

### Controls, disturbances, and parameters

- Controls and disturbances are always explicit trajectory channels.
- Parameters are per-trajectory vectors, not per-timestep channels.
- The single-system model treats params as flat vectors.
- The universal model treats params as padded flat vectors with masks.
- No typed parameter descriptors exist yet.

### Normalization flow

There are two normalization mechanisms in play:

1. Dataset normalization stats from HDF5:
   - used by `LossComputer` and by the trainer/evaluator for normalized loss metrics.
2. `SystemSpec.normalization`:
   - used to initialize encoder/decoder/SDE input scaling.
   - used to build universal metadata tables.

This split is workable, but it means the training loss scale and the model-input scale are sourced from different places.

## C. System Abstraction Map

### `SystemSpec`

`SystemSpec` already covers more than `plan.md` assumes. It includes:

- dimensions: `state_dim`, `control_dim`, `disturbance_dim`, `param_dim`
- semantic names: `state_names`, `control_names`, `disturbance_names`
- output constraints: `decoder_constraints`
- normalization: `NormalizationSpec`
- defaults: `default_initial_state`, `default_nominal_disturbance`
- operating envelopes: `control_ranges`, `disturbance_ranges`
- coarse typed grouping: `state_groups`

### `StateGroupSpec`

This is the current typed-state abstraction:

- `name`
- `kind`
- `indices`

It is already used in the universal model and system configs, but it is still coarser than the channel-level schema requested in `plan.md`.

### `ProcessSimulator`

The simulator base class provides:

- `dynamics`
- `simulate`
- `steady_state`
- data-generation hooks for fast rollout, batch rollout, batch steady state
- parameter sampling hooks
- measurement-noise hooks
- trajectory-validity hooks

This is a strong abstraction boundary and is already reusable for new unit systems.

### Registry-based resolution

Current architecture is explicitly registry-driven:

- `dte/simulators/registry.py` resolves `SystemSpec` and simulators.
- `dte/physics/registry.py` resolves physics losses and diagnostics.

That means new unit systems already enter the platform at the registry boundary instead of requiring model-code branches.

### Universal metadata layer

The universal path builds a second abstraction level on top of `SystemSpec`:

- padded normalization tables
- mask tables
- dimension tables
- numeric system descriptors
- state-group masks and kind ids

This is the repo’s current bridge from per-system YAML specs to shared-checkpoint learning.

## D. Technical Debt And Roadmap Blockers

### High-priority implementation debt

1. `dte/data/dataset.py` still hardcodes the old CSTR-like dimensions in `state_dim`, `control_dim`, `disturbance_dim`, and `param_dim`.
   This directly conflicts with the repo’s claimed generality.

2. `TrajectoryDataset.normalize_params` and `denormalize_params` require `param_mean` and `param_std`, but generated/ingested HDF5 files do not write those stats.
   This is a broken normalization contract, even if those helpers are not on the hot path today.

3. `dte/training/trainer.py` uses `dt` before it is defined in the SDE KL branch.
   The affected line is the call to `self.loss_computer.sde_kl_loss(diff_values, dt)` before `dt = ts[0, 1] - ts[0, 0]` is assigned later in the function.

4. `scripts/run_mpc.py` references `pid_controller` without creating it.
   The PID comparison path will fail when `--compare_pid` is used on CSTR.

5. `scripts/train.py` silently overrides loaded YAML values.
   It forces:
   - `model.initial_diffusion_scale = 0.0001`
   - `optimizer.peak_lr = 5.0e-4`
   - `optimizer.gradient_clip = 0.5`
   - `loss_weights.kl = 0.0`
   This means the YAML alone is not the real source of truth for the training baseline.

### Architectural blockers for the roadmap

6. Typed state semantics stop at `StateGroupSpec`.
   There is no channel-level schema with units, bounds, conserved groups, or role metadata as proposed in `plan.md`.

7. The single-system path is still flat-vector only.
   `Encoder`, `Decoder`, and `LatentSDE` do not consume typed state groups; grouped semantics only exist in the universal path.

8. There is no `ProcessUnitSpec` successor yet.
   `SystemSpec` is richer than the plan assumed, but it still lacks:
   - family/subtype taxonomy
   - law tags
   - parameter descriptors
   - topology ports
   - customer-facing metadata

9. There is no reusable `physics/constraints.py`.
   Generic positivity and bounded decoder constraints exist, but reusable physics constraint utilities do not.

10. There is no flowsheet or stream abstraction.
    The repo has no `flowsheet/`, graph dataset, stream schema, plant topology model, or recycle-loop rollout code.

11. There are no adapter layers or customer calibration modules.
    The nearest existing feature is `FewShotAdapter`, which is a selective fine-tuning utility, not a proper adapter/calibration stack.

12. Law layers are still per-system handwritten physics residuals.
    There is no modular chemistry, thermo, or biology API.

### Control and evaluation gaps

13. The control layer is still effectively CSTR-oriented.
   `dte/control/mpc.py` uses CSTR-shaped assumptions and fixed `F_in`/`Tc_in` bounds from `configs/mpc_default.yaml`.

14. Action sensitivity evaluation is missing.
   The current evaluation stack computes trajectory error and physics diagnostics, but not local control-gain or perturbation-response metrics.

15. Uncertainty calibration is only lightly surfaced.
   The repo has stochastic rollouts and ensemble plots, but only one ad hoc calibration number in `scripts/evaluate.py`.

16. The universal path currently has no physics-loss integration.
   It is a shared-checkpoint baseline, not yet a shared physics-informed foundation stack.

### Documentation and test coverage drift

17. Some docs are stale relative to current code.
   Example: `auto_research.md` still states deterministic training uses `latent_sde.mean_trajectory`, but `Trainer` now uses `model.rollout_latent(..., stochastic=False)`.

18. Test coverage is concentrated in models, datasets, simulators, physics, and autoresearch helpers.
   There is little or no direct test coverage for:
   - FastAPI service behavior
   - Streamlit dashboards
   - Docker paths
   - `scripts/run_mpc.py`
   - universal train/eval scripts as CLIs
   - real-data ingestion end-to-end

## What The Repo Already Has That The Plan Underestimates

- `SystemSpec` is already the central system-agnostic contract.
- `state_groups` already exist in YAML and in `SystemSpec`.
- A grouped universal model already exists.
- The universal model already uses:
  - state-group masks
  - group-kind embeddings
  - descriptor-conditioned FiLM-style modulation
  - grouped masked pretraining in `UniversalTrainer`
- The single-system path already includes:
  - simulator prior blending
  - learned solver gating
  - self-correcting updates
  - optional Neural CDE terms

## Bottom Line

The repo does not need a first-principles rewrite to start the roadmap. The right next move is to treat the current codebase as:

- a stable single-unit physics-informed stack
- a promising but separate universal grouped baseline

and then close the semantic gaps between them:

- richer typed channel metadata
- shared reusable constraint utilities
- adapters/calibration
- graph/flowsheet abstractions

without regressing the existing simulator-registry and physics-registry architecture.
