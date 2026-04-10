# Implementation Mapping Against `plan.md`

Date: 2026-04-10

This document maps the roadmap in `plan.md` to the code that already exists in the repository.

Status legend:

- `Implemented`: the repo already has a working version of the planned capability.
- `Partial`: some of the intent exists, but the exact abstraction or acceptance criteria are not met.
- `Missing`: no meaningful implementation exists yet.

## High-Level Phase Summary

| Phase | Status | Notes |
| --- | --- | --- |
| Phase 0. Repository audit | Implemented | Covered by `docs/repo_audit.md` and this file |
| Phase 1. Stabilize/refactor universal unit model | Implemented | Thin-slice implementation landed: typed channel schema, richer unit spec, optional grouped encoder, reusable constraints, multi-horizon universal loss, and new evaluation diagnostics |
| Phase 2. Adapters and family conditioning | Implemented | Family/subtype/law-tag conditioning, residual adapters, and target-unit calibration now exist in the universal path |
| Phase 3. Flowsheet graph modeling | Implemented | Thin-slice graph layer landed: flowsheet schema, graph dataset, synthetic demo data, shared flowsheet model, recycle-aware rollout, and plant-level proxy losses |
| Phase 4. Modular law layers | Implemented | Reusable chemistry, thermo, and biology law modules now exist with config-driven bundle integration and physics-registry hooks |
| Phase 5. Customer adaptation workflow | Implemented | Customer onboarding schema, template matching, an adaptation CLI, and automatic validation report generation now exist for unit workflows |
| Phase 6. Website/demo app | Implemented | Dedicated demo API routes, a separate interactive Streamlit demo app, and config-driven example demos now exist locally |
| Phase 7. MPC and DRL readiness | Implemented | A generic MPC runtime, Gymnasium-style env wrapper, state-correction hooks, and control-oriented metrics now exist alongside the legacy MPC/PID path |
| Phase 8. Distributed/transport-aware units | Missing | No distributed-unit modeling layer yet |

## Plan Assumptions Already Outdated

Before the phase-by-phase mapping, three plan assumptions are already stale:

1. `SystemSpec` is not “too minimal” in the current repo.
   It already includes dimensions, names, normalization, defaults, ranges, decoder constraints, and `state_groups`.

2. The repo already has typed grouped universal modeling.
   `state_groups` are declared in system YAML files and consumed by `UniversalDigitalTwin`.

3. The repo already has a universal shared-checkpoint path.
   `MultiSystemTrajectoryDataset`, `UniversalDigitalTwin`, `UniversalTrainer`, `scripts/train_universal.py`, and `scripts/evaluate_universal.py` are present and wired together.

## Phase 0. Repository Audit

### Planned outputs

- `docs/repo_audit.md`

### Current repo state

- The repository structure, training paths, data model, loss system, universal path, evaluation surface, API/UI surface, and autoresearch loop have now been documented.

### Status

- `Implemented`

## Phase 1. Stabilize And Refactor The Universal Unit Model

### 1. Introduce typed state groups

Plan intent:

- add state-channel roles like inventory, temperature, concentration, pressure, flow, biology

Current state:

- `dte/core/state_schema.py` now defines `StateChannel`
- `ProcessUnitSpec` exposes ordered `state_channels`
- default system YAMLs now carry explicit typed channel metadata
- `SystemSpec.state_groups` still exist and remain the grouping backbone
- `UniversalDigitalTwin` continues to consume group masks and group-kind ids
- the single-system stack can now opt into a grouped encoder path driven by the same state-group semantics

Status:

- `Implemented`

### 2. Extend `SystemSpec` into `ProcessUnitSpec`

Plan intent:

- richer abstraction with unit type, law tags, parameter descriptors, ports, constraints metadata

Current state:

- `ProcessUnitSpec` now subclasses `SystemSpec`
- it adds state/control/disturbance channel metadata, parameter descriptors, unit family/type metadata, law tags, topology ports, and constraints metadata
- the registry returns `ProcessUnitSpec` while remaining backward-compatible with `SystemSpec` callers

Status:

- `Implemented`

### 3. Add typed encoders instead of one flat encoder only

Current state:

- `UniversalDigitalTwin` already has a grouped encoder path:
  - state-group encoder
  - state-group mixer
  - group-kind embeddings
  - descriptor-conditioned FiLM modulation
- `dte/models/grouped_encoder.py` now adds a grouped typed encoder to the single-system path
- `DigitalTwin.from_config` preserves the flat encoder by default and enables the grouped encoder only when configured

Status:

- `Implemented`

### 4. Add explicit physics constraint utilities

Current state:

- generic decoder constraints exist in `dte/models/decoder.py`
- per-system physics residuals exist in `dte/physics/*.py`
- `dte/physics/constraints.py` now provides reusable positivity, bound, mass-balance, energy-balance, and monotonicity helpers
- the universal trainer now uses the bound and positivity utilities as optional rollout regularizers

Status:

- `Implemented`

### 5. Add multi-horizon training support

Current state:

- the single-system trainer already combines:
  - reconstruction loss
  - full trajectory loss
  - one-step teacher-forced loss
  - curriculum over `seq_len`
  - teacher-forcing annealing
- the universal trainer uses:
  - initial reconstruction
  - full trajectory loss
  - one-step loss
- `UniversalTrainer` now supports configurable K-step teacher-forced losses through `multi_horizon.k_steps` and `loss_weights.k_step`
- full-trajectory rollout loss remains the long-window component

Status:

- `Implemented`

### 6. Add action sensitivity evaluation

Current state:

- `dte/evaluation/control_sensitivity.py` adds finite-difference Jacobian and mismatch metrics
- `scripts/evaluate_universal.py` now compares local control gains from the learned model against the registered simulator

Status:

- `Implemented`

### 7. Improve uncertainty calibration

Current state:

- stochastic latent diffusion exists
- ensemble rollout exists
- `scripts/evaluate.py` computes one calibration number: percent of values within `±2σ`
- `dte/evaluation/uncertainty.py` now adds empirical coverage, calibration gap, Gaussian NLL, and variance-collapse diagnostics
- `scripts/evaluate_universal.py` now reports these metrics from rollout samples drawn from the variational encoder

Status:

- `Implemented`

### Phase 1 acceptance criteria

Planned acceptance criteria:

- existing systems still train
- `cstr`, `heat_exchanger`, `two_tank` run through new abstractions
- universal model still works

Current assessment:

- existing systems still train under the old defaults
- `cstr`, `heat_exchanger`, and `two_tank` now resolve through `ProcessUnitSpec`
- the universal model and evaluation path now include the new Phase 1 hooks

Overall status:

- `Implemented`

## Phase 2. Add Unit Adapters And Family-Level Conditioning

### 1. Add unit-family taxonomy

Current state:

- `ProcessUnitSpec` now exposes `family`, `subtype`, `unit_type`, and `law_tags`
- the default `cstr`, `heat_exchanger`, and `two_tank` configs now declare this metadata
- `dte/data/multi_system_dataset.py` now materializes:
  - `family_id`
  - `subtype_id`
  - `law_tag_mask`
  - `conditioning_value_id`
  - `parameter_law_tag_id`
- system configs now carry structured `conditioning_tags`:
  - `reaction_class`
  - `thermo_regime`
  - `bio_model_family`
  - `operating_regime`

Status:

- `Implemented`

### 2. Add adapter layers

Current state:

- `dte/models/universal_digital_twin.py` now includes optional residual bottleneck adapters for:
  - encoder features
  - latent drift / neural-CDE path features
  - decoder group features
- adapter training is exposed through `UniversalDigitalTwin.trainable_filter_spec(mode="adapters")`
- universal configs now expose `model.adapters`

Status:

- `Implemented`

### 3. Add parameter and law embeddings

Current state:

- the universal model already uses:
  - learned `system_id` embeddings
  - numeric `SystemSpec` descriptor embeddings
  - state-group-kind embeddings
- Phase 2 adds:
  - family embeddings
  - subtype embeddings
  - pooled law-tag embeddings
  - structured conditioning-tag embeddings
  - parameter-law-tag summaries fused into the universal context
- universal encode/decode/drift paths now condition on this richer context directly

Status:

- `Implemented`

### 4. Add calibration pipeline

Current state:

- `dte/calibration/unit_calibration.py` now provides:
  - target-model initialization from a pretrained multi-system checkpoint
  - adapter-only or full calibration filter specs
  - optional normalization recalibration
  - optional selected physics-parameter calibration
- `scripts/calibrate_unit.py` is the calibration entry point
- `UniversalDigitalTwin` now carries calibration tables for:
  - normalization offsets/scales
  - per-parameter additive bias terms with per-system masks

Status:

- `Implemented`

## Phase 3. Introduce Flowsheet Graph Modeling

Planned deliverables:

- flowsheet schema
- graph dataset
- graph simulator
- recycle-loop support
- plant-level losses

Current state:

- `dte/flowsheet/schema.py` now defines:
  - `StreamSpec`
  - `FlowsheetSpec`
  - source/sink validation
  - recycle-loop detection on the internal unit graph
- `dte/flowsheet/types.py` defines the external node sentinels and allowed stream kinds
- `dte/flowsheet/examples.py` now provides two initial demo graphs:
  - exchanger -> reactor -> tank
  - reactor -> separator -> recycle -> reactor
- `dte/data/flowsheet_dataset.py` now provides:
  - `FlowsheetGraphMetadata`
  - `FlowsheetTrajectoryDataset`
  - HDF5 save/load support with topology metadata
  - preserved `seq_len` / `stride` roundtrips
- `dte/flowsheet/synthetic.py` now provides a lightweight synthetic data path so Phase 3 can be trained and tested without adding a full plant simulator stack yet
- `dte/models/flowsheet_model.py` now provides a first shared graph model with:
  - shared unit backbone
  - stream-message aggregation
  - graph-level update block
  - multi-unit rollout
  - recycle-aware delay handling through per-stream lag metadata
- `dte/evaluation/flowsheet_metrics.py` now provides:
  - stream consistency loss
  - unit-output consistency loss
  - plant-balance proxy loss
  - rollout stability penalty
- `dte/training/flowsheet_trainer.py` now provides a runnable train/validate loop for the graph model
- targeted Phase 3 tests now cover:
  - schema validation
  - dataset/HDF5 roundtrip
  - rollout on both example flowsheets
  - recycle-delay behavior
  - one-epoch train/validate execution

Status:

- `Implemented`

Notes:

- this is a thin first slice, not a full Aspen-style simulator
- plant balance is currently a topology-aware proxy consistency term, not a thermodynamically rigorous plant residual layer
- demo/training data is currently synthetic; there is not yet a generic flowsheet data-generation CLI

## Phase 4. Add Modular Law Layers

Planned deliverables:

- `laws/chemistry.py`
- `laws/thermo.py`
- `laws/biology.py`
- unit integration hooks

Current state:

- `dte/laws/chemistry.py` now provides:
  - Arrhenius and power-law rate helpers
  - stoichiometric source terms
  - heat-of-reaction hooks
  - `ChemistryLaw`
- `dte/laws/thermo.py` now provides:
  - heat-capacity correlations
  - enthalpy-like transforms
  - simple equilibrium indicator placeholders
  - `ThermoLaw`
- `dte/laws/biology.py` now provides:
  - Monod growth
  - substrate uptake
  - oxygen transfer
  - inhibition handling
  - `BiologyLaw`
- `dte/laws/base.py` and `dte/laws/integration.py` now provide:
  - `LawModule`
  - `UnitLawBundle`
  - config-driven bundle construction
  - law residual aggregation
  - mechanistic delta / feature-vector hooks
  - `LawAugmentedPhysicsLoss`
- `dte/physics/registry.py` now augments registered physics losses and diagnostics when a config enables `laws`
- example entry points now exist for both chemistry and biology:
  - `configs/cstr_law_example.yaml`
  - `configs/bioreactor_law_example.yaml`
  - `dte/laws/examples.py`
- targeted tests now cover:
  - low-level law helpers
  - bundle features and mechanistic deltas
  - chemistry example integration
  - biology example integration
  - law-augmented physics registry behavior

Status:

- `Implemented`

Notes:

- the default system configs remain backward-compatible; law augmentation is opt-in through `laws.enabled`
- the current integration hooks expose explicit features, mechanistic deltas, and residual terms, but they are not yet directly wired into the neural model architectures as extra input channels

## Phase 5. Build Customer Adaptation Workflow

Planned deliverables:

- onboarding schema
- adaptation script
- validation report generator

Current state:

- `dte/customer/onboarding_schema.py` now defines a validated onboarding payload for:
  - unit list
  - stream list
  - controls
  - disturbances
  - measurements
  - known laws
  - operating ranges
- `dte/customer/template_matching.py` now ranks:
  - registered unit templates
  - Phase 3 example flowsheet templates
- `dte/customer/adaptation.py` now orchestrates:
  - template matching
  - pretrained-weight initialization
  - adapter/calibration fine-tuning
  - report generation
- `scripts/adapt_customer.py` is now the Phase 5 entry point
- `dte/customer/reporting.py` now generates:
  - forecast metrics
  - rollout metrics
  - control sensitivity metrics
  - uncertainty summaries
  - constraint summaries

What is still missing:

- flowsheet adaptation is not yet wired through the customer CLI
- customer-specific UI/demo surfaces are still separate future work

Status:

- `Implemented`

## Phase 6. Website / Demo App

Planned deliverables:

- polished demo backend endpoints
- frontend with multiple demos
- flowsheet demo

Current state:

- `dte/demo/engine.py` now provides shared demo runtime helpers for:
  - deterministic simulator rollouts
  - model-backed or simulator-ensemble rollouts
  - scenario comparison
  - lightweight control-sequence optimisation
  - demo catalog and flowsheet preview generation
- `dte/api/service.py` still exposes:
  - `/health`
  - `/predict`
  - `/ensemble`
  - `/steady_state`
- `dte/api/service.py` now also exposes:
  - `/demo/catalog`
  - `/demo/simulate`
  - `/demo/rollout`
  - `/demo/optimize_control`
  - `/demo/compare_scenarios`
- `app/demo_app.py` now provides a dedicated multi-demo Streamlit experience for:
  - CSTR
  - heat exchanger
  - two-tank system
- `configs/demo_app.yaml` now defines the demo catalog, target states, horizons, and highlighted state channels
- deployment files still exist: `Dockerfile`, `docker-compose.yml`

What is still missing:

- flowsheet support is currently a preview/catalog surface, not a fully interactive plant-graph simulator in the demo UI
- control optimisation is a lightweight demo optimiser, not the Phase 7 generic MPC interface

Status:

- `Implemented`

## Phase 7. MPC And DRL Readiness

Planned deliverables:

- MPC wrapper
- RL env wrapper
- state correction utilities

Current state:

- `dte/control/mpc.py` still provides the original sampling/CEM MPC controller
- `dte/control/pid.py` still provides the PID baseline
- `scripts/run_mpc.py` still runs the legacy MPC comparison loop
- `dte/training/online.py` still provides the heavier online fine-tuning path
- `dte/control/mpc_interface.py` now provides:
  - `ProcessMPCInterface`
  - current state estimate access
  - rollout under candidate actions
  - custom cost hooks
  - custom constraint hooks
  - random-shooting optimisation for controller prototypes
- `dte/control/rl_env.py` now provides:
  - `ProcessControlEnv`
  - Gymnasium-style `reset()` / `step()` signatures
  - observation and action spaces
  - disturbance schedule support
- `dte/control/state_correction.py` now provides:
  - measurement assimilation
  - exponential filtering
  - latent refresh through the encoder when a model is attached
  - one-step latent prediction hooks
- `dte/evaluation/control_metrics.py` now provides:
  - closed-loop cost summaries
  - constraint violation summaries
  - disturbance sensitivity metrics
  - mismatch robustness metrics

What is still intentionally thin:

- the new MPC runtime uses random-shooting rather than a full solver family
- the RL wrapper is Gymnasium-style without taking a hard dependency on `gymnasium`
- no bundled RL training algorithm stack landed in this phase

Status:

- `Implemented`

## Phase 8. Later Distributed / Transport-Aware Units

Current state:

- no distributed-unit package
- no compartment-chain abstraction beyond current lumped simulators
- no operator/neural-field model layer

Status:

- `Missing`

## Cross-Cutting Gaps That Matter Before New Phases

These are not tied to one roadmap phase, but they should be resolved before large feature expansion:

1. Unify the semantics between the single-system and universal model paths.
   Right now grouped typed semantics live mostly in the universal stack.

2. Make training config behavior fully explicit.
   `scripts/train.py` currently overrides loaded YAML values programmatically.

3. Fix the existing correctness bugs before layering on more complexity.
   The trainer SDE-KL bug and PID path bug should not be carried into later phases.

4. Repair the dataset normalization contract.
   Generated HDF5 files and `TrajectoryDataset` disagree about parameter normalization support.

5. Decide whether the universal model is a research sidecar or the main foundation path.
   The roadmap assumes it becomes the foundation path, but the repo still treats it as a parallel baseline.

## Recommended Interpretation Of The Current Repo

The cleanest reading of the repository today is:

- the single-system path is the production-capable unit-model stack
- the universal grouped path is the early foundation-model research stack

The roadmap should therefore build from that split rather than pretending the repo is still only a CSTR-to-generalization refactor. The next practical milestone is to merge more of the universal typed semantics back into the main unit-model abstractions before attempting flowsheets or customer adaptation layers.
