# Architecture Overview

## Purpose

This document describes the current implemented architecture of Digital Twin Engine.
It is a code-aligned overview, not a roadmap sketch. The code remains the source
of truth when this document lags.

The platform is organized around three major modeling paths:

- a first-class single-system digital twin path
- a first-class universal mixed-system path
- a thin-slice flowsheet graph path

Cross-cutting layers provide physics/law integration, customer adaptation,
state correction, control-facing interfaces, and serving surfaces.

## Documentation Views

Use the companion diagrams with distinct roles:

- [`diagram.mmd`](diagram.mmd): structural architecture map
- [`data_flow.mmd`](data_flow.mmd): end-to-end offline training, customer
  deployment, and target-state movement flow
- [`repo_structure_target.md`](repo_structure_target.md): repo-aligned package
  structure target and migration map
- [`repo_refactor_plan.md`](repo_refactor_plan.md): phased execution plan for
  moving from the current layout to the target package structure

---

## Current Platform At A Glance

### 1. Process Specification And Schema

Core abstractions:

- `SystemSpec`
- `ProcessUnitSpec`
- `StateChannel`
- `SignalChannel`
- `ParameterDescriptor`
- `TopologyPort`
- `StreamSpec`
- `FlowsheetSpec`

This layer provides typed process metadata, dimensions, bounds, normalization,
state grouping, law tags, and topology-facing stream definitions.

### 2. Simulation And Registry Layer

Core abstractions:

- `ProcessSimulator`
- `get_system_spec(config)`
- `get_simulator(name, config)`
- `list_systems()`

Registered simulators provide the physical process interface used for data
generation, steady-state calculation, demo rollouts, and control baselines.

### 3. Single-System Digital Twin Path

Primary modules:

- `Encoder`
- `GroupedStateEncoder`
- `Decoder`
- `LatentSDE`
- `DigitalTwin`
- `Trainer`

This is the mature unit-level path used for one-system training, inference,
stochastic rollout, uncertainty via ensembles, and legacy-compatible control
workflows.

### 4. Universal Shared-Checkpoint Path

Primary modules:

- `MultiSystemTrajectoryDataset`
- `UniversalSystemMetadata`
- `UniversalDigitalTwin`
- `UniversalTrainer`
- `BottleneckAdapter`

This is the foundation-style path for mixed-system training. It uses padded
metadata tables, typed state-group conditioning, optional law/channel
conditioning, and low-parameter adapters for target-unit calibration.

### 5. Physics And Law Layer

Primary modules:

- `PhysicsLoss` implementations in `dte/physics/`
- physics registry in `dte/physics/registry.py`
- `LawModule` and `UnitLawBundle` in `dte/laws/`

Physics losses remain system-specific and are resolved through registries.
Law bundles currently augment diagnostics, residual losses, and conditioning
features; they are not yet a full mechanistic replacement for learned dynamics.

### 6. Data Layer

Primary modules:

- `TrajectoryDataset`
- `GenericDataGenerator`
- `RealDataIngestion`
- `MultiSystemTrajectoryDataset`
- `FlowsheetTrajectoryDataset`

The repo supports physical-unit datasets for:

- single-system synthetic training
- universal mixed-system training
- customer CSV/Parquet ingestion
- synthetic flowsheet experiments

### 7. Customer Adaptation And Onboarding

Primary modules:

- `UnitCalibrator`
- `run_customer_adaptation(...)`
- onboarding helpers in `dte/api/onboarding.py`
- template matching in `dte/customer/template_matching.py`

This is a first-class unit-adaptation path. Flowsheet onboarding exists at the
template-matching level, but flowsheet calibration is not yet orchestrated as a
mainline workflow.

### 8. Flowsheet Graph Path

Primary modules:

- flowsheet schema and examples in `dte/flowsheet/`
- `FlowsheetTrajectoryDataset`
- `FlowsheetModel`
- `FlowsheetTrainer`

This path is real code and test-backed, but it is still thin-slice in scope:
synthetic datasets, small graphs, smoke-style training loops, and no canonical
production deployment path.

### 9. State Correction And Control Interfaces

Primary modules:

- `StateCorrectionHook`
- `ProcessMPCInterface`
- `ProcessControlEnv`

These provide control-facing abstractions for:

- measurement assimilation
- latent/state correction
- candidate rollout and optimization
- RL-style environment wrapping

The older MPC/PID stack remains in the repo as a legacy compatibility path.

### 10. Serving And User Interfaces

Primary surfaces:

- FastAPI service in `dte/api/service.py`
- browser frontend in `frontend/`
- Streamlit demo in `app/demo_app.py`
- Streamlit dashboard in `app/dashboard.py`

The API now exposes more than basic prediction. It includes:

- deterministic prediction
- stochastic ensemble prediction
- steady-state simulation
- demo catalog/bootstrap endpoints
- scenario comparison
- lightweight control optimization
- customer onboarding and adaptation job endpoints

---

## Supported Paths

### First-Class

- single-system model training and inference
- universal mixed-system training
- unit-level customer calibration
- FastAPI inference and demo APIs
- generic state-correction and control interfaces

### Thin-Slice / Experimental

- flowsheet graph modeling
- flowsheet synthetic datasets and smoke workflows
- deeper law-driven representation learning

### Legacy / Compatibility

- legacy MPC and PID tooling
- older import shims retained for stable script behavior

---

## Actual Training And Deployment Flows

The concise lifecycle view is maintained in [`data_flow.mmd`](data_flow.mmd).
The flows below are the same paths written in text so the document remains
readable without rendering Mermaid.

### Single-System Path

`System YAML -> SystemSpec -> TrajectoryDataset -> DigitalTwin -> Trainer -> evaluate/API`

### Universal Path

`multiple system YAMLs + datasets -> MultiSystemTrajectoryDataset -> UniversalDigitalTwin -> UniversalTrainer -> evaluation -> unit calibration`

### Customer Unit Adaptation

`customer upload -> onboarding preview -> unit template match -> processed dataset -> UnitCalibrator / customer adaptation -> validation report -> planning workspace`

### Flowsheet Path

`FlowsheetSpec -> synthetic flowsheet dataset -> FlowsheetModel -> FlowsheetTrainer -> smoke/evaluation artifacts`

These are parallel capabilities. They are not one mandatory linear pipeline.

---

## Design Principles

- keep process semantics explicit through typed specs and registries
- separate first-class paths from thin-slice experiments
- preserve physical-unit data at dataset boundaries
- keep physics and law integration modular
- expose control-facing interfaces without forcing one control stack
- favor reusable runtime surfaces over system-specific branching

---

## Summary

Digital Twin Engine is currently best understood as a multi-path platform:

- a mature unit digital twin engine
- a shared universal foundation path
- a partial but real flowsheet graph stack
- a unit-focused customer adaptation workflow

It does not need a structural rewrite. The highest-value work is keeping docs,
runtime contracts, and support boundaries aligned with the code that already
exists.
