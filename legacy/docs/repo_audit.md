# Repository Audit

Date: 2026-04-15

Scope: current code under `dte/`, `scripts/`, `app/`, `frontend/`, `configs/`, and repo-level operational docs.

## Status Note

This file is a point-in-time architecture audit.

- Code is the primary source of truth.
- `WORKFLOW.md` is the operational entry point for humans.
- `plan.md` is now best treated as a historical draft and roadmap sketch, not an execution checklist.

## Executive Summary

The repository is materially farther along than the early roadmap assumptions.

The current platform already includes:

1. A mature single-system latent neural SDE stack centered on `DigitalTwin`
2. A shared mixed-system universal stack centered on `UniversalDigitalTwin`
3. Unit calibration and customer adaptation workflows
4. A thin-slice flowsheet graph stack
5. Modular law bundles that augment physics losses
6. Demo/API surfaces
7. Generic control-facing interfaces plus a legacy MPC/PID path

The main cleanup needs are now:

- stale documentation
- duplicate universal evaluation logic
- broken legacy/runtime branches with weak test coverage
- generated or dead repo artifacts
- clearer support boundaries between first-class and thin-slice subsystems

## Current Architecture Map

### 1. Unit model path

- System abstraction: `dte/simulators/base.py`
- Registry resolution: `dte/simulators/registry.py`
- Single-system model: `dte/models/unit/digital_twin.py`
- Trainer: `dte/training/unit/trainer.py`
- Evaluation: `scripts/evaluate.py`

`SystemSpec` remains the stable base abstraction, and `ProcessUnitSpec` now extends it with richer metadata without breaking older callers.

### 2. Typed process metadata

- Channel schema: `dte/core/state_schema.py`
- Compatibility re-export: `dte/core/process_unit_spec.py`
- Registry parsing of typed metadata: `dte/simulators/registry.py`

The repo now has channel-level roles, bounds, parameter descriptors, law tags, and topology ports. Older roadmap assumptions that these were missing are no longer correct.

### 3. Universal shared-checkpoint path

- Dataset: `dte/data/datasets/universal_unit_dataset.py`
- Model: `dte/models/universal/digital_twin.py`
- Trainer: `dte/training/universal/trainer.py`
- Train/eval scripts: `scripts/train_universal.py`, `scripts/evaluate_universal.py`

This path is the foundation-style mixed-system route. It uses padding, masks, typed group metadata, and richer conditioning tables.

### 4. Calibration and customer workflow

- Calibration: `dte/calibration/unit_calibration.py`
- Customer onboarding: `dte/customer/onboarding_schema.py`
- Template matching: `dte/customer/template_matching.py`
- Reporting: `dte/customer/reporting.py`
- CLI orchestration: `scripts/adapt_customer.py`

Unit adaptation is implemented end to end. Flowsheet adaptation is not yet wired through the main adaptation entry point.

### 5. Flowsheet stack

- Schema/types/examples: `dte/flowsheet/`
- Dataset: `dte/data/datasets/flowsheet_dataset.py`
- Model: `dte/models/flowsheet/flowsheet_model.py`
- Trainer: `dte/training/flowsheet/trainer.py`

This is real code, not a placeholder, but it is still a thin slice: synthetic data, small graphs, and smoke-style workflows rather than a production-grade plant stack.

### 6. Law-layer stack

- Base modules: `dte/laws/base.py`
- Chemistry/thermo/biology bundles: `dte/laws/*.py`
- Config-driven integration: `dte/laws/integration.py`

The law layer exists and is test-backed. Today it augments the physics-loss path rather than acting as a deeply integrated feature source for the encoder/decoder.

### 7. Demo and serving surfaces

- Streamlit dashboard: `app/dashboard.py`
- Streamlit demo app: `app/demo_app.py`
- FastAPI service: `dte/api/service.py`
- Browser frontend: `frontend/`

There are multiple user-facing surfaces, and they are not yet collapsed to one canonical product path.

### 8. Control path

- Legacy MPC path: `dte/control/mpc.py`, `scripts/run_mpc.py`
- New generic interfaces: `dte/control/mpc_interface.py`, `dte/control/rl_env.py`, `dte/control/state_correction.py`

The new interfaces are the cleaner abstraction boundary. The legacy path remains for backwards compatibility and comparison.

## What Is Fully Real Today

The following roadmap items are already implemented in some form:

- typed channel/process metadata
- `ProcessUnitSpec`
- reusable physics constraint helpers in `dte/physics/constraints.py`
- shared universal conditioning and adapters
- flowsheet schema/model/trainer skeleton
- modular law bundles
- unit calibration
- customer onboarding/template matching/report generation
- control sensitivity and uncertainty diagnostics for the universal path

## What Is Still Partial

- Flowsheet workflow is thin-slice and smoke-oriented, not yet a first-class CLI/product path
- Law bundles are not yet injected directly into model representation learning
- Customer adaptation only orchestrates unit calibration, not flowsheet calibration
- Multiple demo surfaces exist without a single supported default
- Legacy control tooling still needs more direct workflow coverage than the test suite currently provides

## Immediate Cleanup Targets

The highest-value cleanup work is:

1. Keep docs aligned with the code and stop treating roadmap assumptions as facts
2. Remove committed generated artifacts and compatibility noise where safe
3. Fix broken but lightly tested runtime branches
4. Consolidate duplicated universal evaluation/reporting logic
5. Mark thin-slice subsystems explicitly so they are not mistaken for fully productized paths

## Known Structural Cautions

- `plan.md` still contains “add this directory/file” steps for code that already exists
- passing tests do not imply all user-facing scripts are safe; some legacy/script branches have historically drifted outside direct coverage
- compatibility shims such as `dte/data/generation.py` are intentional for import stability, but they should stay clearly labeled as compatibility surfaces

## Bottom Line

This repo does not need a rewrite.

It needs a cleanup pass that treats the current architecture as established, removes drift, and sharpens the boundaries between:

- first-class supported paths
- thin-slice experimental paths
- legacy compatibility paths
