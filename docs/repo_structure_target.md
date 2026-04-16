# Repo Structure Target And Migration Map

Date: 2026-04-16

This document turns the proposed folder tree into a repo-aligned target
structure and a migration map from the current layout.

It is intentionally conservative. The goal is to improve package clarity without
renaming the codebase into an architecture the repo does not actually support
yet.

---

## Bottom Line

The proposed structure is directionally good at the domain level, but it does
not align cleanly with the current repository in five important ways:

1. The codebase is rooted under `dte/`, not a flat top-level `repo/core`,
   `repo/models`, `repo/api` layout.
2. `simulators/` and `physics/` are first-class architectural boundaries today
   and should remain explicit.
3. `scripts/` should remain the CLI layer instead of being absorbed into the
   package.
4. `frontend/` and `app/` are two real UI surfaces today and should not be
   collapsed into `web/` yet.
5. `estimation/`, `flowsheet_calibration.py`, and `models/distributed/` are
   ahead of current implementation maturity.

The right move is an incremental internal cleanup inside `dte/`, not a full
top-level repo reshape.

Current progress:

- Phase 1 canonical data paths have landed under `dte/data/datasets/`,
  `dte/data/generators/`, and `dte/data/ingestion/`.
- Phase 2 canonical model paths have landed under `dte/models/unit/`,
  `dte/models/universal/`, and `dte/models/flowsheet/`.
- Phase 3 canonical training paths have landed under `dte/training/unit/`,
  `dte/training/universal/`, `dte/training/flowsheet/`, and
  `dte/training/shared/`.
- Phase 4 import normalization has landed across runtime code, docs examples,
  and install verification.
- The older top-level `dte/data/*.py` and `dte/models/*.py` modules still
  exist as compatibility paths during the transition.
- The older top-level `dte/training/*.py` modules also still exist as
  compatibility paths during the transition.

---

## Recommended Target Structure

```text
repo/
├── dte/
│   ├── core/
│   ├── simulators/
│   ├── physics/
│   ├── laws/
│   ├── flowsheet/
│   ├── data/
│   │   ├── datasets/
│   │   ├── generators/
│   │   └── ingestion/
│   ├── models/
│   │   ├── unit/
│   │   ├── universal/
│   │   └── flowsheet/
│   ├── training/
│   │   ├── unit/
│   │   ├── universal/
│   │   ├── flowsheet/
│   │   └── shared/
│   ├── calibration/
│   ├── customer/
│   ├── control/
│   ├── evaluation/
│   ├── api/
│   ├── demo/
│   ├── autoresearch/
│   └── utils/
├── scripts/
├── configs/
├── frontend/
├── app/
├── docs/
└── tests/
```

This keeps the real package root, preserves the simulator and physics
boundaries, and only introduces subpackages where the repo already has enough
mass to justify them.

---

## Alignment With The Proposed Structure

### Strong Alignment

- `core/`
- `laws/`
- `flowsheet/`
- `calibration/`
- `control/`
- `customer/`
- `api/`

These are already real architectural slices in the current repo.

### Partial Alignment

- `models/`
- `training/`
- `datasets/`

These are real slices, but they should be reorganized inside `dte/`, not moved
to repo root.

### Poor Alignment / Premature

- `models/distributed/`
- `calibration/flowsheet_calibration.py`
- `estimation/`
- `api/routes_simulate.py`, `routes_optimize.py`, `routes_compare.py`
- `web/`

These imply capabilities or packaging boundaries that are not first-class in the
current codebase.

---

## What Should Not Change Yet

Do not make these changes as part of a structure cleanup:

- Do not remove `dte/simulators/`.
- Do not remove `dte/physics/`.
- Do not rename `DigitalTwin` into `UnitModel` just to match a diagram.
- Do not replace generic data generation with per-system generator folders.
- Do not introduce a top-level `estimation/` package before there is a real
  estimator stack beyond state correction hooks.
- Do not add flowsheet customer-calibration modules until the workflow exists as
  more than template assembly and thin-slice experiments.

---

## Migration Map

Status legend:

- `Keep`: leave where it is.
- `Move`: reasonable target in a cleanup pass.
- `Later`: only after a real capability expansion.

### Core, Simulators, Physics, Laws, Flowsheet

| Current path | Target path | Status | Notes |
| --- | --- | --- | --- |
| `dte/core/state_schema.py` | `dte/core/state_schema.py` | Keep | Already the right boundary; do not oversplit into `channel_types.py` and `descriptors.py` yet |
| `dte/core/process_unit_spec.py` | `dte/core/process_unit_spec.py` | Keep | Already aligned |
| `dte/simulators/base.py` | `dte/simulators/base.py` | Keep | Core abstraction boundary |
| `dte/simulators/registry.py` | `dte/simulators/registry.py` | Keep | Central registry boundary |
| `dte/simulators/cstr.py` | `dte/simulators/cstr.py` | Keep | Concrete system simulator |
| `dte/simulators/heat_exchanger.py` | `dte/simulators/heat_exchanger.py` | Keep | Concrete system simulator |
| `dte/simulators/two_tank.py` | `dte/simulators/two_tank.py` | Keep | Concrete system simulator |
| `dte/simulators/bioreactor_compartment.py` | `dte/simulators/bioreactor_compartment.py` | Keep | Concrete system simulator |
| `dte/simulators/isothermal_cstr.py` | `dte/simulators/isothermal_cstr.py` | Keep | Concrete system simulator |
| `dte/simulators/separator.py` | `dte/simulators/separator.py` | Keep | Concrete system simulator |
| `dte/simulators/storage_tank.py` | `dte/simulators/storage_tank.py` | Keep | Concrete system simulator |
| `dte/physics/base.py` | `dte/physics/base.py` | Keep | First-class architecture layer |
| `dte/physics/registry.py` | `dte/physics/registry.py` | Keep | First-class architecture layer |
| `dte/physics/constraints.py` | `dte/physics/constraints.py` | Keep | Reusable constraints fit here better than in `laws/constraints.py` |
| `dte/physics/cstr.py` | `dte/physics/cstr.py` | Keep | System-specific physics |
| `dte/physics/heat_exchanger.py` | `dte/physics/heat_exchanger.py` | Keep | System-specific physics |
| `dte/physics/two_tank.py` | `dte/physics/two_tank.py` | Keep | System-specific physics |
| `dte/physics/bioreactor_compartment.py` | `dte/physics/bioreactor_compartment.py` | Keep | System-specific physics |
| `dte/physics/isothermal_cstr.py` | `dte/physics/isothermal_cstr.py` | Keep | System-specific physics |
| `dte/physics/separator.py` | `dte/physics/separator.py` | Keep | System-specific physics |
| `dte/physics/storage_tank.py` | `dte/physics/storage_tank.py` | Keep | System-specific physics |
| `dte/physics/conservation.py` | `dte/physics/conservation.py` | Keep | Backward-compatibility shim |
| `dte/laws/base.py` | `dte/laws/base.py` | Keep | Correct boundary |
| `dte/laws/chemistry.py` | `dte/laws/chemistry.py` | Keep | Correct boundary |
| `dte/laws/thermo.py` | `dte/laws/thermo.py` | Keep | Correct boundary |
| `dte/laws/biology.py` | `dte/laws/biology.py` | Keep | Correct boundary |
| `dte/laws/integration.py` | `dte/laws/integration.py` | Keep | Correct boundary |
| `dte/laws/examples.py` | `dte/laws/examples.py` | Keep | Example/support file |
| `dte/flowsheet/schema.py` | `dte/flowsheet/schema.py` | Keep | Main flowsheet schema boundary |
| `dte/flowsheet/types.py` | `dte/flowsheet/types.py` | Keep | Current flowsheet support file |
| `dte/flowsheet/synthetic.py` | `dte/flowsheet/synthetic.py` | Keep | Thin-slice synthetic path |
| `dte/flowsheet/examples.py` | `dte/flowsheet/examples.py` | Keep | Example/support file |

### Data

| Current path | Target path | Status | Notes |
| --- | --- | --- | --- |
| `dte/data/dataset.py` | `dte/data/datasets/unit_dataset.py` | Move | Clearer name once dataset subpackages exist |
| `dte/data/multi_system_dataset.py` | `dte/data/datasets/universal_unit_dataset.py` | Move | Aligns with universal path |
| `dte/data/flowsheet_dataset.py` | `dte/data/datasets/flowsheet_dataset.py` | Move | Aligns with flowsheet path |
| `dte/data/generation_generic.py` | `dte/data/generators/generic.py` | Move | This is the preferred generator boundary |
| `dte/data/generation.py` | `dte/data/generation.py` | Keep | Keep as the stable HDF5 loading / compatibility helper; it blocks a sibling `generation/` package name |
| `dte/data/real_data.py` | `dte/data/ingestion/real_data.py` | Move | Better matches role |

### Models

| Current path | Target path | Status | Notes |
| --- | --- | --- | --- |
| `dte/models/encoder.py` | `dte/models/unit/encoder.py` | Move | Real single-system component |
| `dte/models/grouped_encoder.py` | `dte/models/unit/grouped_encoder.py` | Move | Real single-system component |
| `dte/models/decoder.py` | `dte/models/unit/decoder.py` | Move | Real single-system component |
| `dte/models/latent_sde.py` | `dte/models/unit/latent_sde.py` | Move | Keep actual name; do not rename to `latent_dynamics.py` unless the abstraction changes |
| `dte/models/digital_twin.py` | `dte/models/unit/digital_twin.py` | Move | Keep current class naming |
| `dte/models/universal_digital_twin.py` | `dte/models/universal/digital_twin.py` | Move | Split by path, not by marketing label |
| `dte/models/flowsheet_model.py` | `dte/models/flowsheet/flowsheet_model.py` | Move | Aligns with real flowsheet slice |

### Training

| Current path | Target path | Status | Notes |
| --- | --- | --- | --- |
| `dte/training/trainer.py` | `dte/training/unit/trainer.py` | Move | Keeps library code inside package |
| `dte/training/universal_trainer.py` | `dte/training/universal/trainer.py` | Move | Keeps library code inside package |
| `dte/training/flowsheet_trainer.py` | `dte/training/flowsheet/trainer.py` | Move | Thin-slice but real |
| `dte/training/losses.py` | `dte/training/shared/losses.py` | Move | Shared across paths |
| `dte/training/config_resolution.py` | `dte/training/shared/config_resolution.py` | Move | Shared support module |
| `dte/training/transfer.py` | `dte/training/shared/transfer.py` | Move | Shared adaptation support |
| `dte/training/online.py` | `dte/training/online.py` | Keep | Already a clear top-level training utility |

### Calibration, Customer, Control, Evaluation

| Current path | Target path | Status | Notes |
| --- | --- | --- | --- |
| `dte/calibration/unit_calibration.py` | `dte/calibration/unit_calibration.py` | Keep | Real first-class calibration module |
| `dte/customer/adaptation.py` | `dte/customer/adaptation.py` | Keep | Current workflow orchestrator; unit-first |
| `dte/customer/onboarding_schema.py` | `dte/customer/onboarding_schema.py` | Keep | Real onboarding boundary |
| `dte/customer/template_matching.py` | `dte/customer/template_matching.py` | Keep | Real onboarding boundary |
| `dte/customer/reporting.py` | `dte/customer/reporting.py` | Keep | Rename to `validation_report.py` only if the scope stays narrow |
| `dte/control/mpc.py` | `dte/control/mpc.py` | Keep | Existing runtime |
| `dte/control/mpc_interface.py` | `dte/control/mpc_interface.py` | Keep | Existing interface layer |
| `dte/control/pid.py` | `dte/control/pid.py` | Keep | Legacy compatibility path |
| `dte/control/rl_env.py` | `dte/control/rl_env.py` | Keep | Existing evaluation/training surface |
| `dte/control/state_correction.py` | `dte/control/state_correction.py` | Keep | Do not move into `estimation/` yet |
| `dte/evaluation/control_metrics.py` | `dte/evaluation/control_metrics.py` | Keep | Real evaluation slice |
| `dte/evaluation/control_sensitivity.py` | `dte/evaluation/control_sensitivity.py` | Keep | Real evaluation slice |
| `dte/evaluation/flowsheet_metrics.py` | `dte/evaluation/flowsheet_metrics.py` | Keep | Real evaluation slice |
| `dte/evaluation/uncertainty.py` | `dte/evaluation/uncertainty.py` | Keep | Real evaluation slice |
| `dte/evaluation/universal.py` | `dte/evaluation/universal.py` | Keep | Real evaluation slice |

### API, Demo, Autoresearch, Utilities

| Current path | Target path | Status | Notes |
| --- | --- | --- | --- |
| `dte/api/models.py` | `dte/api/models.py` | Keep | Rename to `schemas.py` only if route modules are actually split |
| `dte/api/onboarding.py` | `dte/api/onboarding.py` | Keep | Real API support boundary |
| `dte/api/service.py` | `dte/api/service.py` | Keep | Split into route modules later if file size becomes the real problem |
| `dte/demo/engine.py` | `dte/demo/engine.py` | Keep | Real demo/runtime support layer |
| `dte/autoresearch/workflow.py` | `dte/autoresearch/workflow.py` | Keep | Separate operating mode; should stay explicit |
| `dte/utils/logging.py` | `dte/utils/logging.py` | Keep | Utility layer |
| `dte/utils/plotting.py` | `dte/utils/plotting.py` | Keep | Utility layer |
| `dte/utils/runtime.py` | `dte/utils/runtime.py` | Keep | Utility layer |

### Top-Level Repo Surfaces

| Current path | Target path | Status | Notes |
| --- | --- | --- | --- |
| `scripts/train.py` | `scripts/train.py` | Keep | CLI for single-system training |
| `scripts/train_universal.py` | `scripts/train_universal.py` | Keep | CLI for universal training |
| `scripts/evaluate.py` | `scripts/evaluate.py` | Keep | CLI for evaluation |
| `scripts/evaluate_universal.py` | `scripts/evaluate_universal.py` | Keep | CLI for universal evaluation |
| `scripts/generate_data.py` | `scripts/generate_data.py` | Keep | CLI for generic system generation |
| `scripts/ingest_real_data.py` | `scripts/ingest_real_data.py` | Keep | CLI for ingestion |
| `scripts/calibrate_unit.py` | `scripts/calibrate_unit.py` | Keep | CLI for calibration |
| `scripts/adapt_customer.py` | `scripts/adapt_customer.py` | Keep | CLI for customer workflow |
| `scripts/run_mpc.py` | `scripts/run_mpc.py` | Keep | CLI for control |
| `scripts/agent.py` | `scripts/agent.py` | Keep | Protected autoresearch surface |
| `scripts/autoresearch.py` | `scripts/autoresearch.py` | Keep | Protected autoresearch surface |
| `configs/` | `configs/` | Keep | Correct top-level boundary |
| `frontend/` | `frontend/` | Keep | Separate browser frontend |
| `app/` | `app/` | Keep | Separate Streamlit surfaces |
| `tests/` | `tests/` | Keep | Correct top-level boundary |
| `docs/` | `docs/` | Keep | Correct top-level boundary |

---

## Proposed-But-Not-Yet-Real Areas

These should be tracked as future capability targets, not introduced as empty
folders today:

| Proposed area | Current closest equivalent | Recommendation |
| --- | --- | --- |
| `models/distributed/` | none | Add only when distributed-unit modeling becomes a real supported path |
| `calibration/flowsheet_calibration.py` | none | Add only after flowsheet customer adaptation is first-class |
| `estimation/` | `dte/control/state_correction.py` | Add only after there is a real estimator/filter stack |
| `api/routes_*.py` | `dte/api/service.py` | Split only when route ownership or file size forces it |
| `web/` | `frontend/` and `app/` | Unify only after product surfaces are intentionally merged |

---

## Suggested Refactor Order

If the repo ever does a structure cleanup, the lowest-risk order is:

1. Split `dte/models/` into `unit/`, `universal/`, and `flowsheet/`.
2. Split `dte/data/` into `datasets/`, `generators/`, and `ingestion/`.
3. Split `dte/training/` into `unit/`, `universal/`, `flowsheet/`, and `shared/`.
4. Leave `simulators/`, `physics/`, `laws/`, `control/`, `customer/`,
   `calibration/`, `evaluation/`, `api/`, `demo/`, and `autoresearch/` intact.
5. Revisit route splitting in `dte/api/service.py` only if it becomes a real
   maintenance problem.

This preserves architectural signal and avoids a large rename-only churn cycle.

---

## Summary

The proposed folder structure is a useful design sketch, but the repo-aligned
target is simpler:

- keep `dte/` as the package root
- keep `simulators/` and `physics/` explicit
- split `models/`, `data/`, and `training/` internally
- do not create premature packages for flowsheet calibration, distributed models,
  or estimation

That path aligns with the current architecture, the current support tiers, and
the actual code already in the repository.
