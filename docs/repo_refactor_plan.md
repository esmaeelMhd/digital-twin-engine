# Repo Refactor Plan

Date: 2026-04-16

This document turns the repo structure target into a phased execution plan with
an exact import-move sequence.

The intent is to improve package clarity without a big-bang rename or a broken
intermediate state.

---

## Refactor Goals

- keep `dte/` as the package root
- preserve current behavior while package paths change
- introduce canonical subpackages only where the repo already has real mass
- avoid creating empty or aspirational areas like `estimation/` or
  `models/distributed/`
- preserve script entrypoints and public imports during the migration

---

## Dependency Facts That Drive The Order

The current import graph makes the move order clear:

1. `dte/data/*` is imported by models, training, calibration, customer, API,
   demo, scripts, and tests.
2. `dte/models/*` is imported by training, control, customer, API, demo,
   scripts, and tests.
3. `dte/training/*` is imported by scripts, calibration, customer, evaluation,
   and tests.

That means the safe migration order is:

`data -> models -> training`

If this order is reversed, the refactor either becomes a big-bang rename or
requires unnecessary circular shim logic.

---

## Migration Rules

Use these rules in every phase:

1. Create the new target module first.
2. Move the implementation into the new target module.
3. Leave the old module in place as a compatibility shim that re-exports from
   the new location.
4. Update internal imports in the repo to the new canonical path.
5. Update scripts.
6. Update tests.
7. Only remove shims after at least one stable cycle with all internal imports
   cleaned up.

Recommended shim pattern:

```python
"""Backward-compatible import shim for the new package layout."""

from dte.data.datasets.unit_dataset import *  # noqa: F401,F403
```

Avoid deprecation warnings during the transition. They add noise to tests and
CI for little value while the repo is mid-migration.

---

## Phase 0. Baseline And Package Scaffolding

### Goal

Create the target subpackage skeleton without changing behavior.

### Add

- `dte/data/datasets/__init__.py`
- `dte/data/generation/__init__.py`
- `dte/data/ingestion/__init__.py`
- `dte/models/unit/__init__.py`
- `dte/models/universal/__init__.py`
- `dte/models/flowsheet/__init__.py`
- `dte/training/unit/__init__.py`
- `dte/training/universal/__init__.py`
- `dte/training/flowsheet/__init__.py`
- `dte/training/shared/__init__.py`

### Validation

Run before any file moves:

```bash
source .venv/bin/activate
pytest tests/test_dataset.py tests/test_model.py tests/test_trainer.py -q
pytest tests/test_universal_digital_twin.py tests/test_universal_trainer_phase1.py -q
pytest tests/test_flowsheet_dataset.py tests/test_flowsheet_model.py tests/test_flowsheet_trainer.py -q
pytest tests/test_customer_workflow.py tests/test_api_onboarding.py -q
```

This gives a stable baseline on the main dependency chain.

---

## Phase 1. Split `dte/data/`

### Goal

Move dataset, generation, and ingestion modules to their canonical subpackages
while leaving import compatibility in place.

### Exact Move Order

1. Move generator and ingestion helpers first.
2. Move dataset modules second.
3. Update model imports.
4. Update training imports.
5. Update calibration, customer, API, demo imports.
6. Update scripts.
7. Update tests.

### File Moves

| Old path | New path |
| --- | --- |
| `dte/data/generation_generic.py` | `dte/data/generation/generic.py` |
| `dte/data/generation.py` | `dte/data/generation/cstr_legacy.py` |
| `dte/data/real_data.py` | `dte/data/ingestion/real_data.py` |
| `dte/data/dataset.py` | `dte/data/datasets/unit_dataset.py` |
| `dte/data/multi_system_dataset.py` | `dte/data/datasets/universal_unit_dataset.py` |
| `dte/data/flowsheet_dataset.py` | `dte/data/datasets/flowsheet_dataset.py` |

### Compatibility Shims To Leave Behind

- `dte/data/generation_generic.py`
- `dte/data/generation.py`
- `dte/data/real_data.py`
- `dte/data/dataset.py`
- `dte/data/multi_system_dataset.py`
- `dte/data/flowsheet_dataset.py`

### Canonical Import Rewrites

| Old import | New import |
| --- | --- |
| `from dte.data.dataset import TrajectoryDataset` | `from dte.data.datasets.unit_dataset import TrajectoryDataset` |
| `from dte.data.multi_system_dataset import MultiSystemTrajectoryDataset` | `from dte.data.datasets.universal_unit_dataset import MultiSystemTrajectoryDataset` |
| `from dte.data.multi_system_dataset import UniversalSystemMetadata` | `from dte.data.datasets.universal_unit_dataset import UniversalSystemMetadata` |
| `from dte.data.flowsheet_dataset import FlowsheetTrajectoryDataset` | `from dte.data.datasets.flowsheet_dataset import FlowsheetTrajectoryDataset` |
| `from dte.data.generation_generic import GenericDataGenerator` | `from dte.data.generation.generic import GenericDataGenerator` |
| `from dte.data.real_data import RealDataIngestion` | `from dte.data.ingestion.real_data import RealDataIngestion` |

### Internal Files To Update In This Phase

Models:

- `dte/models/universal_digital_twin.py`
- `dte/models/flowsheet_model.py`

Training and library code:

- `dte/training/trainer.py`
- `dte/training/universal_trainer.py`
- `dte/training/flowsheet_trainer.py`
- `dte/calibration/unit_calibration.py`
- `dte/customer/adaptation.py`
- `dte/flowsheet/synthetic.py`
- `dte/demo/engine.py`
- `dte/api/onboarding.py`

Scripts:

- `scripts/train.py`
- `scripts/train_universal.py`
- `scripts/evaluate.py`
- `scripts/evaluate_universal.py`
- `scripts/generate_data.py`
- `scripts/generate_corpus.py`
- `scripts/benchmark_generation.py`
- `scripts/ingest_real_data.py`
- `scripts/calibrate_unit.py`
- `scripts/pilot_real_data_adaptation.py`
- `scripts/smoke_phase3.py`

Tests:

- `tests/test_dataset.py`
- `tests/test_generation.py`
- `tests/test_real_data.py`
- `tests/test_universal_digital_twin.py`
- `tests/test_universal_trainer_phase1.py`
- `tests/test_evaluation_utils.py`
- `tests/test_customer_workflow.py`
- `tests/test_flowsheet_dataset.py`
- `tests/test_flowsheet_model.py`
- `tests/test_flowsheet_trainer.py`
- `tests/test_api_demo_routes.py`
- `tests/test_unit_calibration.py`

### Validation

```bash
source .venv/bin/activate
pytest tests/test_dataset.py tests/test_generation.py tests/test_real_data.py -q
pytest tests/test_universal_digital_twin.py tests/test_customer_workflow.py -q
pytest tests/test_flowsheet_dataset.py tests/test_flowsheet_model.py tests/test_flowsheet_trainer.py -q
```

---

## Phase 2. Split `dte/models/`

### Goal

Separate unit, universal, and flowsheet model paths without renaming the core
classes.

### Exact Move Order

1. Move unit-model building blocks.
2. Move `DigitalTwin`.
3. Move `UniversalDigitalTwin`.
4. Move `FlowsheetModel`.
5. Update training, control, customer, API, demo imports.
6. Update scripts.
7. Update tests.

### File Moves

| Old path | New path |
| --- | --- |
| `dte/models/encoder.py` | `dte/models/unit/encoder.py` |
| `dte/models/grouped_encoder.py` | `dte/models/unit/grouped_encoder.py` |
| `dte/models/decoder.py` | `dte/models/unit/decoder.py` |
| `dte/models/latent_sde.py` | `dte/models/unit/latent_sde.py` |
| `dte/models/digital_twin.py` | `dte/models/unit/digital_twin.py` |
| `dte/models/universal_digital_twin.py` | `dte/models/universal/digital_twin.py` |
| `dte/models/flowsheet_model.py` | `dte/models/flowsheet/flowsheet_model.py` |

### Compatibility Shims To Leave Behind

- `dte/models/encoder.py`
- `dte/models/grouped_encoder.py`
- `dte/models/decoder.py`
- `dte/models/latent_sde.py`
- `dte/models/digital_twin.py`
- `dte/models/universal_digital_twin.py`
- `dte/models/flowsheet_model.py`

### Canonical Import Rewrites

| Old import | New import |
| --- | --- |
| `from dte.models.encoder import Encoder` | `from dte.models.unit.encoder import Encoder` |
| `from dte.models.grouped_encoder import GroupedStateEncoder` | `from dte.models.unit.grouped_encoder import GroupedStateEncoder` |
| `from dte.models.decoder import Decoder` | `from dte.models.unit.decoder import Decoder` |
| `from dte.models.latent_sde import LatentSDE` | `from dte.models.unit.latent_sde import LatentSDE` |
| `from dte.models.digital_twin import DigitalTwin` | `from dte.models.unit.digital_twin import DigitalTwin` |
| `from dte.models.universal_digital_twin import UniversalDigitalTwin` | `from dte.models.universal.digital_twin import UniversalDigitalTwin` |
| `from dte.models.flowsheet_model import FlowsheetModel` | `from dte.models.flowsheet.flowsheet_model import FlowsheetModel` |

### Internal Files To Update In This Phase

Library code:

- `dte/training/trainer.py`
- `dte/training/universal_trainer.py`
- `dte/training/flowsheet_trainer.py`
- `dte/training/online.py`
- `dte/training/transfer.py`
- `dte/control/mpc.py`
- `dte/control/mpc_interface.py`
- `dte/control/state_correction.py`
- `dte/calibration/unit_calibration.py`
- `dte/customer/adaptation.py`
- `dte/customer/reporting.py`
- `dte/demo/engine.py`
- `dte/evaluation/universal.py`
- `dte/api/service.py`
- `dte/api/onboarding.py`

Scripts:

- `scripts/train.py`
- `scripts/train_universal.py`
- `scripts/evaluate.py`
- `scripts/evaluate_universal.py`
- `scripts/run_mpc.py`
- `scripts/calibrate_unit.py`
- `scripts/smoke_phase3.py`
- `scripts/smoke_phase7.py`

Tests:

- `tests/test_model.py`
- `tests/test_latent_sde.py`
- `tests/test_digital_twin.py`
- `tests/test_grouped_encoder.py`
- `tests/test_universal_digital_twin.py`
- `tests/test_universal_trainer_phase1.py`
- `tests/test_flowsheet_model.py`
- `tests/test_flowsheet_trainer.py`
- `tests/test_phase7_control.py`
- `tests/test_trainer.py`

### Validation

```bash
source .venv/bin/activate
pytest tests/test_model.py tests/test_latent_sde.py tests/test_digital_twin.py tests/test_grouped_encoder.py -q
pytest tests/test_universal_digital_twin.py tests/test_universal_trainer_phase1.py -q
pytest tests/test_flowsheet_model.py tests/test_flowsheet_trainer.py tests/test_phase7_control.py -q
```

---

## Phase 3. Split `dte/training/`

### Goal

Separate unit, universal, flowsheet, and shared training code while keeping CLI
entrypoints stable in `scripts/`.

### Exact Move Order

1. Move shared support modules.
2. Move unit trainer.
3. Move universal trainer.
4. Move flowsheet trainer.
5. Update scripts.
6. Update calibration, customer, and evaluation imports.
7. Update tests.

### File Moves

| Old path | New path |
| --- | --- |
| `dte/training/losses.py` | `dte/training/shared/losses.py` |
| `dte/training/config_resolution.py` | `dte/training/shared/config_resolution.py` |
| `dte/training/transfer.py` | `dte/training/shared/transfer.py` |
| `dte/training/trainer.py` | `dte/training/unit/trainer.py` |
| `dte/training/universal_trainer.py` | `dte/training/universal/trainer.py` |
| `dte/training/flowsheet_trainer.py` | `dte/training/flowsheet/trainer.py` |

### Keep In Place

- `dte/training/online.py`

This module is already a cross-cutting runtime utility and does not benefit from
further nesting yet.

### Compatibility Shims To Leave Behind

- `dte/training/losses.py`
- `dte/training/config_resolution.py`
- `dte/training/transfer.py`
- `dte/training/trainer.py`
- `dte/training/universal_trainer.py`
- `dte/training/flowsheet_trainer.py`

### Canonical Import Rewrites

| Old import | New import |
| --- | --- |
| `from dte.training.losses import LossComputer` | `from dte.training.shared.losses import LossComputer` |
| `from dte.training.config_resolution import resolve_single_system_training_config` | `from dte.training.shared.config_resolution import resolve_single_system_training_config` |
| `from dte.training.transfer import apply_finetune_mask` | `from dte.training.shared.transfer import apply_finetune_mask` |
| `from dte.training.trainer import Trainer` | `from dte.training.unit.trainer import Trainer` |
| `from dte.training.universal_trainer import UniversalTrainer` | `from dte.training.universal.trainer import UniversalTrainer` |
| `from dte.training.flowsheet_trainer import FlowsheetTrainer` | `from dte.training.flowsheet.trainer import FlowsheetTrainer` |

### Internal Files To Update In This Phase

Library code:

- `dte/calibration/unit_calibration.py`
- `dte/customer/reporting.py`
- `dte/evaluation/universal.py`
- `dte/training/online.py`

Scripts:

- `scripts/train.py`
- `scripts/train_universal.py`
- `scripts/evaluate_universal.py`
- `scripts/smoke_phase3.py`
- `scripts/smoke_phase4.py`

Tests:

- `tests/test_trainer.py`
- `tests/test_universal_trainer_phase1.py`
- `tests/test_config_resolution.py`
- `tests/test_flowsheet_trainer.py`
- `tests/test_physics.py`

### Validation

```bash
source .venv/bin/activate
pytest tests/test_trainer.py tests/test_config_resolution.py tests/test_physics.py -q
pytest tests/test_universal_trainer_phase1.py tests/test_flowsheet_trainer.py -q
```

---

## Phase 4. Normalize Imports Across The Repo

### Goal

After shims exist and the code is stable, make the new paths canonical
everywhere in the repo.

### Update Areas

- all `dte/` library imports
- all `scripts/` imports
- all `tests/` imports
- doc snippets that reference old import paths

### Completion Condition

`rg` should find no internal imports of the old paths except inside compatibility
shim files.

Suggested checks:

```bash
rg -n "from dte\\.data\\.(dataset|multi_system_dataset|flowsheet_dataset|generation_generic|real_data)" dte scripts tests
rg -n "from dte\\.models\\.(encoder|grouped_encoder|decoder|latent_sde|digital_twin|universal_digital_twin|flowsheet_model)" dte scripts tests
rg -n "from dte\\.training\\.(losses|config_resolution|transfer|trainer|universal_trainer|flowsheet_trainer)" dte scripts tests
```

Expected result:

- no matches outside shim modules

---

## Phase 5. Optional Late Splits

Only do these if there is an actual maintenance reason:

- split `dte/api/service.py` into route modules
- rename `dte/customer/reporting.py` to `validation_report.py`
- add `estimation/` only after a real estimator/filter stack exists
- add flowsheet calibration modules only after flowsheet customer adaptation is
  first-class
- add distributed model packages only after distributed-unit modeling is a real
  supported path

These are not part of the core cleanup.

---

## Phase 6. Remove Compatibility Shims

### Preconditions

- all internal imports are on canonical paths
- docs are updated
- scripts are updated
- tests are updated
- at least one stable cycle has passed with the shim layout in place

### Removal Order

1. remove data shims
2. remove model shims
3. remove training shims

This order matches the original dependency layering.

### Final Validation

```bash
source .venv/bin/activate
pytest tests/ -q
```

If the full suite is expensive, at minimum run:

```bash
pytest tests/test_api_onboarding.py tests/test_customer_workflow.py -q
pytest tests/test_trainer.py tests/test_universal_trainer_phase1.py -q
pytest tests/test_flowsheet_trainer.py tests/test_phase7_control.py -q
```

---

## Exact Implementation Sequence

If this work is done as actual code changes, the best execution sequence is:

1. Add package scaffolding.
2. Move data modules and add data shims.
3. Patch imports to canonical data paths.
4. Move model modules and add model shims.
5. Patch imports to canonical model paths.
6. Move training modules and add training shims.
7. Patch imports to canonical training paths.
8. Run targeted tests after each phase.
9. Run the full test suite after Phase 4 or Phase 6.
10. Remove shims only after a stable cycle.

That sequence minimizes churn, keeps diffs reviewable, and matches the actual
dependency graph of the current repo.

---

## Summary

The refactor should not be a package-name redesign exercise.

The practical plan is:

- split `data/` first
- split `models/` second
- split `training/` third
- keep `simulators/`, `physics/`, `laws/`, `control/`, `customer/`,
  `calibration/`, `evaluation/`, `api/`, `demo/`, and `autoresearch/` stable
- use compatibility shims until the whole repo has moved

That is the shortest path from the current repo to the target structure without
breaking the codebase in the middle.
