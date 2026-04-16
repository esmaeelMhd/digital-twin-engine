# Phase 3: Flowsheet Graph Modeling

Date: 2026-04-10

This phase adds a first plant-section modeling layer on top of the unit foundation model path. The implementation is intentionally small and explicit: units are graph nodes, streams are directed edges, and a shared graph model updates all units while exchanging stream messages.

## What Landed

Core modules:

- `dte/flowsheet/schema.py`
  - `StreamSpec`
  - `FlowsheetSpec`
  - validation for external source/sink endpoints
  - recycle-loop detection
- `dte/flowsheet/examples.py`
  - exchanger -> reactor -> tank
  - reactor -> separator -> recycle -> reactor
- `dte/data/flowsheet_dataset.py`
  - `FlowsheetGraphMetadata`
  - `FlowsheetTrajectoryDataset`
  - HDF5 save/load with topology metadata and preserved `seq_len` / `stride`
- `dte/flowsheet/synthetic.py`
  - synthetic data generator for the example graphs
  - convenience dataset builder for tests and smoke runs
- `dte/models/flowsheet_model.py`
  - shared unit backbone
  - stream message projection
  - graph-level correction block
  - recycle-aware rollout using per-stream delay metadata
- `dte/evaluation/flowsheet_metrics.py`
  - `stream_consistency_loss`
  - `unit_output_consistency_loss`
  - `plant_balance_proxy_loss`
  - `rollout_stability_penalty`
- `dte/training/flowsheet_trainer.py`
  - compute loss
  - train step
  - validate
  - one-call `fit(...)`

## Dataset Shape

Flowsheet trajectory tensors use padded per-unit and per-stream axes:

- `states`: `(N, T, n_units, max_state_dim)`
- `controls`: `(N, T, n_units, max_control_dim)`
- `disturbances`: `(N, T, n_units, max_disturbance_dim)`
- `params`: `(N, n_units, max_param_dim)`
- `stream_values`: `(N, T, n_streams, max_stream_vars)`
- `global_controls`: `(N, T, n_global_controls)`
- `global_disturbances`: `(N, T, n_global_disturbances)`
- `time`: `(N, T)`

`FlowsheetGraphMetadata` carries the static graph tables needed by the model:

- per-unit masks, normalization tables, descriptors, family ids
- per-stream source/target indices
- variable index maps
- stream kinds
- stream delays

## Example Usage

```python
import jax

from dte.flowsheet import (
    build_reactor_separator_recycle_flowsheet,
    build_synthetic_flowsheet_dataset,
)
from dte.models.flowsheet.flowsheet_model import FlowsheetModel
from dte.training.flowsheet.trainer import FlowsheetTrainer

flowsheet = build_reactor_separator_recycle_flowsheet()
dataset = build_synthetic_flowsheet_dataset(
    flowsheet,
    n_trajectories=8,
    n_steps=20,
    seq_len=10,
    stride=5,
    seed=0,
)
train_dataset, val_dataset = dataset.split(0.25)

config = {
    "model": {
        "hidden_dim": 64,
        "message_dim": 16,
        "family_embedding_dim": 8,
        "n_layers": 2,
        "graph_layers": 2,
        "message_passing_steps": 2,
    },
    "optimizer": {
        "peak_lr": 1e-3,
        "end_lr": 1e-4,
        "warmup_steps": 1,
        "total_steps": 32,
        "gradient_clip": 1.0,
    },
    "training": {
        "batch_size": 4,
        "max_batches_per_epoch": 4,
    },
    "checkpointing": {
        "max_val_batches": 2,
    },
    "loss_weights": {
        "trajectory": 1.0,
        "stream_consistency": 1.0,
        "unit_consistency": 0.25,
        "plant_balance": 0.1,
        "rollout_stability": 0.01,
    },
}

model = FlowsheetModel.from_config(config, train_dataset.metadata, jax.random.PRNGKey(0))
trainer = FlowsheetTrainer(model, config, train_dataset, val_dataset)
summary = trainer.fit(jax.random.PRNGKey(1), n_epochs=2)
print(summary)
```

## Current Scope

This is the first usable slice, not the final plant stack.

What it does:

- models small unit graphs
- carries typed unit metadata into the graph path
- rolls out recycle flows with delay-aware stream reuse
- supports train/validate loops and topology-aware proxy losses

What it does not do yet:

- rigorous plant-wide thermodynamics or chemistry laws
- generic flowsheet data generation from real unit simulators
- a dedicated Phase 3 CLI or smoke script
- large-plant sparse graph operators

## Verification

Targeted verification for this phase currently includes:

- `tests/test_flowsheet_schema.py`
- `tests/test_flowsheet_dataset.py`
- `tests/test_flowsheet_model.py`
- `tests/test_flowsheet_trainer.py`

These cover:

- schema validation
- HDF5 roundtrip
- rollout on both example flowsheets
- recycle delay behavior
- one-epoch train/validate execution

## Smoke Runner

Reusable smoke coverage for this phase now lives in:

- `scripts/smoke_phase3.py`

Default usage:

```bash
source .venv/bin/activate
python scripts/smoke_phase3.py
```

Useful variants:

```bash
python scripts/smoke_phase3.py --dry_run
python scripts/smoke_phase3.py --workspace_dir outputs/phase3_smoke/manual_run
python scripts/smoke_phase3.py --flowsheets reactor_separator_recycle
python scripts/smoke_phase3.py --n_trajectories 12 --n_steps 24
python scripts/smoke_phase3.py --skip_data_generation
```
