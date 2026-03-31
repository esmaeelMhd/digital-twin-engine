# AGENTS.md — AI Assistant Guide for Digital Twin Engine

This file is read automatically by AI coding agents at the start of every session.
It describes repo conventions, what is safe to change, what to avoid, and key facts
about the current architecture so you do not have to rediscover them from scratch.

---

## What This Repo Is

A general-purpose physics-informed latent neural SDE platform for industrial process
digital twins. It is no longer CSTR-specific. The architecture is fully decoupled from
any particular process system through a `SystemSpec` / `ProcessSimulator` abstraction.

**Primary stack:** JAX, Equinox, Diffrax, Optax  
**Python:** 3.10 – 3.14  
**Virtual environment:** `.venv/` at project root — always activate before running anything:

```bash
source .venv/bin/activate
```

---

## Key Architecture Facts (Current State)

### System abstraction

| File | Role |
|---|---|
| `dte/simulators/base.py` | `SystemSpec` dataclass + `ProcessSimulator` ABC |
| `dte/simulators/registry.py` | `get_system_spec(config)`, `get_simulator(name, config)`, `list_systems()` |
| `dte/simulators/cstr.py` | CSTR concrete simulator |
| `dte/simulators/heat_exchanger.py` | Counter-current heat exchanger simulator |

`SystemSpec` carries: `state_dim`, `control_dim`, `disturbance_dim`, `param_dim`,
`state_names`, `control_names`, `disturbance_names`, `decoder_constraints`,
`normalization` (centers + scales for all channels), `default_initial_state`,
`default_nominal_disturbance`, `control_ranges`, `disturbance_ranges`.

All model components (`Encoder`, `Decoder`, `LatentSDE`, `DigitalTwin`) are
initialised from `SystemSpec` — there are no hardcoded CSTR dimensions or
normalisation constants anywhere in the model code.

### Physics losses

| File | Role |
|---|---|
| `dte/physics/base.py` | `PhysicsLoss` ABC + `NullPhysicsLoss` |
| `dte/physics/registry.py` | Registry-driven physics loss / diagnostic lookup |
| `dte/physics/cstr.py` | `CSTRPhysicsLoss` (mass + energy residuals) |
| `dte/physics/heat_exchanger.py` | `HeatExchangerPhysicsLoss` (energy residual) |
| `dte/physics/conservation.py` | Re-exports from `cstr.py` for backward compatibility |

`LossComputer` (`dte/training/losses.py`) accepts any `PhysicsLoss` instance and
queries `residual_names()` dynamically — it has no CSTR coupling.
Training and evaluation resolve system-specific physics through
`dte/physics/registry.py`, not through hardcoded script branches.

### Training features (all implemented)

- **Stochastic SDE training** — full diffusion path + KL regularisation, gated by
  `sde_training.enabled` in config. Warmup delay controlled by `sde_training.warmup_steps`.
- **Curriculum learning** — `seq_len` ramps from `curriculum.initial_seq_len` to
  `curriculum.final_seq_len` over `curriculum.warmup_epochs`.
- **Teacher-forcing annealing** — one-step loss weight anneals from
  `teacher_forcing.initial_ratio` to `teacher_forcing.final_ratio` over
  `teacher_forcing.anneal_epochs`.

### Data

| File | Role |
|---|---|
| `dte/data/dataset.py` | `TrajectoryDataset` — HDF5 loader, `sample_batch(key, bs, seq_len=None)` |
| `dte/data/generation.py` | CSTR-specific data generator (fast, original) |
| `dte/data/generation_generic.py` | `GenericDataGenerator` — works with any `ProcessSimulator` |
| `dte/data/real_data.py` | `RealDataIngestion` — CSV/Parquet ingestion pipeline |

**HDF5 schema** (all generators must match this):
```
states         (N, T, state_dim)
controls       (N, T, control_dim)
disturbances   (N, T, disturbance_dim)
params         (N, param_dim)
time           (N, T)              ← key is "time", not "t"
normalization/
  state_mean, state_std
  control_mean, control_std
  disturbance_mean, disturbance_std
```

Batches coming from `TrajectoryDataset` are in **physical units**.
The trainer normalises internally before computing reconstruction / trajectory losses.
Physics losses are computed in physical units.

### Production modules

| File | Role |
|---|---|
| `dte/training/online.py` | `OnlineAdapter` — ring buffer, CUSUM drift detection, sliding-window fine-tuning |
| `dte/training/transfer.py` | `FewShotAdapter`, `zero_shot_eval`, `_build_filter_spec` |
| `dte/api/models.py` | Pydantic v2 request/response models |
| `dte/api/service.py` | FastAPI app — `/health`, `/predict`, `/ensemble`, `/steady_state` |
| `Dockerfile` | Multi-stage: `builder` → `api` (lean) + `train` (full) |
| `docker-compose.yml` | `api` + `dashboard` services; `--profile tools` for data-gen/training |

---

## Config Files

| Config | Used for |
|---|---|
| `configs/cstr_default.yaml` | CSTR `SystemSpec`, simulator params, decoder constraints, normalisation |
| `configs/heat_exchanger_default.yaml` | Heat exchanger `SystemSpec` and simulator params |
| `configs/heat_exchanger_training.yaml` | HX-specific training hypers (smaller model, energy loss weights) |
| `configs/training_default.yaml` | Model architecture + training loop + SDE/curriculum/teacher-forcing |
| `configs/mpc_default.yaml` | CEM-MPC horizon, candidates, elite fraction |
| `configs/autoresearch_default.yaml` | Bounded experiment harness settings + agent modifiable-file list |

---

## Scripts

| Script | What it does |
|---|---|
| `scripts/generate_data.py` | Data generation for any registered system (routes by system name) |
| `scripts/ingest_real_data.py` | Real plant CSV/Parquet → HDF5 ingestion CLI |
| `scripts/train.py` | Training; `--finetune <ckpt>` + `--finetune_part {decoder,encoder,all}` for transfer |
| `scripts/evaluate.py` | Evaluation and plot generation |
| `scripts/run_mpc.py` | CEM-MPC simulation; `--compare_pid` supported only for CSTR |
| `scripts/autoresearch.py` | Bounded single-experiment harness |
| `scripts/agent.py` | Autonomous LLM-driven research loop |
| `scripts/verify_install.py` | Install sanity check |

---

## Adding a New Process System

The core model/training engine does not change. New systems are added at the
system / physics registry boundary.

1. `dte/simulators/my_system.py` — subclass `ProcessSimulator`
2. `dte/physics/my_system.py` — subclass `PhysicsLoss` (or use `NullPhysicsLoss`)
3. `configs/my_system_default.yaml` — define `SystemSpec` fields + simulator params
4. `dte/simulators/registry.py` — register spec/simulator builder entries
5. `dte/physics/registry.py` — register physics-loss / diagnostic builders if applicable

---

## What Is Safe to Change

- `dte/models/*.py` — model architecture (always initialised from `SystemSpec`)
- `dte/training/losses.py`, `trainer.py` — loss computation and training loop
- `configs/training_default.yaml` — hyperparameters
- `scripts/train.py` — training CLI (but keep `--system_config` and `--finetune` flags)
- Any new `dte/simulators/`, `dte/physics/` files for new systems

## What to Avoid Changing

| File | Reason |
|---|---|
| `scripts/autoresearch.py` | Measurement harness — changes break experiment comparability |
| `dte/autoresearch/*` | Same |
| `scripts/agent.py` | Autonomous loop driver |
| `program.md` | Autoresearch operating rules |
| `auto_research.md` | LLM prompt context for autoresearch agent |
| `dte/simulators/base.py` | Core abstraction; changes break all downstream code |
| `dte/data/dataset.py` | HDF5 schema contract; key names must stay stable |

---

## Common Mistakes to Avoid

**JAX array fields in Equinox modules**
Do not use `eqx.field(static=True)` for JAX arrays — that triggers a warning and
may silently make gradients not flow. Use plain fields; exclude from optimiser via
`eqx.filter(model, eqx.is_array)`.

**HDF5 time key**
The time dataset key is `"time"`, not `"t"`. Every generator that writes HDF5 must
use `f.create_dataset("time", ...)` or `TrajectoryDataset` will raise a `KeyError`.

**`jnp.array` indexing**
When indexing a JAX array with a Python list, wrap it: `arr[jnp.array(idx_list)]`.
Plain list indexing is deprecated in recent JAX versions.

**`LossComputer` constructor**
Signature is `LossComputer(config, normalization_stats, physics_loss, state_names)`.
`config` is the **full** config dict (must have `loss_weights` at the top level),
not just `config["training"]`.

**`DigitalTwin.from_config` / `DigitalTwin.load`**
Both require a `SystemSpec`. Do not construct or load the model from training config
alone; resolve the spec first with `get_system_spec(system_config)`.

**`eqx.partition` / `eqx.combine` pattern for selective fine-tuning**
Use `trainable, frozen = eqx.partition(model, filter_spec)` and
`eqx.combine(new_trainable, frozen)`. Do not use the removed `eqx.is_not_array`.

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Quick pipeline smoke test (no GPU needed, ~2 minutes):

```bash
python scripts/generate_data.py \
  --config configs/cstr_default.yaml \
  --n_trajectories 100 --output_dir data/test/

python scripts/train.py \
  --data_dir data/test/ --n_epochs 2 --batch_size 8 \
  --output_dir outputs/smoke_test/

python scripts/evaluate.py \
  --model_path outputs/smoke_test/final_model.eqx \
  --config outputs/smoke_test/config.yaml \
  --data_dir data/test/ --output_dir outputs/smoke_test/eval/
```

---

## Autoresearch Context

The autoresearch loop (`scripts/agent.py`) injects `auto_research.md` into every
LLM prompt as a compact repo briefing. `program.md` defines the keep/discard rules.

The primary optimisation target is `best_val_loss` (lower is better) measured within
a bounded wall-clock budget. `timed_out: true` with a finite `best_val_loss` is a
valid result. `best_val_loss: null` is the real failure signal.

Prefer experiments that:
- make a single minimal change to one file
- improve stability early in training
- avoid NaN losses (most commonly caused by large LR, heavy physics weights, or
  aggressive latent variance changes)

---

## Environment Variables (Production / API)

| Variable | Default | Description |
|---|---|---|
| `DTE_SYSTEM_CONFIG` | `configs/cstr_default.yaml` | System YAML path (comma-separated for multiple) |
| `DTE_MODEL_PATH` | `outputs/best_model.eqx` | Trained checkpoint path |
| `DTE_TRAINING_CONFIG` | `configs/training_default.yaml` | Training config used to reconstruct model |
| `DTE_API_KEY` | _(unset)_ | API key for FastAPI auth; unset = auth disabled |
| `STREAMLIT_AUTH_PASSWORD` | _(unset)_ | Dashboard password; unset = auth disabled |
| `JAX_PLATFORMS` | _(unset)_ | Set to `cpu` to force CPU (useful in containers) |
