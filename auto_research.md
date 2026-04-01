# Digital Twin Engine — Autoresearch Context

This file is read by `scripts/agent.py` and injected into the LLM prompt as a compact
repo briefing. It is not the source of truth — the main sources are `README.md`,
`WORKFLOW.md`, `AGENTS.md`, `program.md`, `configs/*.yaml`, and the code itself.

---

## Repo Purpose

Digital Twin Engine is a JAX/Equinox project for learning fast, physics-aware surrogate
models of industrial process systems. The architecture is fully decoupled from any
specific process via the `SystemSpec` / `ProcessSimulator` abstraction.

The product goal is not just forecasting. The learned model must be useful for:
- accurate trajectory prediction
- physically plausible rollouts
- uncertainty-aware simulation
- downstream control, especially MPC

---

## Supported Systems

| System | States | Controls | Disturbances | Physics |
|---|---|---|---|---|
| `cstr` | 4 (Ca, Cb, T, Tc) | 2 (F_in, Tc_in) | 2 (Ca_in, T_in) | Mass + Energy balance |
| `heat_exchanger` | 2 (T_hot, T_cold) | 2 (F_hot, F_cold) | 2 (T_hot_in, T_cold_in) | Energy balance |

New systems require only: a simulator class, an optional physics loss class, a YAML
config, and registration in the simulator and physics registries. No core engine
changes are needed.

---

## Current Architecture (Implemented, Not Aspirational)

### Model pipeline

`physical state + controls → Encoder → latent z → LatentSDE → latent trajectory → Decoder → predicted states`

- **Encoder** (`dte/models/encoder.py`): encodes `state + params + control` → `z_mean`, `z_logvar`; normalisation loaded from `SystemSpec`
- **Decoder** (`dte/models/decoder.py`): decodes `latent + params + control`; output constraints are generic (`softplus`, `sigmoid_range`) driven by `SystemSpec.decoder_constraints`
- **LatentSDE** (`dte/models/latent_sde.py`): separate drift and diffusion MLPs conditioned on `z + control + disturbance + params`; normalisation loaded from `SystemSpec`
- **DigitalTwin** (`dte/models/digital_twin.py`): composes all components; `from_config(config, key, system_spec)` and `load(path, config, system_spec)` both require a resolved `SystemSpec`

All dimensions and normalisation constants come from `SystemSpec` — there are no hardcoded
CSTR values in the model code.

### Training (all features implemented and active by default or via config)

| Feature | Config key | Status |
|---|---|---|
| Deterministic rollout | default (SDE disabled) | Always available |
| Stochastic SDE training | `sde_training.enabled: true` | Implemented; disabled by default |
| Curriculum learning | `curriculum.enabled: true` | Implemented; off by default |
| Teacher-forcing annealing | `teacher_forcing.initial_ratio` | Implemented; off by default |
| KL annealing | `kl_annealing.*` | Always active |

The training loop in `dte/training/trainer.py` uses `model.latent_sde.mean_trajectory(...)`
when `sde_training.enabled` is false (the default). When enabled, it switches to the full
stochastic `model.latent_sde(...)` call.

---

## Current Training Defaults (`configs/training_default.yaml`)

Note: `scripts/train.py` applies a few bootstrap overrides on top of the YAML
defaults during standard training runs, most notably:
- `model.initial_diffusion_scale = 1e-4`
- `optimizer.peak_lr = 5e-4`
- `optimizer.gradient_clip = 0.5`
- `loss_weights.kl = 0.0`

```yaml
model:
  latent_dim: 32
  hidden_dim: 128
  n_layers: 3
  drift_layers: 3
  diffusion_layers: 2
  diffusion_hidden_dim: 64
  initial_diffusion_scale: 0.1

sde:
  dt_ratio: 0.25

training:
  batch_size: 128
  seq_len: 25
  stride: 5

optimizer:
  peak_lr: 5.0e-4
  warmup_steps: 200
  gradient_clip: 0.5

loss_weights:
  reconstruction: 1.0
  kl: 0.0001
  one_step: 0.0
  trajectory: 10.0
  mass_balance: 0.001
  species_mass_balance: 0.001
  energy_balance: 0.001
```

Physics-loss lookup is registry-driven via `dte/physics/registry.py`, and model
construction/loading is `SystemSpec`-driven throughout.

---

## Data

Training data lives in HDF5. The schema is:
```
states          (N, T, state_dim)
controls        (N, T, control_dim)
disturbances    (N, T, disturbance_dim)
params          (N, param_dim)
time            (N, T)              ← key is "time", not "t"
normalization/  state_mean, state_std, control_mean, control_std, ...
```

Batches from `TrajectoryDataset` are in **physical units**.
The trainer normalises internally before computing reconstruction / trajectory losses.
Physics losses are computed in physical units.

`sample_batch(key, batch_size, seq_len=None)` supports an optional `seq_len` argument
for curriculum training (truncates to shorter subsequences when set).

---

## What Is Actually Optimised

Primary metric for autoresearch: `best_val_loss` — lower is better.

Bounded-run interpretation:
- `timed_out: true` with finite `best_val_loss` = valid result
- `best_val_loss: null` = real failure signal
- early stable progress > ambitious edits that collapse numerically

Advisory context available to the agent:
- recent kept runs may also include lightweight deterministic eval summaries
- `rmse_per_state` / `nrmse_per_state` help expose state-specific regressions
- these eval metrics are context only; they do not replace `best_val_loss` as the promotion rule

---

## Files the Agent Should Rely On

- `README.md` — architecture, supported systems, API reference
- `WORKFLOW.md` — operational flow for all phases
- `AGENTS.md` — coding conventions, safe/unsafe files, common mistakes
- `program.md` — keep/discard rules and experiment boundaries
- `configs/training_default.yaml` — baseline hyperparameters
- `configs/autoresearch_default.yaml` — bounded harness settings
- `scripts/train.py` — training CLI
- `dte/training/trainer.py` — training loop

---

## Experiment Boundaries

The agent may only modify files listed in
`configs/autoresearch_default.yaml → agent.modifiable_files`. Defaults:

- `configs/training_default.yaml`
- `scripts/train.py`
- `dte/models/encoder.py`, `decoder.py`, `latent_sde.py`, `digital_twin.py`
- `dte/training/trainer.py`, `losses.py`

One file, one idea, minimal patch.

Do **not** modify the measurement harness:
- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `program.md`
- this file, unless the human explicitly asks

Architecture invariants:
- Preserve the generic `SystemSpec` / `ProcessSimulator` / `PhysicsLoss` flow.
- In `dte/`, do not introduce system-specific branches or string literals like `cstr` / `heat_exchanger`.
- In `dte/models` and `dte/training`, do not hardcode config-like numeric values such as decoder bounds, normalization constants, default physical states, or fixed dimensions.
- If a numeric constraint or scale is needed, put it in YAML / `SystemSpec` / config-driven plumbing instead.
- Ordinary generic numeric algorithmic changes are fine if they are not baked-in system constraints.

---

## Physics and Losses

Physics consistency is a regulariser, not the sole goal. Priority order:

1. Improve predictive validation performance
2. Keep rollouts physically plausible
3. Avoid numerical instability
4. Avoid hard-to-maintain complexity

Mass and energy residuals are in `dte/physics/cstr.py` and `dte/physics/heat_exchanger.py`.
`LossComputer` delegates to whichever `PhysicsLoss` instance is passed in — it has no
direct dependency on any specific system.

---

## Search Heuristics

Prefer experiments that:
- improve stability early in training
- improve `best_val_loss` within the bounded budget
- make minimal single-file changes
- preserve physically plausible outputs

Be cautious with:
- latent variance changes (can cause numerical explosion)
- solver and timestep changes
- decoder output constraint weakening
- large loss-weight increases (common NaN trigger)

---

## Known Failure Patterns

- Latent variance instability → numerical explosion
- Aggressive timestep / solver changes → training collapse
- Decoder edits that weaken physical constraints → unphysical predictions
- Large physics loss weights → NaN training
- `eqx.field(static=True)` on JAX arrays → silent gradient issues (use plain fields)
- HDF5 key `"t"` instead of `"time"` → `KeyError` at dataset load time
- Python list indexing into JAX arrays → deprecation warning (use `jnp.array(idx_list)`)

---

## Current Search Context

- Default autoresearch config currently targets the CSTR benchmark (`data/cstr/` +
  `configs/cstr_default.yaml`)
- Search is based on bounded runs, not full-convergence comparisons
- Stable partial training > ambitious edits that collapse numerically
