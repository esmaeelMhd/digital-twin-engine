# Digital Twin Engine — Complete Workflow

Step-by-step guide from environment setup to production deployment. All four phases of the generalisation roadmap are complete.

---

## Phase 1: Environment Setup

### 1.1 Clone and Create Virtual Environment

```bash
cd digital-twin-engine
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### 1.2 Verify Installation

```bash
python scripts/verify_install.py
# Expected: "✓ ALL CHECKS PASSED"
```

### 1.3 Run Tests

```bash
pytest tests/ -v
```

**Checkpoint:** all packages installed, tests passing.

---

## Phase 2: Data

The engine supports three data sources: **synthetic simulation**, **generic simulator**, and **real plant files**.

### 2.1 Synthetic Data — CSTR (default)

```bash
python scripts/generate_data.py \
  --config configs/cstr_default.yaml \
  --n_trajectories 10000 \
  --n_steps 1000 \
  --output_dir data/cstr/
```

Output: `data/cstr/train_data.h5` (~400 MB)

### 2.2 Synthetic Data — Heat Exchanger

```bash
python scripts/generate_data.py \
  --config configs/heat_exchanger_default.yaml \
  --n_trajectories 10000 \
  --n_steps 1000 \
  --output_dir data/heat_exchanger/
```

### 2.3 Synthetic Data — Any Registered System

The `--config` flag selects the system. The script routes to the appropriate generator automatically.

```bash
python scripts/generate_data.py \
  --config configs/two_tank_default.yaml \
  --output_dir data/two_tank/

# Or any future registered system
python scripts/generate_data.py \
  --config configs/my_system_default.yaml \
  --output_dir data/my_system/
```

### 2.4 Real Plant Data (CSV / Parquet)

```bash
python scripts/ingest_real_data.py \
  --source data/raw/plant_run_01.csv \
  --output data/cstr_real/train_data.h5 \
  --system_config configs/cstr_default.yaml \
  --state_columns Ca Cb T Tc \
  --control_columns F_in Tc_in \
  --disturbance_columns Ca_in T_in \
  --timestamp_column time \
  --dt 0.1 \
  --trajectory_duration 100.0 \
  --trajectory_stride 10.0 \
  --outlier_sigma 5.0
```

The ingestion pipeline handles:
- Irregular timestamps (float seconds or ISO datetime strings)
- Automatic sorting and deduplication
- Linear interpolation to a uniform grid
- Configurable large-gap handling (`--max_gap_fill`, `--drop_large_gaps`)
- Z-score outlier detection and median replacement
- Sensor noise characterisation
- Output is a standards-compliant HDF5 file directly loadable by the trainer

Save the ingestion summary as JSON for auditing:

```bash
python scripts/ingest_real_data.py ... --save_summary data/cstr_real/ingestion_summary.json
```

**Checkpoint:** HDF5 file created, shape and normalization stats printed.

---

## Phase 3: Training

### 3.1 Train from Scratch

```bash
python scripts/train.py \
  --config configs/training_default.yaml \
  --system_config configs/cstr_default.yaml \
  --data_dir data/cstr/ \
  --output_dir outputs/cstr_v1/ \
  --n_epochs 100 \
  --batch_size 64 \
  --seed 42
```

Training features (all configurable in `configs/training_default.yaml`):

| Feature | Config key | Description |
|---|---|---|
| Stochastic SDE training | `sde_training.enabled` | Activates the diffusion path; KL term regularises diffusion scale |
| Curriculum learning | `curriculum.enabled` | `seq_len` ramps linearly from `initial_seq_len` to `final_seq_len` |
| Teacher-forcing annealing | `teacher_forcing.initial_ratio` | One-step loss weight decays toward free-rollout loss |

Outputs:
- `outputs/cstr_v1/best_model.eqx` — best validation checkpoint
- `outputs/cstr_v1/final_model.eqx` — final epoch
- `outputs/cstr_v1/training_history.json` — loss curves
- `outputs/cstr_v1/training_summary.json` — machine-readable summary

### 3.2 Train a Different System

Change only `--system_config` and `--data_dir`; everything else is system-agnostic:

```bash
python scripts/train.py \
  --config configs/heat_exchanger_training.yaml \
  --system_config configs/heat_exchanger_default.yaml \
  --data_dir data/heat_exchanger/ \
  --output_dir outputs/hx_v1/

python scripts/train.py \
  --config configs/two_tank_training.yaml \
  --system_config configs/two_tank_default.yaml \
  --data_dir data/two_tank/ \
  --output_dir outputs/two_tank_v1/
```

### 3.3 Train a Shared Universal Baseline

Train one checkpoint across all registered systems listed in
`configs/training_universal.yaml`:

```bash
python scripts/train_universal.py \
  --config configs/training_universal.yaml \
  --output_dir outputs/universal_v1/ \
  --n_epochs 20 \
  --batch_size 128 \
  --seed 42
```

Evaluate the shared checkpoint:

```bash
python scripts/evaluate_universal.py \
  --model_path outputs/universal_v1/best_model.eqx \
  --config outputs/universal_v1/config.yaml \
  --output_dir outputs/universal_v1/eval/
```

The universal path uses:
- Mixed-system padded batches from `dte/data/multi_system_dataset.py`
- Typed state groups from each system config (`thermal`, `concentration`, `inventory`, etc.)
- One shared grouped universal model in `dte/models/universal_digital_twin.py`

Outputs:
- `outputs/universal_v1/best_model.eqx` — best mixed validation checkpoint
- `outputs/universal_v1/final_model.eqx` — final epoch
- `outputs/universal_v1/summary.json` — mixed + per-system validation summary
- `outputs/universal_v1/eval/summary.json` — evaluation summary

### 3.4 Few-Shot Transfer Learning (Fine-tune a Pre-trained Model)

Freeze the encoder and latent SDE; update only the decoder on N new-unit trajectories:

```bash
python scripts/train.py \
  --finetune outputs/cstr_v1/best_model.eqx \
  --finetune_part decoder \
  --system_config configs/cstr_default.yaml \
  --data_dir data/cstr_unit2/ \
  --output_dir outputs/cstr_unit2/ \
  --n_epochs 10
```

Options for `--finetune_part`: `decoder` (default), `encoder`, `all`.

### 3.5 Autoresearch Loop

```bash
python scripts/autoresearch.py \
  --config configs/autoresearch_default.yaml \
  --description baseline \
  --data_dir data/test/
```

The harness runs training with a fixed wall-clock budget, logs results to `outputs/autoresearch/results.tsv`, and promotes only improvements to the baseline directory.

```bash
# Autonomous LLM-driven agent (Gemini by default)
python scripts/agent.py --max-runs 50

# Other providers
python scripts/agent.py --claude    # Claude Sonnet
python scripts/agent.py --openai    # OpenAI o3
python scripts/agent.py --grok      # xAI Grok 3
```

**Checkpoint:** `best_val_loss` reported, model checkpoint saved.

---

## Phase 4: Evaluation

```bash
python scripts/evaluate.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --config outputs/cstr_v1/config.yaml \
  --system_config configs/cstr_default.yaml \
  --data_dir data/cstr/ \
  --output_dir outputs/cstr_v1/eval/
```

For a shared universal checkpoint:

```bash
python scripts/evaluate_universal.py \
  --model_path outputs/universal_v1/best_model.eqx \
  --config outputs/universal_v1/config.yaml \
  --output_dir outputs/universal_v1/eval/
```

Generated artefacts:
- Trajectory comparison plots (predicted vs ground truth)
- Per-state prediction error plots
- Physics violation metrics (mass / energy balance residuals)
- Uncertainty calibration statistics (ensemble coverage)

Key metrics to target:

| Metric | Target |
|---|---|
| 1-step normalised MSE | < 0.01 |
| Full-sequence normalised MSE | < 0.1 |
| Mass balance violation (mean) | < 0.01 |
| Energy balance violation (mean) | < 1.0 |
| Ensemble coverage (±2σ) | ~95% |

### Zero-Shot and Few-Shot Transfer Evaluation (Python API)

```python
from dte.training.transfer import FewShotAdapter, zero_shot_eval

# Zero-shot: evaluate pre-trained model on a new unit without any updates
metrics = zero_shot_eval(pretrained_model, new_unit_dataset, n_batches=50)
print(metrics)  # {"mse": ..., "rmse": ..., "norm_mse": ...}

# Few-shot: fine-tune decoder on 5 trajectories, then evaluate
adapter = FewShotAdapter(pretrained_model, system_spec, learning_rate=3e-4)
finetuned = adapter.finetune(new_unit_dataset, n_steps=200, part="decoder")
metrics_fs = zero_shot_eval(finetuned, new_unit_dataset)
```

**Checkpoint:** accuracy targets met, physics constraints satisfied.

---

## Phase 5: Model Predictive Control

```bash
python scripts/run_mpc.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --system_config configs/cstr_default.yaml \
  --setpoint_T 340.0 \
  --setpoint_Ca 0.8 \
  --disturbance_scenario step \
  --n_steps 200 \
  --compare_pid \
  --output_dir outputs/mpc_results/
```

The `--compare_pid` flag is supported only for CSTR. For other systems, the script runs the AI-MPC only.

Try different scenarios:

```bash
# Random disturbance
python scripts/run_mpc.py ... --disturbance_scenario random

# Different setpoint
python scripts/run_mpc.py ... --setpoint_T 360.0 --setpoint_Ca 0.5
```

**Checkpoint:** MPC stable, ISE improvement over baseline reported.

---

## Phase 6: Online Adaptation

Use `OnlineAdapter` when the model is deployed and new plant observations arrive in real time.

```python
from dte.training.online import OnlineAdapter, OnlineAdapterConfig

adapter = OnlineAdapter(
    model,
    system_spec,
    OnlineAdapterConfig(
        window_size=500,       # ring buffer of recent observations
        finetune_every=50,     # run gradient steps every N pushes
        n_finetune_steps=10,   # gradient steps per trigger
        seq_len=20,            # subsequence length sampled from buffer
        drift_threshold=3.0,   # CUSUM alarm threshold
        drift_slack=0.5,       # allowable deviation before CUSUM accumulates
        ewc_lambda=0.0,        # set > 0 to penalise forgetting
    ),
    key=jax.random.PRNGKey(0),
)

# In the plant loop:
for t, measurement in plant_stream:
    result = adapter.push(
        states=measurement.states,
        controls=measurement.controls,
        disturbances=measurement.disturbances,
        t=t,
    )
    if result["drift"]:
        print(f"Drift detected at t={t}  CUSUM={result['cusum']:.2f}")
    model = adapter.model   # always up to date

# Diagnostics
print(adapter.get_diagnostics())
```

The CUSUM detector fires when rolling prediction error exceeds `drift_threshold` standard deviations above the baseline. Each alarm triggers an extra fine-tune pass in addition to the periodic schedule.

---

## Phase 7: Dashboard

```bash
streamlit run app/dashboard.py
```

Access at `http://localhost:8501`.

Optional password protection:

```bash
STREAMLIT_AUTH_PASSWORD=secret streamlit run app/dashboard.py
```

Features:
- Live simulation with Open Loop / PID / AI-MPC modes
- State trajectory plots with uncertainty bands
- Disturbance scenario selection
- Performance metrics (ISE, settling time, overshoot)
- System selection (CSTR, heat exchanger, or any registered system)

---

## Phase 8: REST API

### Local

```bash
# Set environment
export DTE_SYSTEM_CONFIG=configs/cstr_default.yaml
export DTE_MODEL_PATH=outputs/cstr_v1/best_model.eqx
export DTE_TRAINING_CONFIG=configs/training_default.yaml
# Optional auth:
# export DTE_API_KEY=your-secret-key

uvicorn dte.api.service:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# API only
docker build --target api -t dte-api .
docker run -p 8000:8000 \
  -e DTE_MODEL_PATH=outputs/best_model.eqx \
  -e DTE_API_KEY=your-secret-key \
  -v $(pwd)/outputs:/app/outputs \
  dte-api

# Full stack (API + dashboard)
docker compose up

# Include training and data-generation tools
docker compose --profile tools up
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health + loaded systems |
| `POST` | `/predict` | Deterministic mean-trajectory rollout |
| `POST` | `/ensemble` | Stochastic ensemble (uncertainty bands) |
| `POST` | `/steady_state` | Steady-state operating point |

Authentication: include `X-API-Key: <key>` header when `DTE_API_KEY` is set.

### Example Requests

```bash
# Health
curl http://localhost:8000/health

# Deterministic prediction (10-step horizon)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "system": "cstr",
    "initial_state": [0.8, 0.5, 325.0, 320.0],
    "controls": [[55.0, 300.0], [56.0, 302.0], [55.0, 298.0]],
    "dt": 0.1
  }'

# Stochastic ensemble (50 samples)
curl -X POST http://localhost:8000/ensemble \
  -H "Content-Type: application/json" \
  -d '{
    "system": "cstr",
    "initial_state": [0.8, 0.5, 325.0, 320.0],
    "controls": [[55.0, 300.0]],
    "n_samples": 50,
    "dt": 0.1
  }'

# Steady-state query
curl -X POST http://localhost:8000/steady_state \
  -H "Content-Type: application/json" \
  -d '{"system": "cstr"}'
```

---

## Phase 9: Adding a New System

The engine is fully decoupled from CSTR. Adding a new process requires only
registry-boundary changes; the core model/training code stays untouched.

### Step 1 — Simulator (`dte/simulators/my_system.py`)

```python
from dte.simulators.base import ProcessSimulator, SystemSpec

class MySystemSimulator(ProcessSimulator):
    @property
    def spec(self) -> SystemSpec:
        return ...  # build from config

    def dynamics(self, state, control, disturbance, params, t):
        ...  # ODEs

    def simulate(self, initial_state, controls, disturbances, params, ts):
        ...

    def steady_state(self, nominal_control, nominal_disturbance):
        ...
```

### Step 2 — Physics Loss (`dte/physics/my_system.py`)

```python
from dte.physics.base import PhysicsLoss

class MySystemPhysicsLoss(PhysicsLoss):
    def residual_names(self):
        return ["energy"]

    def compute_residuals(self, states, controls, disturbances, dt):
        return {"energy": ...}  # JAX array
```

Skip this step and use `NullPhysicsLoss` if no physics constraints are needed.

### Step 3 — Config (`configs/my_system_default.yaml`)

Define `system.name`, `system.state_dim`, `system.state_names`, `system.normalization`, `system.decoder_constraints`, and simulator-specific parameters.

### Step 4 — Register the System (`dte/simulators/registry.py`)

Add builder entries for your spec and simulator in the registry tables.

### Step 5 — Register Physics (`dte/physics/registry.py`)

If the system has physics residuals or evaluation diagnostics, register the
corresponding builders in the physics registry.

That is the complete extension path. The training, evaluation, MPC, API, and dashboard all adapt automatically.

---

## Phase 10: Hyperparameter Reference

### Model (`configs/training_default.yaml → model`)

```yaml
model:
  latent_dim: 32       # latent space dimension (try 8–32)
  hidden_dim: 128      # MLP hidden width (try 64–256)
  n_layers: 3
  drift_layers: 3
  diffusion_layers: 2
  diffusion_hidden_dim: 64
```

### Training (`training`)

```yaml
training:
  n_epochs: 100
  batch_size: 128      # reduce if OOM
  seq_len: 25          # sequence length (overridden by curriculum)
  stride: 5
  val_split: 0.2
```

### Loss Weights (`loss_weights`)

```yaml
loss_weights:
  reconstruction: 1.0
  kl: 0.0001
  trajectory: 10.0
  one_step: 0.0
  mass_balance: 0.001
  species_mass_balance: 0.001
  energy_balance: 0.001
```

### Stochastic SDE (`sde_training`)

```yaml
sde_training:
  enabled: false        # set true to activate diffusion path
  warmup_steps: 2000    # delay before SDE path activates
  sde_kl_weight: 0.00001
```

### Curriculum (`curriculum`)

```yaml
curriculum:
  enabled: false
  initial_seq_len: 5
  final_seq_len: 50
  warmup_epochs: 20
```

### Teacher Forcing (`teacher_forcing`)

```yaml
teacher_forcing:
  initial_ratio: 1.0   # start: 100% one-step loss
  final_ratio: 0.0     # end:   100% free-rollout loss
  anneal_epochs: 30
```

### Optimizer (`optimizer`)

```yaml
optimizer:
  peak_lr: 5.0e-4
  warmup_steps: 200
  total_steps: 5000
  end_lr: 1.0e-6
  gradient_clip: 0.5
```

### MPC (`configs/mpc_default.yaml`)

```yaml
mpc:
  horizon: 10           # prediction horizon (try 5–20)
  n_candidates: 500     # CEM candidate trajectories
  n_elite: 50           # elite fraction for refinement
  n_iterations: 3       # CEM iterations
```

---

## Troubleshooting

### Out of memory during training

```bash
python scripts/train.py --batch_size 16 \
  --config configs/training_default.yaml
```
Or reduce `seq_len` in the config.

### NaN losses

```yaml
# configs/training_default.yaml
optimizer:
  peak_lr: 1.0e-4
  gradient_clip: 0.5
loss_weights:
  mass_balance: 0.01
  energy_balance: 0.01
```

### Training too slow

```python
import jax
print(jax.devices())   # should list GPU(s)
```
If only CPU is shown, check your CUDA / JAX installation.

### Real data: "No valid trajectory windows"

- Reduce `--trajectory_duration` to match available data length.
- Increase `--max_gap_fill` to tolerate larger sensor gaps.
- Set `--drop_large_gaps` only when gaps are genuinely unusable.

### API returns 503 on /predict

The model checkpoint was not found. Check `DTE_MODEL_PATH` points to a valid `.eqx` file and that the path is mounted in the container (`-v $(pwd)/outputs:/app/outputs`).

### Drift detector fires too often

Increase `drift_threshold` or `drift_slack` in `OnlineAdapterConfig`, or increase `drift_reference_steps` to give the baseline estimator more data.

---

## Quick Reference

```bash
# Minimal end-to-end smoke test
python scripts/generate_data.py --n_trajectories 100 --output_dir data/test/
python scripts/train.py --data_dir data/test/ --n_epochs 3 --batch_size 8 \
  --output_dir outputs/test/
python scripts/evaluate.py \
  --model_path outputs/test/final_model.eqx \
  --config outputs/test/config.yaml \
  --data_dir data/test/ --output_dir outputs/test/eval/

# Heat exchanger end-to-end
python scripts/generate_data.py \
  --config configs/heat_exchanger_default.yaml --output_dir data/hx/
python scripts/train.py \
  --config configs/heat_exchanger_training.yaml \
  --system_config configs/heat_exchanger_default.yaml \
  --data_dir data/hx/ --output_dir outputs/hx_v1/

# Deploy
docker compose up
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## Success Metrics Summary

| Phase | Metric | Target |
|---|---|---|
| Data generation | No NaN values | 100% |
| Real data ingestion | Trajectories extracted | > 0 |
| Training | Validation loss | < 0.1 |
| Evaluation | 1-step normalised MSE | < 0.01 |
| Evaluation | Mass balance violation | < 0.01 |
| Evaluation | Energy balance violation | < 1.0 |
| Control | AI-MPC vs PID ISE improvement | > 20% |
| Transfer | Few-shot MSE vs zero-shot | Reduction |
| Online adaptation | Drift detection and recovery | Working |
| API | `/health` responds | 200 OK |
| Dashboard | Interactive demo | Running |

---

## Additional Resources

- `README.md` — Project overview and API reference snippets
- `QUICK_START.md` — Condensed getting started guide
- `configs/training_default.yaml` — Annotated training configuration
- `configs/cstr_default.yaml` — Annotated system spec with all fields
- `notebooks/01_exploration.ipynb` — Interactive CSTR examples
- `tests/` — Unit tests showing usage of every module
- `program.md` — Autonomous agent operating rules
