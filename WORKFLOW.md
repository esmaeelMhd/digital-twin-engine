# Digital Twin Engine Workflow

Updated for the post-Phase-7 architecture on 2026-04-10.

This file replaces the earlier workflow that assumed only the older single-system stack plus a partial generalization effort. The repository now contains a broader architecture:

- single-system unit modeling with `DigitalTwin`
- universal mixed-system modeling with `UniversalDigitalTwin`
- unit adapters and calibration
- flowsheet graph modeling
- modular law layers
- customer adaptation workflow
- demo website and demo API
- generic MPC / RL-readiness interfaces

Phase 8 from `plan.md` is still not implemented. Everything else through Phase 7 is now present in some usable form.

---

## 1. Choose The Workflow You Actually Need

Use this file as an entry-point selector.

| Goal | Primary path |
| --- | --- |
| Train one system end to end | Section 4 |
| Train one shared universal checkpoint | Section 5 |
| Calibrate a universal model to a new unit | Section 6 |
| Run customer onboarding and automatic adaptation | Section 7 |
| Work on small flowsheets | Section 8 |
| Use modular chemistry / thermo / biology law layers | Section 9 |
| Run the demo website or demo API | Section 10 |
| Use the repo for MPC or RL-style control experiments | Section 11 |
| Validate a phase quickly | Section 12 |

---

## 2. Current Architecture In Practice

The repo now has seven practical layers:

1. **Unit model path**
   - `dte/models/digital_twin.py`
   - `scripts/train.py`
   - `scripts/evaluate.py`

2. **Universal foundation path**
   - `dte/models/universal_digital_twin.py`
   - `scripts/train_universal.py`
   - `scripts/evaluate_universal.py`

3. **Calibration and customer adaptation**
   - `scripts/calibrate_unit.py`
   - `scripts/adapt_customer.py`

4. **Flowsheet graph path**
   - `dte/flowsheet/*`
   - `dte/models/flowsheet_model.py`
   - `dte/training/flowsheet_trainer.py`

5. **Law-layer path**
   - `dte/laws/*`
   - optional law-bundle integration through `dte/physics/registry.py`

6. **Demo and serving path**
   - `app/demo_app.py`
   - `app/dashboard.py`
   - `dte/api/service.py`

7. **Control path**
   - legacy: `dte/control/mpc.py`, `scripts/run_mpc.py`
   - new: `dte/control/mpc_interface.py`, `dte/control/rl_env.py`, `dte/control/state_correction.py`

Recommended reading if you need phase-specific detail:

- [docs/repo_audit.md](docs/repo_audit.md)
- [docs/implementation_mapping.md](docs/implementation_mapping.md)
- [docs/phase1_unit_foundation_model.md](docs/phase1_unit_foundation_model.md)
- [docs/phase2_adapters_and_calibration.md](docs/phase2_adapters_and_calibration.md)
- [docs/phase3_flowsheet_graph_modeling.md](docs/phase3_flowsheet_graph_modeling.md)
- [docs/phase4_modular_law_layers.md](docs/phase4_modular_law_layers.md)
- [docs/phase5_customer_adaptation_workflow.md](docs/phase5_customer_adaptation_workflow.md)
- [docs/phase6_demo_app.md](docs/phase6_demo_app.md)
- [docs/phase7_mpc_and_drl_readiness.md](docs/phase7_mpc_and_drl_readiness.md)

---

## 3. Setup And Baseline Verification

### 3.1 Environment

```bash
cd digital-twin-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3.2 Verify Install

```bash
python scripts/verify_install.py
```

### 3.3 Run The Test Suite

```bash
pytest tests/ -v
```

### 3.4 Quick Smoke Checks

If you want fast validation instead of the full suite, use the phase smoke runners in Section 12.

---

## 4. Single-System Unit Workflow

This is still the simplest path when you want one digital twin for one process system.

### 4.1 Generate Data

```bash
python scripts/generate_data.py \
  --config configs/cstr_default.yaml \
  --n_trajectories 1000 \
  --n_steps 200 \
  --output_dir data/cstr/
```

Any registered system works the same way:

```bash
python scripts/generate_data.py \
  --config configs/heat_exchanger_default.yaml \
  --output_dir data/heat_exchanger/

python scripts/generate_data.py \
  --config configs/two_tank_default.yaml \
  --output_dir data/two_tank/
```

### 4.2 Ingest Real Plant Data

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
  --trajectory_stride 10.0
```

### 4.3 Train

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

The single-system path remains backward-compatible, but can now optionally use the Phase 1 grouped encoder through config:

```yaml
model:
  grouped_encoder:
    enabled: true
```

### 4.4 Evaluate

```bash
python scripts/evaluate.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --config outputs/cstr_v1/config.yaml \
  --system_config configs/cstr_default.yaml \
  --data_dir data/cstr/ \
  --output_dir outputs/cstr_v1/eval/
```

### 4.5 Legacy Closed-Loop MPC

```bash
python scripts/run_mpc.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --model_config outputs/cstr_v1/config.yaml \
  --system_config configs/cstr_default.yaml \
  --setpoint_T 340.0 \
  --setpoint_Ca 0.8 \
  --disturbance_scenario step \
  --n_steps 200 \
  --output_dir outputs/mpc_results/
```

Notes:

- `--compare_pid` is only meaningful for the legacy CSTR path.
- For generic control workflows, prefer Section 11.

---

## 5. Universal Foundation Workflow

Use this when you want one shared checkpoint across multiple systems.

### 5.1 Prepare The Data Directories Referenced By `configs/training_universal.yaml`

By default the universal config expects:

- `data/cstr/`
- `data/heat_exchanger/`
- `data/two_tank/`

Generate those datasets first if they do not exist.

### 5.2 Train A Shared Checkpoint

```bash
python scripts/train_universal.py \
  --config configs/training_universal.yaml \
  --output_dir outputs/universal_v1/ \
  --n_epochs 20 \
  --batch_size 128 \
  --seed 42
```

### 5.3 Evaluate The Shared Checkpoint

```bash
python scripts/evaluate_universal.py \
  --model_path outputs/universal_v1/best_model.eqx \
  --config outputs/universal_v1/config.yaml \
  --output_dir outputs/universal_v1/eval/
```

This path now includes Phase 1 and Phase 2 additions:

- typed unit metadata through `ProcessUnitSpec`
- family / subtype / law-tag conditioning
- optional adapters
- multi-horizon loss support
- uncertainty and local control-sensitivity reporting

---

## 6. Calibration Workflow For A New Unit

Use this when you already have a pretrained universal checkpoint and want to adapt it to a new target unit dataset.

```bash
python scripts/calibrate_unit.py \
  --model_path outputs/universal_v1/best_model.eqx \
  --config configs/training_universal.yaml \
  --system_config configs/cstr_default.yaml \
  --data_dir data/cstr_variant/ \
  --system_name cstr_variant \
  --output_dir outputs/cstr_variant_calibration/ \
  --trainable_mode adapters \
  --tune_normalization
```

Use this path when:

- the target unit is already known
- you have target data
- you want direct calibration without the customer-onboarding/reporting wrapper

---

## 7. Customer Adaptation Workflow

Use this when you want the full Phase 5 onboarding → template matching → adaptation → reporting flow.

```bash
python scripts/adapt_customer.py \
  --onboarding path/to/customer_onboarding.yaml \
  --model_path outputs/universal_v1/best_model.eqx \
  --config configs/training_universal.yaml \
  --system_config configs/cstr_default.yaml \
  --data_dir data/cstr_variant/ \
  --output_dir outputs/customer_cstr_variant/ \
  --trainable_mode adapters \
  --tune_normalization
```

This path produces:

- onboarding snapshot
- template matches
- calibrated target config
- summary JSON
- validation report JSON
- validation report Markdown

Current scope:

- unit adaptation is end-to-end
- flowsheet template matching exists
- flowsheet calibration is not yet wired through this CLI

---

## 8. Flowsheet Workflow

Phase 3 added a small flowsheet graph stack. It is usable, but still a thin slice.

### 8.1 Best Entry Point: The Smoke Runner

```bash
python scripts/smoke_phase3.py
```

This is currently the most practical end-to-end way to validate the flowsheet path because there is not yet a dedicated Phase 3 CLI beyond the synthetic/smoke tooling.

### 8.2 Python API Workflow

The intended path is:

1. build an example flowsheet from `dte/flowsheet/examples.py`
2. build a synthetic dataset from `dte/flowsheet/synthetic.py`
3. train with `dte/training/flowsheet_trainer.py`

See [docs/phase3_flowsheet_graph_modeling.md](docs/phase3_flowsheet_graph_modeling.md) for the working example.

Current scope:

- small plant-section graphs
- synthetic graph datasets
- train/validate loop
- proxy plant losses

Not yet present:

- real-data flowsheet ingestion
- a dedicated high-level CLI
- full plant thermodynamics / chemistry

---

## 9. Law-Layer Workflow

Phase 4 adds reusable chemistry, thermo, and biology law bundles that can augment the existing physics-loss path.

### 9.1 Exercise The Law Layer

```bash
python scripts/smoke_phase4.py
```

### 9.2 Use Example Configs

Relevant examples:

- `configs/cstr_law_example.yaml`
- `configs/bioreactor_law_example.yaml`

Law bundles are opt-in through config:

```yaml
laws:
  enabled: true
  chemistry:
    - name: primary_reaction
      kind: arrhenius_reaction
```

Important scope note:

- the law layer augments the physics-loss path
- it does not yet inject law features directly into the model encoder/decoder

---

## 10. Demo And Serving Workflow

There are now two different Streamlit surfaces.

### 10.1 Training / Inspection Dashboard

```bash
streamlit run app/dashboard.py
```

This remains the run-inspection dashboard.

### 10.2 Customer-Facing Demo Site

```bash
streamlit run app/demo_app.py
```

This is the Phase 6 interactive demo surface.

### 10.3 Demo API

```bash
python -m dte.api.service --host 0.0.0.0 --port 8000 \
  --system_config configs/cstr_default.yaml,configs/heat_exchanger_default.yaml,configs/two_tank_default.yaml
```

The API now serves both the original inference routes and the demo routes:

- `/health`
- `/predict`
- `/ensemble`
- `/steady_state`
- `/demo/catalog`
- `/demo/simulate`
- `/demo/rollout`
- `/demo/optimize_control`
- `/demo/compare_scenarios`

### 10.4 Full Demo Smoke

```bash
python scripts/smoke_phase6.py
```

Important scope note:

- the demo flowsheet surface is currently a preview/catalog, not a full interactive plant simulator

---

## 11. Control Workflow

There are now two control entry styles: legacy script-level MPC and the new generic Phase 7 interfaces.

### 11.1 Legacy Script-Level MPC

Use [Section 4.5](#45-legacy-closed-loop-mpc) when you want the old `scripts/run_mpc.py` flow.

### 11.2 Generic Control Runtime

Use `ProcessMPCInterface` when you want a process-agnostic control-facing runtime:

```python
from dte.control import MPCInterfaceConfig, ProcessMPCInterface

runtime = ProcessMPCInterface(spec, simulator, model=model, config=MPCInterfaceConfig(dt=0.1))
runtime.reset()
best = runtime.optimize_random_shooting(
    target_state=spec.default_initial_state_array(),
    horizon=12,
    n_candidates=16,
)
```

### 11.3 RL-Style Environment

```python
from dte.control import ProcessControlEnv

env = ProcessControlEnv(spec, simulator)
obs, info = env.reset(seed=0)
action = env.action_space.sample(seed=1)
obs, reward, terminated, truncated, info = env.step(action)
```

### 11.4 Measurement Correction

```python
from dte.control import StateCorrectionHook

hook = StateCorrectionHook(spec, model=model)
update = hook.correct(
    prior_state=prior_state,
    measurement=measurement,
    control=control,
)
```

### 11.5 Full Control Smoke

```bash
python scripts/smoke_phase7.py
```

Current scope:

- generic rollout/evaluate hooks exist
- the RL env is Gymnasium-style but does not require `gymnasium`
- no bundled RL training algorithm stack exists yet

---

## 12. Smoke Runner Matrix

Use these to validate one phase quickly.

| Phase | Smoke runner | Purpose |
| --- | --- | --- |
| 1 | `scripts/smoke_phase1.py` | typed unit spec, grouped encoder, universal loss/eval additions |
| 2 | `scripts/smoke_phase2.py` | family conditioning, adapters, calibration |
| 3 | `scripts/smoke_phase3.py` | flowsheet graph synthetic train/eval path |
| 4 | `scripts/smoke_phase4.py` | modular law bundles and law-augmented physics |
| 5 | `scripts/smoke_phase5.py` | customer adaptation end to end |
| 6 | `scripts/smoke_phase6.py` | demo API + demo frontend |
| 7 | `scripts/smoke_phase7.py` | MPC runtime, RL env, correction hooks, control metrics |

Default pattern:

```bash
source .venv/bin/activate
python scripts/smoke_phaseN.py
```

Most smoke runners also support:

- `--dry_run`
- `--workspace_dir outputs/phaseN_smoke/manual_run`

The later smoke runners default to CPU-oriented execution for reproducibility on ordinary dev machines.

---

## 13. Deployment Workflow

### 13.1 Local API

```bash
export DTE_SYSTEM_CONFIG=configs/cstr_default.yaml
export DTE_MODEL_PATH=outputs/cstr_v1/best_model.eqx
export DTE_TRAINING_CONFIG=configs/training_default.yaml

uvicorn dte.api.service:app --host 0.0.0.0 --port 8000
```

### 13.2 Docker Images

```bash
docker build --target api -t dte-api .
docker build --target train -t dte-train .
```

### 13.3 Compose Stack

```bash
docker compose up
docker compose --profile tools up
```

Important note:

- `docker compose up` currently starts the API and the legacy `app/dashboard.py` dashboard
- it does **not** currently start `app/demo_app.py`

---

## 14. Extending The Repository With A New Unit System

The extension boundary is still registry-driven.

1. Add a simulator under `dte/simulators/`
2. Add or reuse a `PhysicsLoss` under `dte/physics/`
3. Add a system config under `configs/`
4. Register the system in `dte/simulators/registry.py`
5. Register physics / diagnostics in `dte/physics/registry.py`

With the newer architecture, you should also consider:

- typed `state_channels`
- typed `control_channels`
- typed `disturbance_channels`
- `family`, `subtype`, `unit_type`
- `law_tags`
- `conditioning_tags`
- topology ports if the unit will later connect into a flowsheet

That is enough for:

- single-system training
- universal training
- calibration
- customer template matching
- demo-catalog inclusion
- control-interface reuse

---

## 15. Current Scope Boundaries

These limits are important when planning work:

- Phase 8 distributed / transport-aware modeling is still missing
- flowsheet modeling is implemented as a thin slice, not a full plant stack
- flowsheet adaptation is not yet wired through `scripts/adapt_customer.py`
- the Phase 6 flowsheet demo is a preview surface, not a full interactive simulator
- the Phase 7 control layer is usable, but still intentionally lightweight

---

## 16. Recommended Order For New Users

If you are new to the repo, the lowest-friction path is:

1. Set up the environment and run `scripts/verify_install.py`
2. Generate one single-system dataset
3. Train one single-system model
4. Evaluate it
5. Run `scripts/smoke_phase6.py` to see the demo/API surface
6. Run `scripts/smoke_phase7.py` to see the control surface
7. Move to universal training and customer adaptation only after the single-system path is familiar

If you are validating the full roadmap rather than one feature, run:

1. `scripts/smoke_phase1.py`
2. `scripts/smoke_phase2.py`
3. `scripts/smoke_phase3.py`
4. `scripts/smoke_phase4.py`
5. `scripts/smoke_phase5.py`
6. `scripts/smoke_phase6.py`
7. `scripts/smoke_phase7.py`

---

## 17. Additional References

- `README.md`
- `QUICK_START.md`
- `AGENTS.md`
- `docs/repo_audit.md`
- `docs/implementation_mapping.md`
- `program.md`
- `tests/`
