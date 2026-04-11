# Phase 6 Demo App

Date: 2026-04-10

Phase 6 adds a customer-facing demo surface for the repo: a dedicated interactive frontend plus backend demo endpoints that expose simulator rollouts, uncertainty, scenario comparison, and lightweight control suggestions.

The current V1 presentation layer now goes one step further: both the Streamlit app and the FastAPI service are wired to the blessed universal checkpoint and milestone artifacts from the release workspace, while preserving legacy fallback behavior when those artifacts are unavailable.

## What Landed

Backend:

- `dte/demo/engine.py`
  - shared demo catalog loading
  - deterministic simulator rollouts
  - uncertainty bands through model rollout when a checkpoint is loaded
  - simulator-ensemble fallback when no checkpoint is loaded
  - scenario comparison
  - lightweight random-shooting control optimization
- `dte/api/service.py`
  - loads the same universal release runtime used by the Streamlit app when `DTE_DEMO_CONFIG` exposes it
  - `GET /demo/catalog`
  - `POST /demo/simulate`
  - `POST /demo/rollout`
  - `POST /demo/optimize_control`
  - `POST /demo/compare_scenarios`
  - `/predict` and `/ensemble` now prefer the shared universal runtime and fall back to legacy single-system checkpoints only when the universal runtime is unavailable
- `dte/api/models.py`
  - request/response models for the new demo routes

Frontend:

- `app/demo_app.py`
  - dedicated Streamlit demo site
  - release overview bound to the V1 milestone summaries
  - three preset-driven unit demos:
    - CSTR
    - Heat Exchanger
    - Two-Tank System
  - customer adaptation proof section
  - flowsheet preview section for Phase 3 example graphs

Config:

- `configs/demo_app.yaml`
  - demo copy
  - release artifact paths
  - system mapping
  - horizons and target states
  - baseline, candidate, and disturbance presets for each demo

Release runtime:

- `dte/demo/engine.py`
  - `load_demo_model_runtime(...)`
    - loads the blessed universal checkpoint from the configured release workspace
  - `load_demo_release_snapshot(...)`
    - condenses train/eval/milestone/customer artifacts into a presentation-friendly snapshot
  - `build_signal_sequence(...)`
    - turns config presets into bounded control and disturbance sequences
  - `rollout_with_universal_model(...)`
    - runs app rollouts through the shared checkpoint when available

## Demo Behavior

The frontend is intentionally separate from the existing training dashboard.

- `app/dashboard.py` remains the artifact/run inspection surface
- `app/demo_app.py` is the customer-facing interactive site

Each unit demo now supports:

- a fixed baseline operating policy
- curated disturbance presets
- curated candidate operating moves
- optional fine trim on top of those presets
- live scenario comparison against the baseline trajectory
- uncertainty bands from the blessed universal checkpoint when available
- simulator-ensemble fallback when the checkpoint is unavailable
- one-click control recommendations toward a target state

## Running It

### Demo frontend

```bash
source .venv/bin/activate
streamlit run app/demo_app.py
```

### Demo API

```bash
source .venv/bin/activate
python -m dte.api.service --host 0.0.0.0 --port 8000 \
  --system_config configs/cstr_default.yaml,configs/heat_exchanger_default.yaml,configs/two_tank_default.yaml
```

Optional environment variable:

```bash
export DTE_DEMO_CONFIG=configs/demo_app.yaml
```

## Route Summary

- `/demo/catalog`
  - demo metadata and flowsheet previews
- `/demo/simulate`
  - deterministic simulator trajectory
- `/demo/rollout`
  - mean trajectory plus uncertainty bands
- `/demo/optimize_control`
  - lightweight recommended control schedule
- `/demo/compare_scenarios`
  - baseline/candidate comparison for the frontend

## Current Scope

What is covered now:

- three local interactive unit demos
- release overview bound to milestone artifacts
- customer adaptation proof in the app
- a flowsheet preview section
- reusable backend routes for demo interactions
- config-driven demo definitions

What is still intentionally thin:

- the flowsheet page is a preview, not a full interactive plant graph simulator
- the optimizer is a lightweight demo optimizer, not the full Phase 7 MPC interface
- the legacy single-system checkpoint path still exists as a fallback for inference compatibility

## Verification

- `JAX_PLATFORMS=cpu pytest tests/test_demo_engine.py tests/test_api_demo_routes.py -q`
- `python -m py_compile dte/demo/__init__.py dte/demo/engine.py dte/api/models.py dte/api/service.py app/demo_app.py`
- `timeout 25s streamlit run app/demo_app.py --server.headless true --server.port 8503`
- `curl -I http://127.0.0.1:8503`

The route tests explicitly cover both the no-checkpoint fallback and the universal-runtime API path. The Streamlit presentation layer and the FastAPI service now share the same blessed universal checkpoint from the configured V1 release workspace.

## Smoke Runner

Phase 6 now includes a reusable smoke runner:

- `scripts/smoke_phase6.py`

It starts the real FastAPI service and the real Streamlit app, waits for health checks, exercises the demo API surface, fetches the Streamlit root HTML, and writes logs plus a workspace `summary.json`.

Default usage:

```bash
source .venv/bin/activate
python scripts/smoke_phase6.py
```

Useful variants:

```bash
python scripts/smoke_phase6.py --dry_run
python scripts/smoke_phase6.py --workspace_dir outputs/phase6_smoke/manual_run
python scripts/smoke_phase6.py --skip_streamlit
python scripts/smoke_phase6.py --skip_api
```

The runner defaults subprocesses to `JAX_PLATFORMS=cpu` so smoke runs stay reproducible and do not depend on GPU availability.
