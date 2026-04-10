# Phase 6 Demo App

Date: 2026-04-10

Phase 6 adds a customer-facing demo surface for the repo: a dedicated interactive frontend plus backend demo endpoints that expose simulator rollouts, uncertainty, scenario comparison, and lightweight control suggestions.

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
  - `GET /demo/catalog`
  - `POST /demo/simulate`
  - `POST /demo/rollout`
  - `POST /demo/optimize_control`
  - `POST /demo/compare_scenarios`
- `dte/api/models.py`
  - request/response models for the new demo routes

Frontend:

- `app/demo_app.py`
  - dedicated Streamlit demo site
  - three interactive unit demos:
    - CSTR
    - Heat Exchanger
    - Two-Tank System
  - flowsheet preview section for Phase 3 example graphs

Config:

- `configs/demo_app.yaml`
  - demo copy
  - system mapping
  - horizons and target states

## Demo Behavior

The frontend is intentionally separate from the existing training dashboard.

- `app/dashboard.py` remains the artifact/run inspection surface
- `app/demo_app.py` is the customer-facing interactive site

Each unit demo supports:

- direct control slider changes
- live scenario comparison against a baseline trajectory
- uncertainty bands
- constraint/risk summaries
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
- a flowsheet preview section
- reusable backend routes for demo interactions
- config-driven demo definitions

What is still intentionally thin:

- the flowsheet page is a preview, not a full interactive plant graph simulator
- the optimizer is a lightweight demo optimizer, not the full Phase 7 MPC interface
- when no trained checkpoint is loaded, uncertainty uses a simulator ensemble fallback

## Verification

- `JAX_PLATFORMS=cpu pytest tests/test_demo_engine.py tests/test_api_demo_routes.py -q`
- `python -m py_compile dte/demo/__init__.py dte/demo/engine.py dte/api/models.py dte/api/service.py app/demo_app.py`

The route tests explicitly cover the no-checkpoint fallback so the demos remain usable locally without pre-trained weights.

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
