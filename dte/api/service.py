"""FastAPI service for the Digital Twin Engine.

Provides REST endpoints for:
- ``GET  /health``         -- Health check + loaded system list
- ``POST /predict``        -- Deterministic rollout (mean trajectory)
- ``POST /ensemble``       -- Stochastic ensemble rollout (uncertainty bands)
- ``POST /steady_state``   -- Find steady-state operating point

Authentication
--------------
Set the environment variable ``DTE_API_KEY`` to require API-key authentication
on all endpoints.  Requests must include ``X-API-Key: <key>`` in the header.
If ``DTE_API_KEY`` is unset, authentication is disabled (development mode).

Usage
-----
Launch with uvicorn::

    uvicorn dte.api.service:app --host 0.0.0.0 --port 8000

Or with the bundled entrypoint::

    python -m dte.api.service --host 0.0.0.0 --port 8000 --system_config configs/cstr_default.yaml

Environment variables
---------------------
- ``DTE_SYSTEM_CONFIG``  Path to system YAML (default: configs/cstr_default.yaml)
- ``DTE_MODEL_PATH``     Path to trained model checkpoint (default: outputs/best_model.eqx)
- ``DTE_TRAINING_CONFIG`` Path to training YAML (default: configs/training_default.yaml)
- ``DTE_API_KEY``        Optional API key for authentication
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from dte.api.models import (
    DemoCatalogResponse,
    DemoCompareScenariosRequest,
    DemoCompareScenariosResponse,
    DemoOptimizeControlRequest,
    DemoOptimizeControlResponse,
    DemoRolloutRequest,
    DemoRolloutResponse,
    DemoSimulateRequest,
    DemoSimulateResponse,
    EnsembleRequest,
    EnsembleResponse,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SteadyStateRequest,
    SteadyStateResponse,
)
from dte.demo.engine import (
    compare_scenarios,
    constraint_summary,
    default_disturbance_sequence,
    demo_catalog_from_config,
    load_demo_config,
    optimize_control_sequence,
    rollout_scenario,
    simulate_open_loop,
    time_axis,
)
from dte.models.digital_twin import DigitalTwin
from dte.simulators.base import SystemSpec
from dte.simulators.registry import get_system_spec, get_simulator


# ---------------------------------------------------------------------------
# Global model registry (populated at startup)
# ---------------------------------------------------------------------------

_models: Dict[str, DigitalTwin] = {}
_specs: Dict[str, SystemSpec] = {}
_system_configs: Dict[str, dict] = {}
_startup_time: float = 0.0


def _load_system(system_config_path: str, model_path: Optional[str], training_config_path: str):
    """Load a system spec and optionally a model checkpoint."""
    with open(system_config_path) as f:
        sys_cfg = yaml.safe_load(f)
    with open(training_config_path) as f:
        train_cfg = yaml.safe_load(f)

    spec = get_system_spec(sys_cfg)
    _specs[spec.name] = spec
    _system_configs[spec.name] = sys_cfg

    if model_path and os.path.exists(model_path):
        model = DigitalTwin.load(
            model_path,
            train_cfg,
            system_spec=spec,
            system_config=sys_cfg,
        )
        _models[spec.name] = model
        print(f"[DTE API] Loaded model for system '{spec.name}' from {model_path}")
    else:
        print(
            f"[DTE API] No model found at '{model_path}' for system '{spec.name}'. "
            "Predict/ensemble endpoints will return 503."
        )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load models at startup."""
    global _startup_time
    _startup_time = time.time()

    sys_config_env = os.environ.get("DTE_SYSTEM_CONFIG", "configs/cstr_default.yaml")
    model_path_env = os.environ.get("DTE_MODEL_PATH", "outputs/best_model.eqx")
    train_config_env = os.environ.get("DTE_TRAINING_CONFIG", "configs/training_default.yaml")

    # Support comma-separated list for multiple systems
    for sys_path in sys_config_env.split(","):
        sys_path = sys_path.strip()
        if sys_path and os.path.exists(sys_path):
            _load_system(sys_path, model_path_env, train_config_env)
        elif sys_path:
            print(f"[DTE API] System config not found: {sys_path}")

    print(f"[DTE API] Ready.  Loaded systems: {list(_specs.keys())}")
    yield
    # Cleanup (nothing to do for JAX models)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Digital Twin Engine API",
    description=(
        "REST API for running physics-informed latent neural SDE digital twins.  "
        "Provides deterministic and stochastic rollout, ensemble uncertainty "
        "estimation, and steady-state queries."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_API_KEY_ENV = "DTE_API_KEY"
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(api_key: Optional[str] = Security(_api_key_header)):
    required = os.environ.get(_API_KEY_ENV)
    if required is None:
        return  # Auth disabled
    if api_key != required:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.  Supply it via 'X-API-Key' header.",
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_model_and_spec(system: str):
    spec = _specs.get(system)
    if spec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"System '{system}' not registered.  Available: {list(_specs.keys())}",
        )
    model = _models.get(system)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No trained model loaded for system '{system}'.  "
                   "Set DTE_MODEL_PATH and restart the service.",
        )
    return model, spec


def _get_demo_runtime(system: str):
    spec = _specs.get(system)
    if spec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"System '{system}' not registered.  Available: {list(_specs.keys())}",
        )
    system_config = _system_configs.get(system)
    if system_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No system config loaded for '{system}'.",
        )
    try:
        simulator = get_simulator(system, system_config)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not instantiate simulator for '{system}': {exc}",
        )
    model = _models.get(system)
    return spec, simulator, model


def _build_time_array(n_steps: int, dt: float) -> jnp.ndarray:
    return jnp.linspace(0.0, (n_steps - 1) * dt, n_steps)


def _ensure_disturbance_sequence(
    spec: SystemSpec,
    disturbances: Optional[List[List[float]]],
    n_steps: int,
) -> np.ndarray:
    if disturbances:
        values = np.asarray(disturbances, dtype=np.float32)
    else:
        values = np.asarray(default_disturbance_sequence(spec, n_steps), dtype=np.float32)
    if values.shape != (n_steps, spec.disturbance_dim):
        raise HTTPException(
            status_code=422,
            detail=(
                f"disturbances must have shape [{n_steps}, {spec.disturbance_dim}] "
                f"for system '{spec.name}'."
            ),
        )
    return values


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Return service health and loaded systems."""
    return HealthResponse(
        status="ok",
        loaded_systems=list(_specs.keys()),
        version="0.1.0",
    )


@app.get(
    "/demo/catalog",
    response_model=DemoCatalogResponse,
    tags=["demo"],
    dependencies=[Depends(_verify_api_key)],
)
async def demo_catalog():
    """Return the interactive demo catalog and flowsheet previews."""

    demo_config_path = os.environ.get("DTE_DEMO_CONFIG", "configs/demo_app.yaml")
    config = load_demo_config(demo_config_path)
    catalog = demo_catalog_from_config(config, _system_configs)
    return DemoCatalogResponse.model_validate(catalog)


@app.post(
    "/demo/simulate",
    response_model=DemoSimulateResponse,
    tags=["demo"],
    dependencies=[Depends(_verify_api_key)],
)
async def demo_simulate(req: DemoSimulateRequest):
    """Run a deterministic simulator rollout for the demo app."""

    spec, simulator, _ = _get_demo_runtime(req.system)
    controls = np.asarray(req.controls, dtype=np.float32)
    if controls.shape[1] != spec.control_dim:
        raise HTTPException(
            status_code=422,
            detail=f"controls must have shape [T, {spec.control_dim}] for '{req.system}'.",
        )
    disturbances = _ensure_disturbance_sequence(spec, req.disturbances, controls.shape[0])
    states = simulate_open_loop(
        spec,
        simulator,
        np.asarray(req.initial_state, dtype=np.float32),
        controls,
        disturbances,
        float(req.dt),
    )
    return DemoSimulateResponse(
        system=req.system,
        times=time_axis(controls.shape[0], float(req.dt)).tolist(),
        states=states.tolist(),
        state_names=spec.state_names,
        constraint_summary={
            key: float(value)
            for key, value in constraint_summary(spec, states).items()
        },
    )


@app.post(
    "/demo/rollout",
    response_model=DemoRolloutResponse,
    tags=["demo"],
    dependencies=[Depends(_verify_api_key)],
)
async def demo_rollout(req: DemoRolloutRequest):
    """Return mean trajectory and uncertainty bands for one scenario."""

    spec, simulator, model = _get_demo_runtime(req.system)
    controls = np.asarray(req.controls, dtype=np.float32)
    if controls.shape[1] != spec.control_dim:
        raise HTTPException(
            status_code=422,
            detail=f"controls must have shape [T, {spec.control_dim}] for '{req.system}'.",
        )
    disturbances = _ensure_disturbance_sequence(spec, req.disturbances, controls.shape[0])
    result = rollout_scenario(
        spec,
        simulator,
        initial_state=np.asarray(req.initial_state, dtype=np.float32),
        controls=controls,
        disturbances=disturbances,
        dt=float(req.dt),
        model=model,
        params=None if req.params is None else np.asarray(req.params, dtype=np.float32),
        n_samples=int(req.n_samples),
        seed=int(req.seed),
    )
    return DemoRolloutResponse(
        system=req.system,
        source=result["source"],
        times=np.asarray(result["times"]).tolist(),
        mean=np.asarray(result["mean"]).tolist(),
        std=np.asarray(result["std"]).tolist(),
        p05=np.asarray(result["p05"]).tolist(),
        p95=np.asarray(result["p95"]).tolist(),
        state_names=spec.state_names,
        constraint_summary={
            key: float(value) for key, value in result["constraint_summary"].items()
        },
    )


@app.post(
    "/demo/optimize_control",
    response_model=DemoOptimizeControlResponse,
    tags=["demo"],
    dependencies=[Depends(_verify_api_key)],
)
async def demo_optimize_control(req: DemoOptimizeControlRequest):
    """Return a lightweight random-shooting control recommendation."""

    spec, simulator, _ = _get_demo_runtime(req.system)
    disturbances = np.asarray(req.disturbances, dtype=np.float32)
    if disturbances.shape[1] != spec.disturbance_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                f"disturbances must have shape [T, {spec.disturbance_dim}] for '{req.system}'."
            ),
        )
    result = optimize_control_sequence(
        spec,
        simulator,
        initial_state=np.asarray(req.initial_state, dtype=np.float32),
        disturbances=disturbances,
        dt=float(req.dt),
        target_state=np.asarray(req.target_state, dtype=np.float32),
        tracked_state_names=req.tracked_state_names,
        n_candidates=int(req.n_candidates),
        seed=int(req.seed),
    )
    return DemoOptimizeControlResponse(
        system=req.system,
        control_sequence=np.asarray(result["control_sequence"]).tolist(),
        predicted_states=np.asarray(result["predicted_states"]).tolist(),
        objective=float(result["objective"]),
        tracked_state_names=list(result["tracked_state_names"]),
        state_names=spec.state_names,
        constraint_summary={
            key: float(value) for key, value in result["constraint_summary"].items()
        },
    )


@app.post(
    "/demo/compare_scenarios",
    response_model=DemoCompareScenariosResponse,
    tags=["demo"],
    dependencies=[Depends(_verify_api_key)],
)
async def demo_compare_scenarios(req: DemoCompareScenariosRequest):
    """Compare baseline and candidate control schedules for one demo system."""

    spec, simulator, model = _get_demo_runtime(req.system)
    baseline_controls = np.asarray(req.baseline_controls, dtype=np.float32)
    candidate_controls = np.asarray(req.candidate_controls, dtype=np.float32)
    if baseline_controls.shape != candidate_controls.shape:
        raise HTTPException(
            status_code=422,
            detail="baseline_controls and candidate_controls must have the same shape.",
        )
    if baseline_controls.shape[1] != spec.control_dim:
        raise HTTPException(
            status_code=422,
            detail=f"controls must have shape [T, {spec.control_dim}] for '{req.system}'.",
        )
    disturbances = _ensure_disturbance_sequence(
        spec,
        req.disturbances,
        baseline_controls.shape[0],
    )
    result = compare_scenarios(
        spec,
        simulator,
        initial_state=np.asarray(req.initial_state, dtype=np.float32),
        baseline_controls=baseline_controls,
        candidate_controls=candidate_controls,
        disturbances=disturbances,
        dt=float(req.dt),
        model=model,
        params=None if req.params is None else np.asarray(req.params, dtype=np.float32),
        n_samples=int(req.n_samples),
        seed=int(req.seed),
    )
    baseline = result["baseline"]
    candidate = result["candidate"]
    return DemoCompareScenariosResponse(
        system=req.system,
        state_names=spec.state_names,
        baseline_mean=np.asarray(baseline["mean"]).tolist(),
        candidate_mean=np.asarray(candidate["mean"]).tolist(),
        baseline_p05=np.asarray(baseline["p05"]).tolist(),
        baseline_p95=np.asarray(baseline["p95"]).tolist(),
        candidate_p05=np.asarray(candidate["p05"]).tolist(),
        candidate_p95=np.asarray(candidate["p95"]).tolist(),
        summary=result["summary"],
        baseline_constraints={
            key: float(value) for key, value in baseline["constraint_summary"].items()
        },
        candidate_constraints={
            key: float(value) for key, value in candidate["constraint_summary"].items()
        },
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    dependencies=[Depends(_verify_api_key)],
)
async def predict(req: PredictRequest):
    """Run a deterministic mean-trajectory rollout.

    Returns the predicted state sequence over the control horizon.
    """
    model, spec = _get_model_and_spec(req.system)

    n_steps = len(req.controls)
    if n_steps == 0:
        raise HTTPException(status_code=422, detail="controls must be non-empty.")

    controls = jnp.array(req.controls, dtype=jnp.float32)  # (T, control_dim)
    disturbances = (
        jnp.array(req.disturbances, dtype=jnp.float32)
        if req.disturbances
        else jnp.zeros((n_steps, spec.disturbance_dim), dtype=jnp.float32)
    )
    params = (
        jnp.array(req.params, dtype=jnp.float32)
        if req.params
        else jnp.ones(spec.param_dim, dtype=jnp.float32)
    )
    initial_state = jnp.array(req.initial_state, dtype=jnp.float32)
    ts = _build_time_array(n_steps, req.dt)

    key = jax.random.PRNGKey(0)
    _, z_mean, _ = model.encode(initial_state, params, controls[0], key)
    z_traj = model.rollout_latent(
        ts, z_mean, controls, params, disturbances=disturbances, stochastic=False
    )

    import equinox as eqx
    decode_fn = jax.vmap(lambda z, u: model.decode(z, params, u), in_axes=(0, 0))
    pred_states = decode_fn(z_traj, controls)

    return PredictResponse(
        system=req.system,
        predicted_states=pred_states.tolist(),
        state_names=spec.state_names,
        latent_trajectory=z_traj.tolist() if req.return_latent else None,
        dt=req.dt,
        n_steps=n_steps,
    )


@app.post(
    "/ensemble",
    response_model=EnsembleResponse,
    tags=["inference"],
    dependencies=[Depends(_verify_api_key)],
)
async def ensemble(req: EnsembleRequest):
    """Run stochastic SDE ensemble rollout to estimate uncertainty.

    Returns mean, standard deviation, and 5th/95th percentile bands.
    """
    model, spec = _get_model_and_spec(req.system)

    n_steps = len(req.controls)
    if n_steps == 0:
        raise HTTPException(status_code=422, detail="controls must be non-empty.")

    controls = jnp.array(req.controls, dtype=jnp.float32)
    disturbances = (
        jnp.array(req.disturbances, dtype=jnp.float32)
        if req.disturbances
        else jnp.zeros((n_steps, spec.disturbance_dim), dtype=jnp.float32)
    )
    params = (
        jnp.array(req.params, dtype=jnp.float32)
        if req.params
        else jnp.ones(spec.param_dim, dtype=jnp.float32)
    )
    initial_state = jnp.array(req.initial_state, dtype=jnp.float32)
    ts = _build_time_array(n_steps, req.dt)

    base_key = jax.random.PRNGKey(0)
    enc_key, *sde_keys = jax.random.split(base_key, req.n_samples + 1)

    _, z_mean, _ = model.encode(initial_state, params, controls[0], enc_key)

    def _one_sample(sde_key):
        z_traj = model.rollout_latent(
            ts,
            z_mean,
            controls,
            params,
            disturbances=disturbances,
            key=sde_key,
            stochastic=True,
        )
        decode_fn = jax.vmap(lambda z, u: model.decode(z, params, u), in_axes=(0, 0))
        return decode_fn(z_traj, controls)

    sde_keys_arr = jnp.stack(sde_keys)
    all_samples = jax.vmap(_one_sample)(sde_keys_arr)  # (n_samples, T, state_dim)

    mean = jnp.mean(all_samples, axis=0)
    std = jnp.std(all_samples, axis=0)
    p05 = jnp.percentile(all_samples, 5, axis=0)
    p95 = jnp.percentile(all_samples, 95, axis=0)

    return EnsembleResponse(
        system=req.system,
        mean=mean.tolist(),
        std=std.tolist(),
        p05=p05.tolist(),
        p95=p95.tolist(),
        state_names=spec.state_names,
        n_samples=req.n_samples,
        dt=req.dt,
    )


@app.post(
    "/steady_state",
    response_model=SteadyStateResponse,
    tags=["simulation"],
    dependencies=[Depends(_verify_api_key)],
)
async def steady_state(req: SteadyStateRequest):
    """Find the steady-state operating point for a given system."""
    spec, simulator, _ = _get_demo_runtime(req.system)

    control = (
        jnp.array(req.nominal_control, dtype=jnp.float32)
        if req.nominal_control
        else jnp.array(spec.default_initial_state[:spec.control_dim], dtype=jnp.float32)
    )
    disturbance = (
        jnp.array(req.nominal_disturbance, dtype=jnp.float32)
        if req.nominal_disturbance
        else jnp.array(spec.default_nominal_disturbance, dtype=jnp.float32)
    )

    import numpy as np
    ss = simulator.steady_state(
        np.asarray(control),
        np.asarray(disturbance),
    )
    return SteadyStateResponse(
        system=req.system,
        steady_state=np.asarray(ss).tolist(),
        state_names=spec.state_names,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Start the DTE FastAPI service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--system_config", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--training_config", default=None)
    args = parser.parse_args()

    if args.system_config:
        os.environ["DTE_SYSTEM_CONFIG"] = args.system_config
    if args.model_path:
        os.environ["DTE_MODEL_PATH"] = args.model_path
    if args.training_config:
        os.environ["DTE_TRAINING_CONFIG"] = args.training_config

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
