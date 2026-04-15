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
- ``DTE_MODEL_PATH``     Path to trained single-system checkpoint fallback
- ``DTE_TRAINING_CONFIG`` Path to training YAML (default: configs/training_default.yaml)
- ``DTE_DEMO_CONFIG``    Demo config; when its runtime section points to a universal
  release checkpoint, the API will use that shared runtime first
- ``DTE_API_KEY``        Optional API key for authentication
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from dte.api.models import (
    DemoCatalogResponse,
    DemoCompareScenariosRequest,
    DemoCompareScenariosResponse,
    DemoOptimizeControlRequest,
    DemoOptimizeControlResponse,
    DemoPageResponse,
    DemoRolloutRequest,
    DemoRolloutResponse,
    DemoSimulateRequest,
    DemoSimulateResponse,
    EnsembleRequest,
    EnsembleResponse,
    ErrorResponse,
    HealthResponse,
    OnboardingCreateJobRequest,
    OnboardingJobReportResponse,
    OnboardingJobResponse,
    OnboardingPreviewRequest,
    OnboardingPreviewResponse,
    OnboardingTemplateListResponse,
    OnboardingWorkspaceResponse,
    OnboardingUploadResponse,
    PredictRequest,
    PredictResponse,
    SteadyStateRequest,
    SteadyStateResponse,
)
from dte.api.onboarding import (
    build_job_workspace,
    build_onboarding_templates,
    initialize_job_status,
    job_dir,
    load_completed_job_context,
    load_job_demo_runtime,
    load_job_report,
    load_job_status,
    load_preview_record,
    new_id,
    persist_upload,
    resolve_adaptation_runtime,
    run_onboarding_job,
    run_onboarding_preview,
    update_job_status,
)
from dte.demo.engine import (
    UniversalDemoRuntime,
    compare_scenarios,
    constraint_summary,
    demo_page_from_config,
    default_disturbance_sequence,
    demo_catalog_from_config,
    load_demo_config,
    load_demo_model_runtime,
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
_universal_runtime: UniversalDemoRuntime | None = None
_startup_time: float = 0.0


def _register_system(system_config_path: str):
    """Load a system spec and register it for API use."""
    with open(system_config_path) as f:
        sys_cfg = yaml.safe_load(f)
    spec = get_system_spec(sys_cfg)
    _specs[spec.name] = spec
    _system_configs[spec.name] = sys_cfg


def _load_single_system_model(
    system_config_path: str,
    model_path: Optional[str],
    training_config_path: str,
):
    """Load a legacy single-system checkpoint for one registered system."""

    with open(system_config_path) as f:
        sys_cfg = yaml.safe_load(f)
    with open(training_config_path) as f:
        train_cfg = yaml.safe_load(f)
    spec = get_system_spec(sys_cfg)

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
    global _startup_time, _universal_runtime
    _startup_time = time.time()
    _models.clear()
    _specs.clear()
    _system_configs.clear()
    _universal_runtime = None

    sys_config_env = os.environ.get("DTE_SYSTEM_CONFIG", "configs/cstr_default.yaml")
    model_path_env = os.environ.get("DTE_MODEL_PATH", "outputs/best_model.eqx")
    train_config_env = os.environ.get("DTE_TRAINING_CONFIG", "configs/training_default.yaml")
    demo_config_env = os.environ.get("DTE_DEMO_CONFIG", "configs/demo_app.yaml")
    disable_universal_runtime = os.environ.get("DTE_DISABLE_UNIVERSAL_RUNTIME", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }

    # Support comma-separated list for multiple systems
    system_paths: list[str] = []
    for sys_path in sys_config_env.split(","):
        sys_path = sys_path.strip()
        if sys_path and os.path.exists(sys_path):
            system_paths.append(sys_path)
            _register_system(sys_path)
        elif sys_path:
            print(f"[DTE API] System config not found: {sys_path}")

    if not disable_universal_runtime:
        try:
            demo_config = load_demo_config(demo_config_env)
            _universal_runtime = load_demo_model_runtime(
                demo_config,
                config_path=demo_config_env,
            )
            if _universal_runtime is not None:
                print(
                    "[DTE API] Loaded shared universal runtime from "
                    f"{_universal_runtime.model_path}"
                )
        except Exception as exc:
            print(f"[DTE API] Universal runtime load failed: {exc}")

    if _universal_runtime is None:
        for sys_path in system_paths:
            _load_single_system_model(sys_path, model_path_env, train_config_env)

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
# CORS — allow browser frontends to call this API
# ---------------------------------------------------------------------------

_cors_origins = os.environ.get("DTE_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler — always returns ErrorResponse shape
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    import traceback
    detail = str(exc) if str(exc) else type(exc).__name__
    return JSONResponse(
        status_code=500,
        content={"detail": detail, "code": "internal_error"},
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


def _get_inference_runtime_and_spec(system: str):
    spec = _specs.get(system)
    if spec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"System '{system}' not registered.  Available: {list(_specs.keys())}",
        )
    if _universal_runtime is not None and system in _universal_runtime.system_ids:
        return _universal_runtime, spec
    model = _models.get(system)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"No trained runtime loaded for system '{system}'. "
                "Configure DTE_DEMO_CONFIG for the universal release runtime or "
                "set DTE_MODEL_PATH for the legacy single-system path."
            ),
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
    if _universal_runtime is not None and system in _universal_runtime.system_ids:
        model = _universal_runtime
    else:
        model = _models.get(system)
    return spec, simulator, model


def _validate_state_vector(spec: SystemSpec, values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (spec.state_dim,):
        raise HTTPException(
            status_code=422,
            detail=f"initial_state must have shape [{spec.state_dim}] for '{spec.name}'.",
        )
    return arr


def _validate_control_sequence(spec: SystemSpec, controls: List[List[float]]) -> np.ndarray:
    arr = np.asarray(controls, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != spec.control_dim:
        raise HTTPException(
            status_code=422,
            detail=f"controls must have shape [T, {spec.control_dim}] for '{spec.name}'.",
        )
    return arr


def _validate_params(spec: SystemSpec, params: Optional[List[float]]) -> np.ndarray:
    if params is None:
        return np.ones(spec.param_dim, dtype=np.float32)
    arr = np.asarray(params, dtype=np.float32)
    if arr.shape != (spec.param_dim,):
        raise HTTPException(
            status_code=422,
            detail=f"params must have shape [{spec.param_dim}] for '{spec.name}'.",
        )
    return arr


def _start_onboarding_job(
    job_id: str,
    preview_record: dict,
    request_payload: dict,
    demo_config_path: str,
) -> None:
    def _runner():
        try:
            run_onboarding_job(
                job_id=job_id,
                preview_record=preview_record,
                request_payload=request_payload,
                demo_config_path=demo_config_path,
            )
        except Exception as exc:
            update_job_status(
                job_id,
                status="failed",
                stage="adaptation",
                progress_message="Customer adaptation failed.",
                error=str(exc),
            )

    threading.Thread(target=_runner, daemon=True).start()


def _prepare_universal_inputs(
    runtime: UniversalDemoRuntime,
    spec: SystemSpec,
    *,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    params: np.ndarray,
    dt: float,
) -> dict[str, jax.Array]:
    model = runtime.model
    system_id = runtime.system_ids.get(spec.name)
    if system_id is None:
        raise HTTPException(
            status_code=503,
            detail=f"Universal runtime does not contain system '{spec.name}'.",
        )

    system_id_arr = jnp.asarray(system_id, dtype=jnp.int32)
    n_steps = int(controls.shape[0])
    padded_state = jnp.zeros((model.max_state_dim,), dtype=jnp.float32).at[: spec.state_dim].set(
        jnp.asarray(initial_state, dtype=jnp.float32)
    )
    padded_controls = jnp.zeros((n_steps, model.max_control_dim), dtype=jnp.float32).at[
        :, : spec.control_dim
    ].set(jnp.asarray(controls, dtype=jnp.float32))
    padded_disturbances = jnp.zeros(
        (n_steps, model.max_disturbance_dim),
        dtype=jnp.float32,
    ).at[:, : spec.disturbance_dim].set(jnp.asarray(disturbances, dtype=jnp.float32))
    padded_params = jnp.zeros((model.max_param_dim,), dtype=jnp.float32).at[: spec.param_dim].set(
        jnp.asarray(params, dtype=jnp.float32)
    )

    state_mask = model.state_mask_table[system_id_arr]
    control_mask = model.control_mask_table[system_id_arr]
    disturbance_mask = model.disturbance_mask_table[system_id_arr]
    param_mask = model.param_mask_table[system_id_arr]
    ts = _build_time_array(n_steps, dt)

    controls_norm = model.normalize_controls(padded_controls, system_id_arr) * control_mask
    disturbances_norm = (
        model.normalize_disturbances(padded_disturbances, system_id_arr) * disturbance_mask
    )
    params_scaled = model.scale_params(padded_params, system_id_arr) * param_mask
    state_norm = model.normalize_states(padded_state, system_id_arr) * state_mask
    return {
        "system_id": system_id_arr,
        "ts": ts,
        "state_mask": state_mask,
        "control_mask": control_mask,
        "disturbance_mask": disturbance_mask,
        "param_mask": param_mask,
        "controls_norm": controls_norm,
        "disturbances_norm": disturbances_norm,
        "params_scaled": params_scaled,
        "state_norm": state_norm,
    }


def _predict_with_universal_runtime(
    runtime: UniversalDemoRuntime,
    spec: SystemSpec,
    *,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    params: np.ndarray,
    dt: float,
    return_latent: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    model = runtime.model
    context = _prepare_universal_inputs(
        runtime,
        spec,
        initial_state=initial_state,
        controls=controls,
        disturbances=disturbances,
        params=params,
        dt=dt,
    )
    z_mean = model.encode(
        context["state_norm"],
        context["params_scaled"],
        context["controls_norm"][0],
        context["state_mask"],
        context["control_mask"],
        context["param_mask"],
        context["system_id"],
        None,
    )[1]
    z_traj = model.rollout_latent(
        context["ts"],
        z_mean,
        context["controls_norm"],
        context["disturbances_norm"],
        context["params_scaled"],
        context["control_mask"],
        context["disturbance_mask"],
        context["param_mask"],
        context["system_id"],
    )
    pred_norm = jax.vmap(
        lambda z_t, control_t: model.decode(
            z_t,
            context["params_scaled"],
            control_t,
            context["state_mask"],
            context["control_mask"],
            context["param_mask"],
            context["system_id"],
        )
    )(z_traj, context["controls_norm"])
    pred_states = np.asarray(
        model.denormalize_states(pred_norm, context["system_id"])[:, : spec.state_dim]
    )
    latent = np.asarray(z_traj) if return_latent else None
    return pred_states, latent


def _ensemble_with_universal_runtime(
    runtime: UniversalDemoRuntime,
    spec: SystemSpec,
    *,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    params: np.ndarray,
    dt: float,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = runtime.model
    context = _prepare_universal_inputs(
        runtime,
        spec,
        initial_state=initial_state,
        controls=controls,
        disturbances=disturbances,
        params=params,
        dt=dt,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), int(n_samples))

    def _one_sample(sample_key):
        z0 = model.encode(
            context["state_norm"],
            context["params_scaled"],
            context["controls_norm"][0],
            context["state_mask"],
            context["control_mask"],
            context["param_mask"],
            context["system_id"],
            sample_key,
        )[0]
        z_traj = model.rollout_latent(
            context["ts"],
            z0,
            context["controls_norm"],
            context["disturbances_norm"],
            context["params_scaled"],
            context["control_mask"],
            context["disturbance_mask"],
            context["param_mask"],
            context["system_id"],
        )
        pred_norm = jax.vmap(
            lambda z_t, control_t: model.decode(
                z_t,
                context["params_scaled"],
                control_t,
                context["state_mask"],
                context["control_mask"],
                context["param_mask"],
                context["system_id"],
            )
        )(z_traj, context["controls_norm"])
        return model.denormalize_states(pred_norm, context["system_id"])[:, : spec.state_dim]

    all_samples = np.asarray(jax.vmap(_one_sample)(keys))
    mean = np.mean(all_samples, axis=0)
    std = np.std(all_samples, axis=0)
    p05 = np.percentile(all_samples, 5.0, axis=0)
    p95 = np.percentile(all_samples, 95.0, axis=0)
    return mean, std, p05, p95


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


def _validate_active_control_names(
    spec: SystemSpec,
    active_control_names: Optional[List[str]],
) -> list[str] | None:
    if not active_control_names:
        return None
    invalid = sorted(set(active_control_names) - set(spec.control_names))
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=(
                f"active_control_names must be drawn from {spec.control_names}; "
                f"got {invalid}."
            ),
        )
    return [name for name in active_control_names if name in spec.control_names]


def _build_compare_response(
    *,
    system: str,
    spec: SystemSpec,
    result: dict[str, object],
) -> DemoCompareScenariosResponse:
    baseline = result["baseline"]
    candidate = result["candidate"]
    return DemoCompareScenariosResponse(
        system=system,
        times=np.asarray(candidate["times"]).tolist(),
        baseline_source=str(baseline["source"]),
        candidate_source=str(candidate["source"]),
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


def _build_optimize_response(
    *,
    system: str,
    spec: SystemSpec,
    result: dict[str, object],
) -> DemoOptimizeControlResponse:
    return DemoOptimizeControlResponse(
        system=system,
        control_sequence=np.asarray(result["control_sequence"]).tolist(),
        predicted_states=np.asarray(result["predicted_states"]).tolist(),
        objective=float(result["objective"]),
        tracked_state_names=list(result["tracked_state_names"]),
        state_names=spec.state_names,
        constraint_summary={
            key: float(value) for key, value in result["constraint_summary"].items()
        },
    )


def _run_demo_compare(
    *,
    system: str,
    spec: SystemSpec,
    simulator,
    model,
    req: DemoCompareScenariosRequest,
) -> DemoCompareScenariosResponse:
    baseline_controls = _validate_control_sequence(spec, req.baseline_controls)
    candidate_controls = _validate_control_sequence(spec, req.candidate_controls)
    if baseline_controls.shape != candidate_controls.shape:
        raise HTTPException(
            status_code=422,
            detail="baseline_controls and candidate_controls must have the same shape.",
        )
    disturbances = _ensure_disturbance_sequence(
        spec,
        req.disturbances,
        baseline_controls.shape[0],
    )
    result = compare_scenarios(
        spec,
        simulator,
        initial_state=_validate_state_vector(spec, req.initial_state),
        baseline_controls=baseline_controls,
        candidate_controls=candidate_controls,
        disturbances=disturbances,
        dt=float(req.dt),
        model=model,
        params=_validate_params(spec, req.params) if req.params is not None else None,
        n_samples=int(req.n_samples),
        seed=int(req.seed),
    )
    return _build_compare_response(system=system, spec=spec, result=result)


def _run_demo_optimize(
    *,
    system: str,
    spec: SystemSpec,
    simulator,
    req: DemoOptimizeControlRequest,
) -> DemoOptimizeControlResponse:
    disturbances = np.asarray(req.disturbances, dtype=np.float32)
    if disturbances.ndim != 2 or disturbances.shape[1] != spec.disturbance_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                f"disturbances must have shape [T, {spec.disturbance_dim}] for '{system}'."
            ),
        )
    reference_controls = None
    if req.reference_controls is not None:
        reference_controls = _validate_control_sequence(spec, req.reference_controls)
        if reference_controls.shape[0] != disturbances.shape[0]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"reference_controls must have shape [T, {spec.control_dim}] with the same "
                    f"T as disturbances for '{system}'."
                ),
            )
    result = optimize_control_sequence(
        spec,
        simulator,
        initial_state=_validate_state_vector(spec, req.initial_state),
        disturbances=disturbances,
        reference_controls=reference_controls,
        active_control_names=_validate_active_control_names(spec, req.active_control_names),
        dt=float(req.dt),
        target_state=_validate_state_vector(spec, req.target_state),
        tracked_state_names=req.tracked_state_names,
        n_candidates=int(req.n_candidates),
        seed=int(req.seed),
    )
    return _build_optimize_response(system=system, spec=spec, result=result)


def _get_onboarding_job_demo_runtime(job_id: str):
    job_payload, preview_record, _summary = load_completed_job_context(job_id)
    template_system_name = str(preview_record["template_system_name"])
    system_config = _system_configs.get(template_system_name)
    if system_config is None:
        with open(preview_record["template_config_path"], "r", encoding="utf-8") as handle:
            system_config = yaml.safe_load(handle) or {}
    spec = get_system_spec(system_config)
    simulator = get_simulator(template_system_name, system_config)
    model = load_job_demo_runtime(job_id)
    return job_payload, preview_record, spec, simulator, model


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Return service health, loaded systems, and uptime."""
    uptime_s = round(time.time() - _startup_time, 1) if _startup_time else None
    models_ready = bool(_models or _universal_runtime)
    return HealthResponse(
        status="ok" if models_ready else "degraded",
        loaded_systems=list(_specs.keys()),
        version="0.1.0",
        uptime_seconds=uptime_s,
        models_ready=models_ready,
    )


@app.get(
    "/onboarding/templates",
    response_model=OnboardingTemplateListResponse,
    tags=["onboarding"],
    dependencies=[Depends(_verify_api_key)],
)
async def onboarding_templates():
    """Return supported unit templates for the customer onboarding wizard."""

    demo_config_path = os.environ.get("DTE_DEMO_CONFIG", "configs/demo_app.yaml")
    config = load_demo_config(demo_config_path)
    templates = build_onboarding_templates(_system_configs, demo_config=config)
    return OnboardingTemplateListResponse.model_validate({"templates": templates})


@app.post(
    "/onboarding/uploads",
    response_model=OnboardingUploadResponse,
    tags=["onboarding"],
    dependencies=[Depends(_verify_api_key)],
)
async def onboarding_upload(file: UploadFile = File(...)):
    """Persist one uploaded CSV/Parquet file and return detected columns."""

    if not file.filename:
        raise HTTPException(status_code=422, detail="Upload must include a filename.")
    try:
        payload = persist_upload(file.filename, await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return OnboardingUploadResponse.model_validate(payload)


@app.post(
    "/onboarding/preview",
    response_model=OnboardingPreviewResponse,
    tags=["onboarding"],
    dependencies=[Depends(_verify_api_key)],
)
async def onboarding_preview(req: OnboardingPreviewRequest):
    """Validate onboarding selections and run a preview ingestion step."""

    try:
        payload = run_onboarding_preview(
            req.model_dump(),
            system_configs=_system_configs,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return OnboardingPreviewResponse.model_validate(payload)


@app.post(
    "/onboarding/jobs",
    response_model=OnboardingJobResponse,
    tags=["onboarding"],
    dependencies=[Depends(_verify_api_key)],
)
async def onboarding_create_job(req: OnboardingCreateJobRequest):
    """Launch a durable customer adaptation job from a validated preview."""

    preview_record = load_preview_record(req.preview_id)
    if preview_record is None:
        raise HTTPException(status_code=404, detail=f"Preview '{req.preview_id}' was not found.")
    if not preview_record.get("valid", False):
        raise HTTPException(
            status_code=409,
            detail=f"Preview '{req.preview_id}' is not valid for adaptation.",
        )

    demo_config_path = os.environ.get("DTE_DEMO_CONFIG", "configs/demo_app.yaml")
    model_path, config_path = resolve_adaptation_runtime(
        requested_model_path=req.model_path,
        requested_config_path=req.config_path,
        demo_config_path=demo_config_path,
    )
    if model_path is None or not model_path.exists():
        raise HTTPException(
            status_code=422,
            detail="Could not resolve a valid universal model_path for customer adaptation.",
        )
    if config_path is None or not config_path.exists():
        raise HTTPException(
            status_code=422,
            detail="Could not resolve a valid universal config_path for customer adaptation.",
        )

    job_id = new_id("job")
    job_directory = job_dir(job_id)
    preview_directory = Path(preview_record["artifacts"]["processed_data_dir"]).parent
    job_artifacts = {
        "onboarding_json": preview_record["artifacts"]["onboarding_json"],
        "preview_summary": str((preview_directory / "summary.json").resolve()),
        "uploaded_file": preview_record["artifacts"]["uploaded_file"],
        "summary_json": str((job_directory / "summary.json").resolve()),
        "report_json": str((job_directory / "validation_report.json").resolve()),
        "report_markdown": str((job_directory / "validation_report.md").resolve()),
        "log_path": str((job_directory / "logs" / "adapt_customer.log").resolve()),
    }

    status_payload = initialize_job_status(
        job_id=job_id,
        preview_id=req.preview_id,
        artifacts=job_artifacts,
    )
    _start_onboarding_job(
        job_id,
        preview_record,
        req.model_dump(),
        demo_config_path,
    )
    return OnboardingJobResponse.model_validate(status_payload)


@app.get(
    "/onboarding/jobs/{job_id}",
    response_model=OnboardingJobResponse,
    tags=["onboarding"],
    dependencies=[Depends(_verify_api_key)],
)
async def onboarding_job(job_id: str):
    """Return status for one customer adaptation job."""

    payload = load_job_status(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' was not found.")
    return OnboardingJobResponse.model_validate(payload)


@app.get(
    "/onboarding/jobs/{job_id}/report",
    response_model=OnboardingJobReportResponse,
    tags=["onboarding"],
    dependencies=[Depends(_verify_api_key)],
)
async def onboarding_job_report(job_id: str):
    """Return the final validation report for a completed onboarding job."""

    job_payload = load_job_status(job_id)
    if job_payload is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' was not found.")
    if job_payload.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not completed yet.",
        )
    try:
        summary, report_markdown = load_job_report(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return OnboardingJobReportResponse.model_validate(
        {
            "job": job_payload,
            "summary": summary,
            "report_markdown": report_markdown,
        }
    )


@app.get(
    "/onboarding/jobs/{job_id}/workspace",
    response_model=OnboardingWorkspaceResponse,
    tags=["onboarding"],
    dependencies=[Depends(_verify_api_key)],
)
async def onboarding_job_workspace(job_id: str):
    """Return the customer planning workspace bootstrap for one completed job."""

    demo_config_path = os.environ.get("DTE_DEMO_CONFIG", "configs/demo_app.yaml")
    try:
        payload = build_job_workspace(
            job_id,
            demo_config_path=demo_config_path,
            system_configs=_system_configs,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return OnboardingWorkspaceResponse.model_validate(payload)


@app.post(
    "/onboarding/jobs/{job_id}/compare_scenarios",
    response_model=DemoCompareScenariosResponse,
    tags=["onboarding"],
    dependencies=[Depends(_verify_api_key)],
)
async def onboarding_job_compare_scenarios(job_id: str, req: DemoCompareScenariosRequest):
    """Compare customer scenarios against the adapted job-specific runtime."""

    try:
        _job_payload, preview_record, spec, simulator, model = _get_onboarding_job_demo_runtime(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    expected_system = str(preview_record["template_system_name"])
    if req.system != expected_system:
        raise HTTPException(
            status_code=422,
            detail=(
                f"system must match the job template system '{expected_system}' "
                f"for onboarding job '{job_id}'."
            ),
        )
    return _run_demo_compare(
        system=req.system,
        spec=spec,
        simulator=simulator,
        model=model,
        req=req,
    )


@app.post(
    "/onboarding/jobs/{job_id}/optimize_control",
    response_model=DemoOptimizeControlResponse,
    tags=["onboarding"],
    dependencies=[Depends(_verify_api_key)],
)
async def onboarding_job_optimize_control(job_id: str, req: DemoOptimizeControlRequest):
    """Recommend a customer control schedule using the adapted workspace context."""

    try:
        _job_payload, preview_record, spec, simulator, _model = _get_onboarding_job_demo_runtime(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    expected_system = str(preview_record["template_system_name"])
    if req.system != expected_system:
        raise HTTPException(
            status_code=422,
            detail=(
                f"system must match the job template system '{expected_system}' "
                f"for onboarding job '{job_id}'."
            ),
        )
    return _run_demo_optimize(
        system=req.system,
        spec=spec,
        simulator=simulator,
        req=req,
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


@app.get(
    "/demo/page",
    response_model=DemoPageResponse,
    tags=["demo"],
    dependencies=[Depends(_verify_api_key)],
)
async def demo_page():
    """Return the full browser bootstrap payload for the marketing demo frontend."""

    demo_config_path = os.environ.get("DTE_DEMO_CONFIG", "configs/demo_app.yaml")
    config = load_demo_config(demo_config_path)
    payload = demo_page_from_config(
        config,
        _system_configs,
        config_path=demo_config_path,
        runtime_loaded=_universal_runtime is not None,
    )
    return DemoPageResponse.model_validate(payload)


@app.post(
    "/demo/simulate",
    response_model=DemoSimulateResponse,
    tags=["demo"],
    dependencies=[Depends(_verify_api_key)],
)
async def demo_simulate(req: DemoSimulateRequest):
    """Run a deterministic simulator rollout for the demo app."""

    spec, simulator, _ = _get_demo_runtime(req.system)
    controls = _validate_control_sequence(spec, req.controls)
    disturbances = _ensure_disturbance_sequence(spec, req.disturbances, controls.shape[0])
    states = simulate_open_loop(
        spec,
        simulator,
        _validate_state_vector(spec, req.initial_state),
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
    controls = _validate_control_sequence(spec, req.controls)
    disturbances = _ensure_disturbance_sequence(spec, req.disturbances, controls.shape[0])
    result = rollout_scenario(
        spec,
        simulator,
        initial_state=_validate_state_vector(spec, req.initial_state),
        controls=controls,
        disturbances=disturbances,
        dt=float(req.dt),
        model=model,
        params=_validate_params(spec, req.params) if req.params is not None else None,
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
    return _run_demo_optimize(
        system=req.system,
        spec=spec,
        simulator=simulator,
        req=req,
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
    return _run_demo_compare(
        system=req.system,
        spec=spec,
        simulator=simulator,
        model=model,
        req=req,
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
    runtime, spec = _get_inference_runtime_and_spec(req.system)

    controls = _validate_control_sequence(spec, req.controls)
    disturbances = _ensure_disturbance_sequence(spec, req.disturbances, controls.shape[0])
    params = _validate_params(spec, req.params)
    initial_state = _validate_state_vector(spec, req.initial_state)

    if isinstance(runtime, UniversalDemoRuntime):
        pred_states, latent = _predict_with_universal_runtime(
            runtime,
            spec,
            initial_state=initial_state,
            controls=controls,
            disturbances=disturbances,
            params=params,
            dt=float(req.dt),
            return_latent=bool(req.return_latent),
        )
    else:
        model = runtime
        ts = _build_time_array(controls.shape[0], req.dt)
        _, z_mean, _ = model.encode(
            jnp.asarray(initial_state, dtype=jnp.float32),
            jnp.asarray(params, dtype=jnp.float32),
            jnp.asarray(controls[0], dtype=jnp.float32),
            jax.random.PRNGKey(0),
        )
        z_traj = model.rollout_latent(
            ts,
            z_mean,
            jnp.asarray(controls, dtype=jnp.float32),
            jnp.asarray(params, dtype=jnp.float32),
            disturbances=jnp.asarray(disturbances, dtype=jnp.float32),
            stochastic=False,
        )
        decode_fn = jax.vmap(lambda z, u: model.decode(z, jnp.asarray(params, dtype=jnp.float32), u), in_axes=(0, 0))
        pred_states = np.asarray(decode_fn(z_traj, jnp.asarray(controls, dtype=jnp.float32)))
        latent = np.asarray(z_traj) if req.return_latent else None

    return PredictResponse(
        system=req.system,
        predicted_states=pred_states.tolist(),
        state_names=spec.state_names,
        latent_trajectory=latent.tolist() if latent is not None else None,
        dt=req.dt,
        n_steps=int(controls.shape[0]),
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
    runtime, spec = _get_inference_runtime_and_spec(req.system)
    controls = _validate_control_sequence(spec, req.controls)
    disturbances = _ensure_disturbance_sequence(spec, req.disturbances, controls.shape[0])
    params = _validate_params(spec, req.params)
    initial_state = _validate_state_vector(spec, req.initial_state)

    if isinstance(runtime, UniversalDemoRuntime):
        mean, std, p05, p95 = _ensemble_with_universal_runtime(
            runtime,
            spec,
            initial_state=initial_state,
            controls=controls,
            disturbances=disturbances,
            params=params,
            dt=float(req.dt),
            n_samples=int(req.n_samples),
        )
    else:
        model = runtime
        ts = _build_time_array(controls.shape[0], req.dt)
        base_key = jax.random.PRNGKey(0)
        enc_key, *sde_keys = jax.random.split(base_key, req.n_samples + 1)
        _, z_mean, _ = model.encode(
            jnp.asarray(initial_state, dtype=jnp.float32),
            jnp.asarray(params, dtype=jnp.float32),
            jnp.asarray(controls[0], dtype=jnp.float32),
            enc_key,
        )

        def _one_sample(sde_key):
            z_traj = model.rollout_latent(
                ts,
                z_mean,
                jnp.asarray(controls, dtype=jnp.float32),
                jnp.asarray(params, dtype=jnp.float32),
                disturbances=jnp.asarray(disturbances, dtype=jnp.float32),
                key=sde_key,
                stochastic=True,
            )
            decode_fn = jax.vmap(
                lambda z, u: model.decode(z, jnp.asarray(params, dtype=jnp.float32), u),
                in_axes=(0, 0),
            )
            return decode_fn(z_traj, jnp.asarray(controls, dtype=jnp.float32))

        all_samples = np.asarray(jax.vmap(_one_sample)(jnp.stack(sde_keys)))
        mean = np.mean(all_samples, axis=0)
        std = np.std(all_samples, axis=0)
        p05 = np.percentile(all_samples, 5.0, axis=0)
        p95 = np.percentile(all_samples, 95.0, axis=0)

    return EnsembleResponse(
        system=req.system,
        mean=np.asarray(mean).tolist(),
        std=np.asarray(std).tolist(),
        p05=np.asarray(p05).tolist(),
        p95=np.asarray(p95).tolist(),
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
        else jnp.asarray(
            [
                0.5 * sum(spec.control_ranges[name])
                for name in spec.control_names
            ],
            dtype=jnp.float32,
        )
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
