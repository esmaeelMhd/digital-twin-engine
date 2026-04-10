"""Pydantic request/response models for the Digital Twin Engine REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Common primitives
# ---------------------------------------------------------------------------


class StateVector(BaseModel):
    """A named state vector."""

    values: List[float] = Field(..., description="State variable values.")
    names: Optional[List[str]] = Field(
        None, description="Optional variable names (must match system spec)."
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint.

    Runs a deterministic (mean-trajectory) rollout of the digital twin from an
    initial state under a given control and disturbance sequence.
    """

    system: str = Field("cstr", description="Registered system name (e.g. 'cstr', 'heat_exchanger', 'two_tank').")
    initial_state: List[float] = Field(
        ..., description="Initial physical state vector (length = state_dim)."
    )
    controls: List[List[float]] = Field(
        ...,
        description=(
            "Control sequence, shape [T, control_dim].  "
            "T is the prediction horizon."
        ),
    )
    disturbances: Optional[List[List[float]]] = Field(
        None,
        description=(
            "Disturbance sequence, shape [T, disturbance_dim].  "
            "Zeros if omitted."
        ),
    )
    params: Optional[List[float]] = Field(
        None,
        description="Parameter vector (length = param_dim).  Ones if omitted.",
    )
    dt: float = Field(0.1, description="Sampling interval in seconds.", gt=0)
    return_latent: bool = Field(False, description="If True, also return the latent trajectory.")


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""

    system: str
    predicted_states: List[List[float]] = Field(
        ..., description="Predicted state trajectory, shape [T, state_dim]."
    )
    state_names: List[str]
    latent_trajectory: Optional[List[List[float]]] = Field(
        None, description="Latent trajectory (only when return_latent=True)."
    )
    dt: float
    n_steps: int


# ---------------------------------------------------------------------------
# Ensemble prediction (uncertainty quantification)
# ---------------------------------------------------------------------------


class EnsembleRequest(BaseModel):
    """Request body for the /ensemble endpoint.

    Runs N stochastic SDE samples to estimate prediction uncertainty.
    """

    system: str = Field("cstr")
    initial_state: List[float]
    controls: List[List[float]]
    disturbances: Optional[List[List[float]]] = None
    params: Optional[List[float]] = None
    dt: float = Field(0.1, gt=0)
    n_samples: int = Field(50, ge=1, le=1000)


class EnsembleResponse(BaseModel):
    """Response body for the /ensemble endpoint."""

    system: str
    mean: List[List[float]] = Field(..., description="Mean prediction, shape [T, state_dim].")
    std: List[List[float]] = Field(..., description="Standard deviation, shape [T, state_dim].")
    p05: List[List[float]] = Field(..., description="5th percentile, shape [T, state_dim].")
    p95: List[List[float]] = Field(..., description="95th percentile, shape [T, state_dim].")
    state_names: List[str]
    n_samples: int
    dt: float


# ---------------------------------------------------------------------------
# Steady-state
# ---------------------------------------------------------------------------


class SteadyStateRequest(BaseModel):
    """Request body for /steady_state."""

    system: str = Field("cstr")
    nominal_control: Optional[List[float]] = None
    nominal_disturbance: Optional[List[float]] = None
    params: Optional[List[float]] = None


class SteadyStateResponse(BaseModel):
    """Response body for /steady_state."""

    system: str
    steady_state: List[float]
    state_names: List[str]


# ---------------------------------------------------------------------------
# Demo App
# ---------------------------------------------------------------------------


class DemoCatalogItem(BaseModel):
    """One available interactive demo."""

    id: str
    title: str
    system: str
    kind: str
    description: str
    controls: List[str]
    disturbances: List[str]
    states: List[str]
    dt: float
    n_steps: int
    highlight_states: List[str]


class DemoFlowsheetItem(BaseModel):
    """Small flowsheet preview for the demo app."""

    id: str
    title: str
    description: Optional[str] = None
    units: List[Dict[str, Any]]
    streams: List[Dict[str, Any]]


class DemoCatalogResponse(BaseModel):
    """Catalog payload for the Phase 6 demo site."""

    product_name: str
    headline: str
    summary: str
    demos: List[DemoCatalogItem]
    flowsheets: List[DemoFlowsheetItem]


class DemoSimulateRequest(BaseModel):
    """Request for deterministic simulator rollout."""

    system: str
    initial_state: List[float]
    controls: List[List[float]]
    disturbances: Optional[List[List[float]]] = None
    dt: float = Field(0.1, gt=0)


class DemoSimulateResponse(BaseModel):
    """Simulator rollout response for the demo app."""

    system: str
    times: List[float]
    states: List[List[float]]
    state_names: List[str]
    constraint_summary: Dict[str, float]


class DemoRolloutRequest(BaseModel):
    """Request for mean trajectory plus uncertainty bands."""

    system: str
    initial_state: List[float]
    controls: List[List[float]]
    disturbances: Optional[List[List[float]]] = None
    params: Optional[List[float]] = None
    dt: float = Field(0.1, gt=0)
    n_samples: int = Field(16, ge=2, le=128)
    seed: int = 0


class DemoRolloutResponse(BaseModel):
    """Trajectory mean and uncertainty response for the demo app."""

    system: str
    source: str
    times: List[float]
    mean: List[List[float]]
    std: List[List[float]]
    p05: List[List[float]]
    p95: List[List[float]]
    state_names: List[str]
    constraint_summary: Dict[str, float]


class DemoOptimizeControlRequest(BaseModel):
    """Request for a lightweight demo control recommendation."""

    system: str
    initial_state: List[float]
    disturbances: List[List[float]]
    target_state: List[float]
    tracked_state_names: Optional[List[str]] = None
    dt: float = Field(0.1, gt=0)
    n_candidates: int = Field(48, ge=1, le=512)
    seed: int = 0


class DemoOptimizeControlResponse(BaseModel):
    """Recommended control schedule for the demo app."""

    system: str
    control_sequence: List[List[float]]
    predicted_states: List[List[float]]
    objective: float
    tracked_state_names: List[str]
    state_names: List[str]
    constraint_summary: Dict[str, float]


class DemoCompareScenariosRequest(BaseModel):
    """Request to compare baseline and candidate scenarios."""

    system: str
    initial_state: List[float]
    baseline_controls: List[List[float]]
    candidate_controls: List[List[float]]
    disturbances: Optional[List[List[float]]] = None
    params: Optional[List[float]] = None
    dt: float = Field(0.1, gt=0)
    n_samples: int = Field(16, ge=2, le=128)
    seed: int = 0


class DemoCompareScenariosResponse(BaseModel):
    """Comparison payload used by the demo UI."""

    system: str
    state_names: List[str]
    baseline_mean: List[List[float]]
    candidate_mean: List[List[float]]
    baseline_p05: List[List[float]]
    baseline_p95: List[List[float]]
    candidate_p05: List[List[float]]
    candidate_p95: List[List[float]]
    summary: Dict[str, Any]
    baseline_constraints: Dict[str, float]
    candidate_constraints: Dict[str, float]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for /health."""

    status: str = "ok"
    loaded_systems: List[str] = Field(default_factory=list)
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
    code: Optional[str] = None
