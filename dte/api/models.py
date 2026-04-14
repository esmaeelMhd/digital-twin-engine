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

    system: str = Field("cstr", description="Registered system name (e.g. 'cstr', 'heat_exchanger', 'two_tank').")
    initial_state: List[float] = Field(..., description="Initial physical state vector (length = state_dim).")
    controls: List[List[float]] = Field(..., description="Control sequence, shape [T, control_dim].")
    disturbances: Optional[List[List[float]]] = Field(None, description="Disturbance sequence, shape [T, disturbance_dim]. Zeros if omitted.")
    params: Optional[List[float]] = Field(None, description="Parameter vector (length = param_dim). Ones if omitted.")
    dt: float = Field(0.1, gt=0, description="Sampling interval in seconds.")
    n_samples: int = Field(50, ge=1, le=1000, description="Number of stochastic SDE samples for uncertainty estimation.")


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


class DemoProfile(BaseModel):
    """Serializable control or disturbance profile from demo config."""

    type: str = Field("constant")
    channels: Optional[Dict[str, float]] = None
    values: Optional[Dict[str, float]] = None
    start: Optional[Dict[str, float]] = None
    end: Optional[Dict[str, float]] = None
    base: Optional[Dict[str, float]] = None
    pulse: Optional[Dict[str, float]] = None
    start_step: Optional[int] = None
    duration: Optional[int] = None


class DemoPreset(BaseModel):
    """Named preset shown in the interactive demo controls."""

    id: str
    title: str
    description: str = ""
    profile: Optional[DemoProfile] = None


class DemoOptimizationConfig(BaseModel):
    """Optimization defaults for one demo workspace."""

    n_candidates: int = Field(48, ge=1)
    seed: int = 0


class DemoChannelSpec(BaseModel):
    """Frontend-friendly channel metadata."""

    name: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None


class DemoSystemSpec(BaseModel):
    """Subset of system metadata required by the browser demo."""

    name: str
    state_dim: int
    control_dim: int
    disturbance_dim: int
    param_dim: int
    state_names: List[str]
    control_names: List[str]
    disturbance_names: List[str]
    default_initial_state: List[float]
    default_nominal_disturbance: List[float]
    control_ranges: Dict[str, List[float]]
    disturbance_ranges: Dict[str, List[float]]
    state_channels: List[DemoChannelSpec]
    control_channels: List[DemoChannelSpec]
    disturbance_channels: List[DemoChannelSpec]


class DemoDefinition(BaseModel):
    """Full interactive demo definition for the browser frontend."""

    id: str
    title: str
    system: str
    kind: str
    description: str
    operator_goal: Optional[str] = None
    dt: float
    n_steps: int
    highlight_states: List[str]
    target_state: Dict[str, float]
    initial_state: Dict[str, float]
    baseline_control_profile: Optional[DemoProfile] = None
    disturbance_presets: List[DemoPreset]
    candidate_profiles: List[DemoPreset]
    optimization: DemoOptimizationConfig = Field(default_factory=DemoOptimizationConfig)
    run_button_label: str = "Run Scenario"
    optimize_button_label: str = "Recommend Control Sequence"
    system_spec: DemoSystemSpec


class DemoReleaseSnapshot(BaseModel):
    """Release summary and case-study metadata for the demo site."""

    release_label: str
    model_available: bool
    config_available: bool
    runtime_samples: int
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    runtime_loaded: bool = False
    train_best_val_loss: Optional[float] = None
    eval_metric_name: Optional[str] = None
    eval_metric_value: Optional[float] = None
    per_system_total_loss: Dict[str, float]
    milestone_status: Optional[str] = None
    customer_status: Optional[str] = None
    customer_best_unit_template: Optional[str] = None
    customer_best_val_loss: Optional[float] = None
    customer_forecast_rmse: Optional[float] = None
    customer_rollout_rmse: Optional[float] = None
    customer_report_path: Optional[str] = None
    customer_report_exists: bool = False
    customer_report_markdown: Optional[str] = None


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


class DemoPageResponse(BaseModel):
    """Bootstrap payload for the browser-based demo frontend."""

    product_name: str
    headline: str
    summary: str
    release: DemoReleaseSnapshot
    demos: List[DemoDefinition]
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
    reference_controls: Optional[List[List[float]]] = None
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
    times: List[float]
    baseline_source: str
    candidate_source: str
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
# Customer onboarding
# ---------------------------------------------------------------------------


class OnboardingTemplate(BaseModel):
    """One supported onboarding template for a customer pilot."""

    id: str
    title: str
    description: str
    system_spec: DemoSystemSpec
    suggested_objectives: List[str] = Field(default_factory=list)
    suggested_controls: List[str] = Field(default_factory=list)


class OnboardingTemplateListResponse(BaseModel):
    """Supported onboarding templates."""

    templates: List[OnboardingTemplate]


class OnboardingUploadResponse(BaseModel):
    """Metadata returned after persisting an uploaded data file."""

    upload_id: str
    filename: str
    detected_format: str
    columns: List[str]
    row_count: int
    size_bytes: int


class OnboardingPreviewRequest(BaseModel):
    """Preview ingestion and onboarding validation request."""

    upload_id: str
    template_id: str
    customer_name: str = Field(..., min_length=1)
    timestamp_column: Optional[str] = None
    dt: float = Field(0.1, gt=0)
    trajectory_duration: float = Field(100.0, gt=0)
    trajectory_stride: float = Field(10.0, gt=0)
    max_gap_fill: float = Field(10.0, gt=0)
    outlier_sigma: float = Field(5.0, gt=0)
    drop_large_gaps: bool = False
    state_column_map: Dict[str, str]
    control_column_map: Dict[str, str]
    disturbance_column_map: Dict[str, str] = Field(default_factory=dict)
    objective_state_names: List[str] = Field(default_factory=list)
    control_variable_names: List[str] = Field(default_factory=list)


class OnboardingPreviewResponse(BaseModel):
    """Preview result for onboarding ingestion and validation."""

    preview_id: Optional[str] = None
    upload_id: str
    template_id: str
    valid: bool
    blocking_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    ingestion_summary: Optional[Dict[str, Any]] = None
    onboarding_spec: Optional[Dict[str, Any]] = None
    objective_state_names: List[str] = Field(default_factory=list)
    control_variable_names: List[str] = Field(default_factory=list)


class OnboardingCreateJobRequest(BaseModel):
    """Launch an asynchronous customer adaptation job from a preview."""

    preview_id: str
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    trainable_mode: str = Field("adapters")
    tune_normalization: bool = True
    tune_physics_params: bool = False
    param_indices: List[int] = Field(default_factory=list)
    seed: int = 42
    time_budget_minutes: Optional[float] = Field(None, gt=0)

    @model_validator(mode="after")
    def _validate_trainable_mode(self) -> "OnboardingCreateJobRequest":
        if self.trainable_mode not in {"adapters", "full"}:
            raise ValueError("trainable_mode must be one of: adapters, full.")
        return self


class OnboardingJobArtifacts(BaseModel):
    """Known filesystem artifacts for a customer onboarding job."""

    summary_json: Optional[str] = None
    report_markdown: Optional[str] = None
    report_json: Optional[str] = None
    onboarding_json: Optional[str] = None
    preview_summary: Optional[str] = None
    uploaded_file: Optional[str] = None
    log_path: Optional[str] = None


class OnboardingJobMetrics(BaseModel):
    """Summary metrics surfaced in the job dashboard."""

    best_val_loss: Optional[float] = None
    forecast_rmse: Optional[float] = None
    rollout_rmse: Optional[float] = None
    best_unit_template: Optional[str] = None


class OnboardingJobResponse(BaseModel):
    """Status snapshot for an onboarding adaptation job."""

    job_id: str
    preview_id: str
    status: str
    stage: str
    progress_message: Optional[str] = None
    created_at: float
    updated_at: float
    artifacts: OnboardingJobArtifacts = Field(default_factory=OnboardingJobArtifacts)
    metrics: OnboardingJobMetrics = Field(default_factory=OnboardingJobMetrics)
    error: Optional[str] = None


class OnboardingJobReportResponse(BaseModel):
    """Final report payload for a completed onboarding job."""

    job: OnboardingJobResponse
    summary: Dict[str, Any]
    report_markdown: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for /health."""

    status: str = Field("ok", description="'ok' when models are ready, 'degraded' when no models loaded.")
    loaded_systems: List[str] = Field(default_factory=list, description="Names of registered process systems.")
    version: str = Field("0.1.0", description="API version.")
    uptime_seconds: Optional[float] = Field(None, description="Seconds since service startup.")
    models_ready: bool = Field(True, description="True when at least one model or universal runtime is loaded.")


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
    code: Optional[str] = None
