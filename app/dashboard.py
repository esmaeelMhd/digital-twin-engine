"""Streamlit dashboard for presenting trained Digital Twin Engine runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from dte.control.mpc import SamplingMPC
from dte.control.pid import CSTRPIDController
from dte.models.digital_twin import DigitalTwin
from dte.simulators.cstr import CSTRParams, CSTRSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CSTR_CONFIG = PROJECT_ROOT / "configs" / "cstr_default.yaml"
DEFAULT_MPC_CONFIG = PROJECT_ROOT / "configs" / "mpc_default.yaml"


st.set_page_config(
    page_title="Digital Twin Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _safe_read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return None


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 1_000_000 or (numeric != 0 and abs(numeric) < 1e-3):
        return f"{numeric:.3e}"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.3f}"


def _format_duration(seconds: Any) -> str:
    if seconds is None:
        return "N/A"
    try:
        total_seconds = int(float(seconds))
    except (TypeError, ValueError):
        return "N/A"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return "Unknown"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _artifacts_health(summary: dict[str, Any] | None, history: dict[str, Any] | None) -> tuple[str, str]:
    if not summary:
        return "Needs review", "No training summary was found for this run."
    if summary.get("failure_reason"):
        return "Needs review", f"Training recorded failure reason: {summary['failure_reason']}"
    if summary.get("non_finite_detected"):
        return "Needs review", "Training detected non-finite values."
    epochs_completed = summary.get("epochs_completed", 0) or 0
    if epochs_completed < 10:
        return "Early run", "This looks like a short training run, useful for testing but not ideal as a showcase artifact."
    if not history or not history.get("train_loss"):
        return "Partial", "The checkpoint exists, but the dashboard could not find a full training history."
    return "Ready", "This run has the expected training artifacts and looks suitable for a demo."


def _discover_runs() -> list[dict[str, Any]]:
    if not OUTPUTS_DIR.exists():
        return []

    runs: list[dict[str, Any]] = []
    for directory in OUTPUTS_DIR.iterdir():
        if not directory.is_dir():
            continue

        config_path = directory / "config.yaml"
        if not config_path.exists():
            continue

        best_model_path = directory / "best_model.eqx"
        final_model_path = directory / "final_model.eqx"
        if best_model_path.exists():
            model_path = best_model_path
            checkpoint_label = "best"
        elif final_model_path.exists():
            model_path = final_model_path
            checkpoint_label = "final"
        else:
            continue

        cstr_config_path = directory / "cstr_config.yaml"
        summary_path = directory / "training_summary.json"
        history_path = directory / "training_history.json"
        summary = _safe_read_json(summary_path)

        timestamps = [path.stat().st_mtime for path in (model_path, config_path) if path.exists()]
        updated_at = max(timestamps) if timestamps else None
        display_name = directory.name

        if summary and summary.get("epochs_completed") is not None:
            display_name = f"{display_name} ({summary['epochs_completed']} epochs)"

        runs.append(
            {
                "name": directory.name,
                "display_name": display_name,
                "directory": directory,
                "model_path": model_path,
                "config_path": config_path,
                "cstr_config_path": cstr_config_path if cstr_config_path.exists() else DEFAULT_CSTR_CONFIG,
                "summary_path": summary_path,
                "history_path": history_path,
                "checkpoint_label": checkpoint_label,
                "updated_at": updated_at,
            }
        )

    runs.sort(key=lambda item: item["updated_at"] or 0.0, reverse=True)
    return runs


@st.cache_data(show_spinner=False)
def load_run_artifacts(
    config_path: str,
    cstr_config_path: str,
    summary_path: str,
    history_path: str,
) -> dict[str, Any]:
    return {
        "config": _safe_read_yaml(Path(config_path)) or {},
        "cstr_config": _safe_read_yaml(Path(cstr_config_path)) or {},
        "summary": _safe_read_json(Path(summary_path)),
        "history": _safe_read_json(Path(history_path)),
    }


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, config_path: str) -> tuple[DigitalTwin | None, dict[str, Any] | None]:
    config = _safe_read_yaml(Path(config_path))
    if not config:
        return None, None
    model = DigitalTwin.load(model_path, config)
    return model, config


@st.cache_resource(show_spinner=False)
def load_simulator(cstr_config_path: str) -> tuple[CSTRSimulator | None, dict[str, Any] | None]:
    config = _safe_read_yaml(Path(cstr_config_path))
    if not config:
        return None, None
    cstr_params_dict = {key: float(value) for key, value in config.get("cstr", {}).items()}
    params = CSTRParams(**cstr_params_dict)
    return CSTRSimulator(params), config


def _build_training_history_figure(history: dict[str, Any] | None) -> go.Figure | None:
    if not history or not history.get("train_loss"):
        return None

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    steps = history.get("step", list(range(1, len(train_loss) + 1)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=train_loss,
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#0F766E", width=3),
            marker=dict(size=6),
        )
    )
    if val_loss:
        if len(val_loss) == len(steps):
            val_steps = steps
        else:
            val_steps = np.linspace(steps[0], steps[-1], len(val_loss))
        fig.add_trace(
            go.Scatter(
                x=val_steps,
                y=val_loss,
                mode="lines+markers",
                name="Val Loss",
                line=dict(color="#B45309", width=3, dash="dash"),
                marker=dict(size=7),
            )
        )

    fig.update_layout(
        title="Training History",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="Loss",
        xaxis_title="Step",
    )
    fig.update_yaxes(type="log")
    return fig


def _build_simulation_figure(
    times: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    *,
    ca_setpoint: float,
    t_setpoint: float,
    disturbance_step: float | None,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Ca Concentration",
            "Cb Product",
            "Reactor Temperature",
            "Coolant Temperature",
            "Feed Flow Rate",
            "Coolant Inlet Temperature",
        ),
        vertical_spacing=0.1,
    )

    palette = {
        "state": "#0F766E",
        "setpoint": "#B91C1C",
        "control": "#1D4ED8",
    }
    state_names = ["Ca", "Cb", "T", "Tc"]

    for index, name in enumerate(state_names):
        row = index // 2 + 1
        col = index % 2 + 1
        fig.add_trace(
            go.Scatter(
                x=times,
                y=states[:, index],
                mode="lines",
                name=name,
                line=dict(color=palette["state"], width=3),
            ),
            row=row,
            col=col,
        )
        if index in (0, 2):
            target = ca_setpoint if index == 0 else t_setpoint
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[target] * len(times),
                    mode="lines",
                    name=f"{name} setpoint",
                    line=dict(color=palette["setpoint"], dash="dash", width=2),
                ),
                row=row,
                col=col,
            )

    control_names = ["F_in", "Tc_in"]
    for index, name in enumerate(control_names):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=controls[:, index],
                mode="lines",
                name=name,
                line=dict(color=palette["control"], width=3),
            ),
            row=3,
            col=index + 1,
        )

    if disturbance_step is not None:
        for row in (1, 2, 3):
            for col in (1, 2):
                fig.add_vline(
                    x=disturbance_step,
                    line_width=1,
                    line_dash="dot",
                    line_color="#6B7280",
                    row=row,
                    col=col,
                )

    fig.update_layout(
        height=820,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _render_overview(
    run: dict[str, Any],
    model: DigitalTwin | None,
    model_config: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    history: dict[str, Any] | None,
) -> None:
    status_label, status_message = _artifacts_health(summary, history)
    param_counts = model.get_parameter_count() if model is not None else {}

    left_col, right_col = st.columns([1.3, 1.0])
    with left_col:
        st.subheader("CSTR Digital Twin Showcase")
        st.markdown(
            "This dashboard presents a trained physics-informed latent neural SDE for "
            "continuous stirred-tank reactor control. The goal is to make the trained "
            "model understandable to visitors before moving into live simulation and "
            "real-time deployment scenarios."
        )
        st.caption(
            f"Selected run: `{run['name']}` | checkpoint: `{run['checkpoint_label']}` | "
            f"updated: {_format_timestamp(run['updated_at'])}"
        )
    with right_col:
        st.info(f"Run Status: **{status_label}**\n\n{status_message}")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Checkpoint", run["checkpoint_label"].upper())
    metric_cols[1].metric(
        "Epochs",
        _format_number(summary.get("epochs_completed") if summary else None),
    )
    metric_cols[2].metric(
        "Training Time",
        _format_duration(summary.get("training_seconds") if summary else None),
    )
    metric_cols[3].metric(
        "Best Val Loss",
        _format_number(summary.get("best_val_loss") if summary else None),
    )
    metric_cols[4].metric(
        "Parameters",
        _format_number(param_counts.get("total")),
    )

    chart_col, notes_col = st.columns([1.5, 1.0])
    with chart_col:
        history_figure = _build_training_history_figure(history)
        if history_figure is None:
            st.warning("Training history is not available for this run.")
        else:
            st.plotly_chart(history_figure, use_container_width=True)

    with notes_col:
        with st.container(border=True):
            st.markdown("**Model Snapshot**")
            st.write(f"Run directory: `{run['directory'].name}`")
            st.write(f"Config file: `{run['config_path'].name}`")
            if model_config:
                st.write(f"Latent dimension: `{model_config['model'].get('latent_dim', 'N/A')}`")
                st.write(f"Hidden dimension: `{model_config['model'].get('hidden_dim', 'N/A')}`")
                st.write(f"Sequence length: `{model_config['training'].get('seq_len', 'N/A')}`")
                st.write(f"Batch size: `{model_config['training'].get('batch_size', 'N/A')}`")

        with st.container(border=True):
            st.markdown("**Why This Matters**")
            st.write("The dashboard is being structured around three proof points:")
            st.write("1. The model was trained and versioned cleanly.")
            st.write("2. The digital twin can drive closed-loop reactor simulations.")
            st.write("3. The same interface can later host explainability and live system hooks.")

    detail_cols = st.columns(3)
    with detail_cols[0]:
        with st.container(border=True):
            st.markdown("**Artifacts Present**")
            st.write(f"Model: `{run['model_path'].name}`")
            st.write(f"Summary: `{'yes' if summary else 'no'}`")
            st.write(f"History: `{'yes' if history else 'no'}`")
    with detail_cols[1]:
        with st.container(border=True):
            st.markdown("**Training Summary**")
            if summary:
                st.write(f"Steps completed: `{_format_number(summary.get('steps_completed'))}`")
                st.write(f"Validation cadence: `{_format_number(summary.get('val_every'))}`")
                st.write(f"Seed: `{_format_number(summary.get('seed'))}`")
            else:
                st.write("No `training_summary.json` found.")
    with detail_cols[2]:
        with st.container(border=True):
            st.markdown("**Architecture**")
            if param_counts:
                st.write(f"Encoder: `{_format_number(param_counts.get('encoder'))}`")
                st.write(f"Decoder: `{_format_number(param_counts.get('decoder'))}`")
                st.write(f"Latent SDE: `{_format_number(param_counts.get('latent_sde'))}`")
            else:
                st.write("Model metadata unavailable.")


def _render_simulation(
    *,
    model: DigitalTwin | None,
    model_config: dict[str, Any] | None,
    simulator: CSTRSimulator | None,
    cstr_config: dict[str, Any] | None,
    control_mode: str,
    disturbance_scenario: str,
    n_steps: int,
    ca_setpoint: float,
    t_setpoint: float,
    run_simulation: bool,
) -> None:
    st.subheader("Closed-Loop Simulation")
    st.caption("Use the selected trained artifact to compare open-loop, PID, and AI-MPC reactor behavior.")

    if not run_simulation:
        st.info("Choose a scenario in the sidebar and click `Run Simulation`.")
        return

    if model is None or model_config is None:
        st.error("The selected run could not be loaded.")
        return
    if simulator is None or cstr_config is None:
        st.error("The CSTR simulator configuration could not be loaded.")
        return

    with st.spinner("Running simulation..."):
        key = jax.random.PRNGKey(42)
        dt = cstr_config.get("simulation", {}).get("dt", 0.1)
        initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
        setpoints = jnp.array([ca_setpoint, 0.0, t_setpoint, 300.0])
        disturbance_index = n_steps // 4
        nominal_disturbance = jnp.array([1.0, 320.0])

        disturbances = jnp.ones((n_steps, 2)) * nominal_disturbance
        disturbance_step_time: float | None = None
        if disturbance_scenario == "Step in Ca_in":
            disturbances = disturbances.at[disturbance_index:, 0].set(1.5)
            disturbance_step_time = disturbance_index * dt
        elif disturbance_scenario == "Step in T_in":
            disturbances = disturbances.at[disturbance_index:, 1].set(330.0)
            disturbance_step_time = disturbance_index * dt
        elif disturbance_scenario == "Random":
            key, subkey = jax.random.split(key)
            disturbances = jax.random.uniform(subkey, (n_steps, 2))
            disturbances = disturbances * jnp.array([1.0, 30.0]) + jnp.array([0.5, 305.0])
        else:
            disturbance_step_time = None

        if control_mode == "AI-MPC":
            mpc_config = _safe_read_yaml(DEFAULT_MPC_CONFIG) or {}
            mpc_controller = SamplingMPC(model, {"mpc": mpc_config["mpc"]})
            params = jnp.ones(model_config["model"].get("param_dim", 6))
            key, subkey = jax.random.split(key)
            result = mpc_controller.run_closed_loop(
                simulator,
                initial_state,
                disturbances,
                setpoints,
                params,
                n_steps,
                dt,
                subkey,
            )
        elif control_mode == "PID":
            pid_controller = CSTRPIDController(
                T_setpoint=t_setpoint,
                Ca_setpoint=ca_setpoint,
                dt=dt,
            )
            result = pid_controller.run_closed_loop(
                simulator,
                initial_state,
                disturbances,
                n_steps,
                dt,
            )
        else:
            controls = jnp.tile(jnp.array([50.0, 300.0])[None, :], (n_steps, 1))
            result = simulator.simulate(
                initial_state,
                controls,
                disturbances,
                (0.0, n_steps * dt),
                dt,
                n_steps,
            )
            result["controls"] = controls

    times = np.arange(n_steps) * dt
    states = np.array(result["states"])
    controls = np.array(result["controls"])
    simulation_figure = _build_simulation_figure(
        times,
        states,
        controls,
        ca_setpoint=ca_setpoint,
        t_setpoint=t_setpoint,
        disturbance_step=disturbance_step_time,
    )
    st.plotly_chart(simulation_figure, use_container_width=True)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Avg T Error", f"{np.abs(states[:, 2] - t_setpoint).mean():.2f} K")
    metric_cols[1].metric("Avg Ca Error", f"{np.abs(states[:, 0] - ca_setpoint).mean():.3f} mol/L")
    metric_cols[2].metric("Control Effort", f"{np.sum(np.diff(controls, axis=0) ** 2):.2f}")
    metric_cols[3].metric("Simulation Horizon", f"{times[-1]:.1f} min")

    st.caption(
        "Current AI-MPC playback uses a nominal parameter vector for the learned model. "
        "Parameter-aware simulation controls can be added in the next iteration."
    )


def _render_placeholder(title: str, description: str, bullets: list[str]) -> None:
    st.subheader(title)
    st.write(description)
    with st.container(border=True):
        st.markdown("**Next build items**")
        for bullet in bullets:
            st.write(f"- {bullet}")


def _render_model_details(
    run: dict[str, Any],
    model: DigitalTwin | None,
    model_config: dict[str, Any] | None,
    cstr_config: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> None:
    st.subheader("Model Details")
    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            st.markdown("**Run Files**")
            st.write(f"Directory: `{run['directory']}`")
            st.write(f"Model: `{run['model_path'].name}`")
            st.write(f"Config: `{run['config_path'].name}`")
            st.write(f"CSTR config: `{Path(run['cstr_config_path']).name}`")

        with st.container(border=True):
            st.markdown("**Training Config**")
            if model_config:
                st.json(model_config)
            else:
                st.write("Config unavailable.")

    with right_col:
        with st.container(border=True):
            st.markdown("**Model Metadata**")
            if model is None:
                st.write("Model unavailable.")
            else:
                param_counts = model.get_parameter_count()
                st.write(f"Total parameters: `{_format_number(param_counts.get('total'))}`")
                st.write(f"Encoder parameters: `{_format_number(param_counts.get('encoder'))}`")
                st.write(f"Decoder parameters: `{_format_number(param_counts.get('decoder'))}`")
                st.write(f"Latent SDE parameters: `{_format_number(param_counts.get('latent_sde'))}`")

        with st.container(border=True):
            st.markdown("**Operating Context**")
            if summary:
                st.write(f"Output dir: `{summary.get('output_dir', run['directory'])}`")
                st.write(f"Timed out: `{summary.get('timed_out', False)}`")
            if cstr_config:
                st.json(cstr_config)


st.title("Digital Twin Engine")
st.markdown("### Presenting trained CSTR digital twins for process-control demos")

available_runs = _discover_runs()

if not available_runs:
    st.warning("No trained runs were found in `outputs/`. Train a model first with `python scripts/train.py`.")
    st.stop()

run_options = {run["display_name"]: run for run in available_runs}
selected_run_label = st.sidebar.selectbox("Featured CSTR Run", list(run_options.keys()))
selected_run = run_options[selected_run_label]

artifacts = load_run_artifacts(
    str(selected_run["config_path"]),
    str(selected_run["cstr_config_path"]),
    str(selected_run["summary_path"]),
    str(selected_run["history_path"]),
)
model, model_config = load_model(
    str(selected_run["model_path"]),
    str(selected_run["config_path"]),
)
simulator, cstr_config = load_simulator(str(selected_run["cstr_config_path"]))

with st.sidebar:
    st.caption(
        f"Showing `{selected_run['checkpoint_label']}` checkpoint from `{selected_run['name']}` "
        f"updated {_format_timestamp(selected_run['updated_at'])}."
    )
    st.divider()
    st.subheader("Simulation Scenario")
    ca_setpoint = st.slider("Ca Setpoint (mol/L)", 0.1, 2.0, 0.8, 0.1)
    t_setpoint = st.slider("Temperature Setpoint (K)", 300.0, 400.0, 340.0, 5.0)
    control_mode = st.radio("Control Mode", ["Open Loop", "PID", "AI-MPC"], index=2)
    disturbance_scenario = st.radio(
        "Disturbance",
        ["None", "Step in Ca_in", "Step in T_in", "Random"],
        index=1,
    )
    n_steps = st.slider("Simulation Steps", 50, 500, 200, 50)
    run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)
    st.divider()
    st.subheader("Artifact Notes")
    st.write("This version focuses on model selection, training history, and a cleaner CSTR presentation.")
    st.write("Explainability and real-time benchmark tabs are prepared as the next build step.")

overview_tab, simulation_tab, twin_tab, benchmark_tab, details_tab = st.tabs(
    [
        "Overview",
        "Live Simulation",
        "Twin vs Reality",
        "Benchmarks",
        "Model Details",
    ]
)

with overview_tab:
    _render_overview(
        selected_run,
        model,
        model_config,
        artifacts["summary"],
        artifacts["history"],
    )

with simulation_tab:
    _render_simulation(
        model=model,
        model_config=model_config,
        simulator=simulator,
        cstr_config=cstr_config,
        control_mode=control_mode,
        disturbance_scenario=disturbance_scenario,
        n_steps=n_steps,
        ca_setpoint=ca_setpoint,
        t_setpoint=t_setpoint,
        run_simulation=run_simulation,
    )

with twin_tab:
    _render_placeholder(
        "Twin vs Reality",
        "This tab will become the explainability and validation view for a selected trained run.",
        [
            "Overlay true simulator trajectories and model predictions.",
            "Show uncertainty bands from ensemble forecasts.",
            "Plot error-over-time and conservation residuals.",
            "Add local sensitivity views around the chosen operating point.",
        ],
    )

with benchmark_tab:
    _render_placeholder(
        "Controller Benchmarks",
        "This section is reserved for scenario-based comparisons between Open Loop, PID, and AI-MPC.",
        [
            "Track overshoot, settling time, and control effort.",
            "Benchmark disturbance rejection across canned CSTR scenarios.",
            "Summarize controller tradeoffs in one presentation-ready view.",
        ],
    )

with details_tab:
    _render_model_details(
        selected_run,
        model,
        model_config,
        artifacts["cstr_config"],
        artifacts["summary"],
    )

st.markdown("---")
st.caption(
    "Digital Twin Engine dashboard for model presentation and control demos. "
    "Current milestone: artifact-aware CSTR showcase."
)
