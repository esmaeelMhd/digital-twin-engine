"""Streamlit dashboard for presenting trained Digital Twin Engine runs."""

from __future__ import annotations

import json
import os
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
from dte.models.unit.digital_twin import DigitalTwin
from dte.simulators.registry import get_system_spec, get_simulator

from app._theme import inject_theme


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


# ---------------------------------------------------------------------------
# Optional password authentication
# ---------------------------------------------------------------------------

def _check_auth() -> bool:
    """Return True when authentication passes or is not required."""
    required_pwd = os.environ.get("STREAMLIT_AUTH_PASSWORD", "")
    if not required_pwd:
        return True

    if st.session_state.get("_dte_authenticated"):
        return True

    st.title("Digital Twin Engine")
    st.subheader("Login required")
    with st.form("auth_form"):
        pwd = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if pwd == required_pwd:
                st.session_state["_dte_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
    return False


if not _check_auth():
    st.stop()

inject_theme()


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


@st.cache_data(ttl=10)
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

        system_config_path = directory / "system_config.yaml"
        if not system_config_path.exists():
            system_config_path = directory / "cstr_config.yaml"
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
                "system_config_path": str(system_config_path) if system_config_path.exists() else str(DEFAULT_CSTR_CONFIG),
                "summary_path": summary_path,
                "history_path": history_path,
                "checkpoint_label": checkpoint_label,
                "updated_at": updated_at,
            }
        )

    runs.sort(key=lambda item: item["updated_at"] or 0.0, reverse=True)
    return runs


@st.cache_data(show_spinner="Reading artifacts...")
def load_run_artifacts(
    config_path: str,
    system_config_path: str,
    summary_path: str,
    history_path: str,
) -> dict[str, Any]:
    return {
        "config": _safe_read_yaml(Path(config_path)) or {},
        "system_config": _safe_read_yaml(Path(system_config_path)) or {},
        "summary": _safe_read_json(Path(summary_path)),
        "history": _safe_read_json(Path(history_path)),
    }


@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_model(model_path: str, config_path: str, system_config_path: str) -> tuple[DigitalTwin | None, dict[str, Any] | None, str | None]:
    try:
        config = _safe_read_yaml(Path(config_path))
        if not config:
            return None, None, "Training config not found."
        system_config = _safe_read_yaml(Path(system_config_path)) or {}
        system_spec = get_system_spec(system_config)
        model = DigitalTwin.load(model_path, config, system_spec=system_spec)
        return model, config, None
    except Exception as exc:
        return None, None, str(exc)


@st.cache_resource(show_spinner="Loading simulator...")
def load_simulator(system_config_path: str):
    try:
        system_config = _safe_read_yaml(Path(system_config_path))
        if not system_config:
            return None, None, "System config not found."
        system_spec = get_system_spec(system_config)
        simulator = get_simulator(system_spec.name, system_config)
        return simulator, system_config, None
    except Exception as exc:
        return None, None, str(exc)


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
            hovertemplate="Step %{x}<br>Train loss: %{y:.4e}<extra></extra>",
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
                hovertemplate="Step %{x:.0f}<br>Val loss: %{y:.4e}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Training History",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="Loss (log scale)",
        xaxis_title="Training Step",
        hovermode="x unified",
    )
    fig.update_yaxes(type="log")
    return fig


def _build_simulation_figure(
    times: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    system_spec,
    *,
    setpoints: dict[str, float] | None = None,
    disturbance_step: float | None = None,
) -> go.Figure:
    """Build a generic simulation figure driven by SystemSpec names."""
    state_names = list(system_spec.state_names)
    control_names = list(system_spec.control_names)

    n_series = len(state_names) + len(control_names)
    n_cols = 2
    n_rows = (n_series + n_cols - 1) // n_cols

    all_titles = state_names + control_names
    # Pad to even number
    while len(all_titles) < n_rows * n_cols:
        all_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=all_titles,
        vertical_spacing=0.08,
    )

    palette = {
        "state": "#0F766E",
        "setpoint": "#B91C1C",
        "control": "#1D4ED8",
    }

    for index, name in enumerate(state_names):
        row = index // n_cols + 1
        col = index % n_cols + 1
        fig.add_trace(
            go.Scatter(
                x=times,
                y=states[:, index],
                mode="lines",
                name=name,
                line=dict(color=palette["state"], width=3),
                hovertemplate=f"{name}: %{{y:.3f}}<br>t=%{{x:.1f}} min<extra></extra>",
            ),
            row=row,
            col=col,
        )
        if setpoints and name in setpoints:
            target = setpoints[name]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[target] * len(times),
                    mode="lines",
                    name=f"{name} setpoint",
                    line=dict(color=palette["setpoint"], dash="dash", width=2),
                    hovertemplate=f"Setpoint: {target:.3f}<extra>{name} target</extra>",
                ),
                row=row,
                col=col,
            )

    for index, name in enumerate(control_names):
        series_idx = len(state_names) + index
        row = series_idx // n_cols + 1
        col = series_idx % n_cols + 1
        fig.add_trace(
            go.Scatter(
                x=times,
                y=controls[:, index],
                mode="lines",
                name=name,
                line=dict(color=palette["control"], width=3),
                hovertemplate=f"{name}: %{{y:.3f}}<br>t=%{{x:.1f}} min<extra></extra>",
            ),
            row=row,
            col=col,
        )

    if disturbance_step is not None:
        for r in range(1, n_rows + 1):
            for c in range(1, n_cols + 1):
                fig.add_vline(
                    x=disturbance_step,
                    line_width=1,
                    line_dash="dot",
                    line_color="#6B7280",
                    row=r,
                    col=c,
                )

    fig.update_layout(
        height=max(420, 280 * n_rows),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title_text="Time (min)")
    return fig


def _render_overview(
    run: dict[str, Any],
    model: DigitalTwin | None,
    model_config: dict[str, Any] | None,
    system_config: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    history: dict[str, Any] | None,
) -> None:
    status_label, status_message = _artifacts_health(summary, history)
    param_counts = model.get_parameter_count() if model is not None else {}

    # Derive system name for dynamic copy
    system_name = "Digital Twin"
    if system_config:
        try:
            spec = get_system_spec(system_config)
            system_name = spec.name.replace("_", " ").title()
        except Exception:
            pass

    left_col, right_col = st.columns([1.3, 1.0])
    with left_col:
        st.subheader(f"{system_name} — Trained Digital Twin")
        st.markdown(
            f"This dashboard presents a trained physics-informed latent neural SDE for "
            f"**{system_name}** process simulation and control. "
            "Select a run in the sidebar, then explore the simulation and model details tabs."
        )
        st.caption(
            f"Run: `{run['name']}` · checkpoint: `{run['checkpoint_label']}` · "
            f"updated: {_format_timestamp(run['updated_at'])}"
        )
    with right_col:
        if status_label == "Ready":
            st.success(f"**{status_label}** — {status_message}")
        elif status_label in ("Needs review",):
            st.error(f"**{status_label}** — {status_message}")
        elif status_label == "Early run":
            st.warning(f"**{status_label}** — {status_message}")
        else:
            st.info(f"**{status_label}** — {status_message}")

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
            st.markdown("**What This Demonstrates**")
            st.write(f"1. A physics-informed neural SDE trained on **{system_name}** dynamics.")
            st.write("2. Closed-loop simulation comparing Open Loop, PID (CSTR only), and AI-MPC control.")
            st.write("3. A model artifact versioned with full training provenance.")

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
    simulator,
    system_config: dict[str, Any] | None,
    control_mode: str,
    disturbance_scenario: str,
    n_steps: int,
    setpoint_values: dict[str, float],
    run_simulation: bool,
) -> None:
    st.subheader("Closed-Loop Simulation")
    st.caption("Select a scenario in the sidebar and click Run Simulation to compare control modes.")

    # Render persisted results even when button is not pressed
    if not run_simulation and "sim_results" not in st.session_state:
        st.info("Choose a scenario in the sidebar and click **Run Simulation**.")
        return

    if model is None or model_config is None:
        st.error("The selected run could not be loaded. Check that the checkpoint and config files are present.")
        return
    if simulator is None or system_config is None:
        st.error("The simulator configuration could not be loaded. Check the system config file.")
        return

    try:
        system_spec = get_system_spec(system_config)
    except Exception as exc:
        st.error(f"Could not build SystemSpec from configuration: {exc}")
        return

    is_cstr = system_spec.name == "cstr"

    if run_simulation:
        with st.spinner("Running simulation..."):
            key = jax.random.PRNGKey(42)
            dt = system_config.get("simulation", {}).get("dt", 0.1)

            initial_state = system_spec.default_initial_state_array()
            nominal_disturbance = system_spec.default_nominal_disturbance_array()
            dist_dim = system_spec.disturbance_dim

            # Build setpoints array for MPC
            setpoints_arr = initial_state
            if is_cstr:
                ca_sp = setpoint_values.get("Ca", 0.8)
                t_sp = setpoint_values.get("T", 340.0)
                setpoints_arr = jnp.array([ca_sp, 0.0, t_sp, 300.0])

            disturbance_index = n_steps // 4
            disturbances = jnp.tile(nominal_disturbance[None, :], (n_steps, 1))
            disturbance_step_time: float | None = None

            if disturbance_scenario == "Step in Ca_in" and is_cstr:
                disturbances = disturbances.at[disturbance_index:, 0].set(1.5)
                disturbance_step_time = disturbance_index * dt
            elif disturbance_scenario == "Step in T_in" and is_cstr:
                disturbances = disturbances.at[disturbance_index:, 1].set(
                    nominal_disturbance[1].item() + 10.0
                )
                disturbance_step_time = disturbance_index * dt
            elif disturbance_scenario == "Random":
                key, subkey = jax.random.split(key)
                disturbances = jax.random.uniform(subkey, (n_steps, dist_dim))
                disturbances = disturbances * 0.1 + nominal_disturbance[None, :]

            if control_mode == "AI-MPC":
                mpc_config = _safe_read_yaml(DEFAULT_MPC_CONFIG) or {}
                mpc_controller = SamplingMPC(model, {"mpc": mpc_config["mpc"]})
                params = jnp.ones(model_config["model"].get("param_dim", system_spec.param_dim))
                key, subkey = jax.random.split(key)
                result = mpc_controller.run_closed_loop(
                    simulator,
                    initial_state,
                    disturbances,
                    setpoints_arr,
                    params,
                    n_steps,
                    dt,
                    subkey,
                )
            elif control_mode == "PID" and is_cstr:
                from dte.control.pid import CSTRPIDController
                ca_sp = setpoint_values.get("Ca", 0.8)
                t_sp = setpoint_values.get("T", 340.0)
                pid_controller = CSTRPIDController(
                    T_setpoint=t_sp,
                    Ca_setpoint=ca_sp,
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
                ranges = system_spec.control_ranges
                if ranges:
                    nominal_control = jnp.array([
                        (r[0] + r[1]) / 2.0
                        for r in ranges.values()
                    ])
                else:
                    ctrl_center = system_spec.normalization.control_center
                    nominal_control = jnp.array(ctrl_center)
                controls_traj = jnp.tile(nominal_control[None, :], (n_steps, 1))
                result = simulator.simulate(
                    initial_state,
                    controls_traj,
                    disturbances,
                    (0.0, n_steps * dt),
                    dt,
                    n_steps,
                )
                result["controls"] = controls_traj

        times = np.arange(n_steps) * dt
        states = np.array(result["states"])
        controls_out = np.array(result["controls"])

        st.session_state["sim_results"] = {
            "times": times,
            "states": states,
            "controls_out": controls_out,
            "disturbance_step_time": disturbance_step_time,
            "setpoint_values": setpoint_values,
            "system_spec_name": system_spec.name,
            "is_cstr": is_cstr,
            "dt": dt,
        }

    sim = st.session_state.get("sim_results")
    if sim is None:
        return

    times = sim["times"]
    states = sim["states"]
    controls_out = sim["controls_out"]
    disturbance_step_time = sim["disturbance_step_time"]
    stored_setpoints = sim["setpoint_values"]
    is_cstr_stored = sim["is_cstr"]

    # Build setpoints dict for overlay
    setpoints_overlay: dict[str, float] | None = None
    if is_cstr_stored:
        ca_sp = stored_setpoints.get("Ca", 0.8)
        t_sp = stored_setpoints.get("T", 340.0)
        setpoints_overlay = {"Ca": ca_sp, "T": t_sp}

    simulation_figure = _build_simulation_figure(
        times,
        states,
        controls_out,
        system_spec,
        setpoints=setpoints_overlay,
        disturbance_step=disturbance_step_time,
    )
    st.plotly_chart(simulation_figure, use_container_width=True)

    if is_cstr_stored:
        ca_sp = stored_setpoints.get("Ca", 0.8)
        t_sp = stored_setpoints.get("T", 340.0)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Avg T Error", f"{np.abs(states[:, 2] - t_sp).mean():.2f} K")
        metric_cols[1].metric("Avg Ca Error", f"{np.abs(states[:, 0] - ca_sp).mean():.3f} mol/L")
        metric_cols[2].metric(
            "Control Effort",
            f"{np.sum(np.diff(controls_out, axis=0) ** 2):.2f}",
            help="Sum of squared control increments (lower = smoother policy)",
        )
        metric_cols[3].metric("Simulation Horizon", f"{times[-1]:.1f} min")

    st.caption(
        "AI-MPC uses a nominal parameter vector. "
        "PID is available for CSTR only."
    )


def _render_model_details(
    run: dict[str, Any],
    model: DigitalTwin | None,
    model_config: dict[str, Any] | None,
    system_config: dict[str, Any] | None,
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
            st.write(f"System config: `{Path(run['system_config_path']).name}`")

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
            if system_config:
                st.json(system_config)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("Digital Twin Engine")
st.subheader("Trained process digital twins for simulation and control demos")

available_runs = _discover_runs()

if not available_runs:
    st.warning(
        "No trained runs found in `outputs/`. "
        "Train a model first with `python scripts/train.py`."
    )
    st.code(
        "python scripts/generate_data.py --config configs/cstr_default.yaml\n"
        "python scripts/train.py --data_dir data/ --n_epochs 10",
        language="bash",
    )
    st.stop()

run_options = {run["display_name"]: run for run in available_runs}
selected_run_label = st.sidebar.selectbox("Featured Run", list(run_options.keys()))
selected_run = run_options[selected_run_label]

artifacts = load_run_artifacts(
    str(selected_run["config_path"]),
    str(selected_run["system_config_path"]),
    str(selected_run["summary_path"]),
    str(selected_run["history_path"]),
)

model, model_config, model_error = load_model(
    str(selected_run["model_path"]),
    str(selected_run["config_path"]),
    str(selected_run["system_config_path"]),
)
if model_error:
    st.error(f"Could not load model: {model_error}")

simulator, system_config_loaded, sim_error = load_simulator(str(selected_run["system_config_path"]))
if sim_error:
    st.error(f"Could not load simulator: {sim_error}")

# Determine system spec for sidebar controls
try:
    _spec_for_sidebar = get_system_spec(system_config_loaded) if system_config_loaded else None
except Exception:
    _spec_for_sidebar = None

_is_cstr_sidebar = _spec_for_sidebar is not None and _spec_for_sidebar.name == "cstr"

with st.sidebar:
    st.caption(
        f"Showing `{selected_run['checkpoint_label']}` checkpoint · "
        f"updated {_format_timestamp(selected_run['updated_at'])}"
    )
    st.divider()
    st.subheader("Simulation Scenario")

    # Build setpoint sliders dynamically from spec
    setpoint_values: dict[str, float] = {}
    if _is_cstr_sidebar:
        setpoint_values["Ca"] = st.slider("Ca Setpoint (mol/L)", 0.1, 2.0, 0.8, 0.1)
        setpoint_values["T"] = st.slider("Temperature Setpoint (K)", 300.0, 400.0, 340.0, 5.0)

    # Control modes: only offer PID for CSTR
    control_options = ["Open Loop", "AI-MPC"]
    if _is_cstr_sidebar:
        control_options.insert(1, "PID")
    control_mode = st.radio("Control Mode", control_options, index=len(control_options) - 1)

    # Disturbance options
    if _is_cstr_sidebar:
        disturbance_options = ["None", "Step in Ca_in", "Step in T_in", "Random"]
        default_dist_idx = 1
    else:
        disturbance_options = ["None", "Random"]
        default_dist_idx = 0
    disturbance_scenario = st.radio("Disturbance", disturbance_options, index=default_dist_idx)

    n_steps = st.slider("Simulation Steps", 50, 500, 200, 50)
    run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)

overview_tab, simulation_tab, details_tab = st.tabs(
    ["Overview", "Live Simulation", "Model Details"]
)

with overview_tab:
    _render_overview(
        selected_run,
        model,
        model_config,
        system_config_loaded,
        artifacts["summary"],
        artifacts["history"],
    )

with simulation_tab:
    _render_simulation(
        model=model,
        model_config=model_config,
        simulator=simulator,
        system_config=system_config_loaded,
        control_mode=control_mode,
        disturbance_scenario=disturbance_scenario,
        n_steps=n_steps,
        setpoint_values=setpoint_values,
        run_simulation=run_simulation,
    )

with details_tab:
    _render_model_details(
        selected_run,
        model,
        model_config,
        artifacts["system_config"],
        artifacts["summary"],
    )

st.markdown("---")
st.caption("Digital Twin Engine · physics-informed latent neural SDE digital twins")
