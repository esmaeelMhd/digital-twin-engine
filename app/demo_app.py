"""Presentation-ready demo website for the Digital Twin Engine V1 release."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dte.demo.engine import (
    UniversalDemoRuntime,
    build_signal_sequence,
    compare_scenarios,
    load_demo_config,
    load_demo_model_runtime,
    load_demo_release_snapshot,
    optimize_control_sequence,
)
from dte.simulators.registry import get_simulator, get_system_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_CONFIG = PROJECT_ROOT / "configs" / "demo_app.yaml"
SYSTEM_CONFIGS = {
    "cstr": PROJECT_ROOT / "configs" / "cstr_default.yaml",
    "heat_exchanger": PROJECT_ROOT / "configs" / "heat_exchanger_default.yaml",
    "two_tank": PROJECT_ROOT / "configs" / "two_tank_default.yaml",
}


st.set_page_config(
    page_title="Digital Twin Engine Demo",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --dte-ink: #15211c;
          --dte-moss: #214b45;
          --dte-sand: #f0ebe0;
          --dte-brass: #b06a2b;
          --dte-cream: #fbf8f1;
          --dte-line: rgba(21, 33, 28, 0.14);
        }
        .stApp {
          background:
            radial-gradient(circle at 20% 15%, rgba(176, 106, 43, 0.10), transparent 30%),
            radial-gradient(circle at 85% 10%, rgba(33, 75, 69, 0.14), transparent 24%),
            linear-gradient(180deg, #f8f3e8 0%, #fbf8f1 55%, #f4eee1 100%);
          color: var(--dte-ink);
        }
        .block-container {
          max-width: 1340px;
          padding-top: 2rem;
          padding-bottom: 4rem;
        }
        .dte-hero {
          position: relative;
          overflow: hidden;
          min-height: 36vh;
          padding: 3rem 4rem 2.5rem 4rem;
          border-top: 1px solid var(--dte-line);
          border-bottom: 1px solid var(--dte-line);
          background:
            linear-gradient(135deg, rgba(33, 75, 69, 0.92), rgba(21, 33, 28, 0.96)),
            linear-gradient(135deg, rgba(176, 106, 43, 0.14), transparent 42%);
          color: #f5f1e7 !important;
          animation-fill-mode: forwards;
        }
        .dte-hero *, .dte-hero h1, .dte-hero h2, .dte-hero h3 {
          color: #f5f1e7 !important;
        }
        .dte-hero-first {
          animation: heroRise 700ms ease-out forwards;
        }
        .dte-hero::after {
          content: "";
          position: absolute;
          inset: 0;
          background:
            linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,0.05) 1px, transparent 1px);
          background-size: 72px 72px;
          mask-image: linear-gradient(180deg, rgba(0,0,0,0.7), transparent);
          pointer-events: none;
        }
        .dte-kicker {
          position: relative;
          z-index: 1;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          font-size: 0.78rem;
          opacity: 0.74;
          margin-bottom: 1rem;
          font-family: "Trebuchet MS", "Gill Sans", Arial, sans-serif;
        }
        .dte-title {
          position: relative;
          z-index: 1;
          max-width: 8.5ch;
          font-size: clamp(2.8rem, 7vw, 6.4rem);
          line-height: 0.9;
          margin: 0;
          font-weight: 600;
          letter-spacing: -0.04em;
          font-family: Georgia, "Times New Roman", serif;
        }
        .dte-summary {
          position: relative;
          z-index: 1;
          max-width: 40rem;
          margin-top: 1.2rem;
          font-size: 1.05rem;
          line-height: 1.65;
          color: rgba(245, 241, 231, 0.86);
          font-family: "Trebuchet MS", "Gill Sans", Arial, sans-serif;
        }
        .dte-chiprow {
          position: relative;
          z-index: 1;
          display: flex;
          flex-wrap: wrap;
          gap: 0.6rem;
          margin-top: 1.6rem;
        }
        .dte-chip {
          border: 1px solid rgba(245, 241, 231, 0.22);
          padding: 0.5rem 0.8rem;
          font-size: 0.82rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          background: rgba(255,255,255,0.04);
          backdrop-filter: blur(4px);
          font-family: "Trebuchet MS", "Gill Sans", Arial, sans-serif;
        }
        h2.dte-section-title {
          font-family: Georgia, "Times New Roman", serif;
          font-size: 2rem;
          line-height: 1.05;
          color: var(--dte-ink);
          margin-top: 2rem;
          margin-bottom: 0.3rem;
        }
        .dte-section-copy {
          max-width: 48rem;
          color: rgba(21, 33, 28, 0.74);
          font-family: "Trebuchet MS", "Gill Sans", Arial, sans-serif;
          margin-bottom: 1.2rem;
        }
        .dte-rule {
          border-top: 1px solid var(--dte-line);
          margin: 1.6rem 0 1rem 0;
        }
        .dte-note {
          font-size: 0.86rem;
          color: rgba(21, 33, 28, 0.68);
          font-family: "Trebuchet MS", "Gill Sans", Arial, sans-serif;
        }
        .dte-flow {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
          gap: 1rem;
          align-items: center;
          margin: 1.2rem 0 0.5rem 0;
        }
        .dte-node {
          padding: 1.1rem 0.9rem;
          border-top: 1px solid var(--dte-line);
          border-bottom: 1px solid var(--dte-line);
          background: rgba(255,255,255,0.45);
          font-family: "Trebuchet MS", "Gill Sans", Arial, sans-serif;
        }
        .dte-arrow {
          text-align: center;
          color: #b06a2b;
          font-size: 1.4rem;
          min-width: 1.5rem;
        }
        @keyframes heroRise {
          from { opacity: 0; transform: translateY(24px); }
          to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_demo_page_config() -> dict[str, Any]:
    return load_demo_config(DEFAULT_DEMO_CONFIG)


@st.cache_data(show_spinner=False)
def _load_release_snapshot() -> dict[str, Any]:
    config = _load_demo_page_config()
    return load_demo_release_snapshot(config, config_path=DEFAULT_DEMO_CONFIG)


@st.cache_resource(show_spinner="Loading shared checkpoint...")
def _load_release_runtime() -> UniversalDemoRuntime | None:
    config = _load_demo_page_config()
    return load_demo_model_runtime(config, config_path=DEFAULT_DEMO_CONFIG)


@st.cache_resource(show_spinner="Loading system runtime...")
def _load_runtime(system_name: str):
    import yaml

    config_path = SYSTEM_CONFIGS[system_name]
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            system_config = yaml.safe_load(handle) or {}
        spec = get_system_spec(system_config)
        simulator = get_simulator(system_name, system_config)
        return spec, simulator
    except Exception as exc:
        return None, None


def _read_text_if_exists(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.4f}"


def _target_state_vector(spec, demo_cfg: dict[str, Any]) -> np.ndarray:
    target = np.asarray(spec.default_initial_state, dtype=np.float32).copy()
    for name, value in (demo_cfg.get("target_state") or {}).items():
        if name in spec.state_names:
            target[spec.state_names.index(name)] = float(value)
    return target


def _initial_state_vector(spec, demo_cfg: dict[str, Any]) -> np.ndarray:
    initial = np.asarray(spec.default_initial_state, dtype=np.float32).copy()
    for name, value in (demo_cfg.get("initial_state") or {}).items():
        if name in spec.state_names:
            initial[spec.state_names.index(name)] = float(value)
    return initial


def _clip_adjusted_sequence(
    spec,
    sequence: np.ndarray,
    adjustments: dict[str, float],
    *,
    kind: str,
) -> np.ndarray:
    adjusted = np.asarray(sequence, dtype=np.float32).copy()
    if kind == "control":
        names = list(spec.control_names)
        ranges = spec.control_ranges
    else:
        names = list(spec.disturbance_names)
        ranges = spec.disturbance_ranges

    for index, name in enumerate(names):
        delta = float(adjustments.get(name, 0.0))
        low, high = ranges[name]
        adjusted[:, index] = np.clip(adjusted[:, index] + delta, low, high)
    return adjusted


def _trajectory_figure(
    spec,
    highlight_states: list[str],
    comparison: dict[str, Any],
    optimized: dict[str, Any] | None,
) -> go.Figure:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    times = np.asarray(candidate["times"])
    n_rows = len(highlight_states)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=highlight_states,
    )
    for row_index, state_name in enumerate(highlight_states, start=1):
        state_idx = spec.state_names.index(state_name)
        candidate_mean = np.asarray(candidate["mean"])[:, state_idx]
        candidate_p05 = np.asarray(candidate["p05"])[:, state_idx]
        candidate_p95 = np.asarray(candidate["p95"])[:, state_idx]
        baseline_mean = np.asarray(baseline["mean"])[:, state_idx]

        # 90% confidence band
        fig.add_trace(
            go.Scatter(
                x=times,
                y=candidate_p95,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row_index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=candidate_p05,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(33, 75, 69, 0.12)",
                name="90% forecast interval" if row_index == 1 else None,
                showlegend=row_index == 1,
                hoverinfo="skip",
            ),
            row=row_index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=baseline_mean,
                mode="lines",
                name="Baseline" if row_index == 1 else None,
                showlegend=row_index == 1,
                line=dict(color="#B06A2B", width=2, dash="dash"),
                hovertemplate=f"{state_name} baseline: %{{y:.3f}}<br>t=%{{x:.2f}}<extra>Baseline</extra>",
            ),
            row=row_index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=candidate_mean,
                mode="lines",
                name="Candidate" if row_index == 1 else None,
                showlegend=row_index == 1,
                line=dict(color="#214B45", width=3),
                hovertemplate=f"{state_name} candidate: %{{y:.3f}}<br>t=%{{x:.2f}}<extra>Candidate</extra>",
            ),
            row=row_index,
            col=1,
        )
        if optimized is not None:
            optimized_mean = np.asarray(optimized["mean"])[:, state_idx]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=optimized_mean,
                    mode="lines",
                    name="Recommended" if row_index == 1 else None,
                    showlegend=row_index == 1,
                    line=dict(color="#15211C", width=2, dash="dot"),
                    hovertemplate=f"{state_name} recommended: %{{y:.3f}}<br>t=%{{x:.2f}}<extra>Recommended</extra>",
                ),
                row=row_index,
                col=1,
            )

    fig.update_layout(
        height=max(360, 230 * n_rows),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.7)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (min)", row=n_rows, col=1)
    fig.update_yaxes(title_text="Value")
    return fig


def _render_release_overview(snapshot: dict[str, Any], runtime: UniversalDemoRuntime | None) -> None:
    st.markdown("<h2 class='dte-section-title'>Release Overview</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-section-copy'>This demo shows the V1 shared checkpoint evaluated across multiple process unit families, including a customer adaptation pilot on an ingested historian dataset.</div>",
        unsafe_allow_html=True,
    )

    top_metrics = st.columns(4)
    top_metrics[0].metric("Release gate", str(snapshot.get("milestone_status", "unknown")).upper())
    top_metrics[1].metric("Best validation loss", _format_metric(snapshot.get("train_best_val_loss")))
    top_metrics[2].metric(
        snapshot.get("eval_metric_name") or "Eval metric",
        _format_metric(snapshot.get("eval_metric_value")),
    )
    top_metrics[3].metric(
        "Customer adaptation",
        _format_metric(snapshot.get("customer_best_val_loss")),
    )

    per_system = snapshot.get("per_system_total_loss", {})
    if per_system:
        st.markdown("**Per-system evaluation totals**")
        system_cols = st.columns(len(per_system))
        for col, (system_name, loss_value) in zip(system_cols, per_system.items()):
            col.metric(system_name.replace("_", " ").title(), _format_metric(loss_value))

    if runtime is not None:
        st.success(
            f"Shared checkpoint loaded from `{snapshot.get('model_path')}` — using universal-model rollouts for unit demos.",
        )
    else:
        st.warning(
            "Release checkpoint not loaded. Demos will fall back to simulator ensembles until the V1 model artifacts are available.",
        )


def _render_unit_demo(
    demo_cfg: dict[str, Any],
    runtime: UniversalDemoRuntime | None,
) -> None:
    system_name = demo_cfg["system"]
    try:
        spec, simulator = _load_runtime(system_name)
    except Exception as exc:
        st.error(f"Could not load system `{system_name}`: {exc}")
        return

    if spec is None or simulator is None:
        st.error(
            f"Could not load system `{system_name}`. "
            "Check that the system config YAML exists in `configs/`."
        )
        return

    n_steps = int(demo_cfg.get("n_steps", 25))
    dt = float(demo_cfg.get("dt", 0.1))
    highlight_states = list(demo_cfg.get("highlight_states", spec.state_names[:2]))
    target_state = _target_state_vector(spec, demo_cfg)
    initial_state = _initial_state_vector(spec, demo_cfg)
    demo_id = demo_cfg["id"]

    baseline_profile = demo_cfg.get("baseline_control_profile")
    baseline_controls = build_signal_sequence(
        spec,
        n_steps,
        signal_kind="control",
        profile=baseline_profile,
    )

    disturbance_presets = demo_cfg.get("disturbance_presets") or [
        {
            "id": "nominal",
            "title": "Nominal operation",
            "description": "Default disturbance path from the system specification.",
            "profile": None,
        }
    ]
    candidate_profiles = demo_cfg.get("candidate_profiles") or [
        {
            "id": "baseline",
            "title": "Baseline policy",
            "description": "No alternate candidate profile configured.",
            "profile": baseline_profile,
        }
    ]

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown(f"<h2 class='dte-section-title'>{demo_cfg['title']}</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='dte-section-copy'>{demo_cfg.get('description', '')}</div>",
            unsafe_allow_html=True,
        )
        if demo_cfg.get("operator_goal"):
            st.caption(f"Operator goal: {demo_cfg['operator_goal']}")

        with st.form(key=f"{demo_id}_scenario_form", border=False):
            selected_disturbance_id = st.selectbox(
                "Disturbance preset",
                options=[item["id"] for item in disturbance_presets],
                format_func=lambda value: next(
                    item["title"] for item in disturbance_presets if item["id"] == value
                ),
                key=f"{demo_id}_disturbance_preset",
            )
            disturbance_cfg = next(
                item for item in disturbance_presets if item["id"] == selected_disturbance_id
            )
            st.markdown(
                f"<div class='dte-note'>{disturbance_cfg.get('description', '')}</div>",
                unsafe_allow_html=True,
            )

            selected_candidate_id = st.selectbox(
                "Candidate operating move",
                options=[item["id"] for item in candidate_profiles],
                format_func=lambda value: next(
                    item["title"] for item in candidate_profiles if item["id"] == value
                ),
                key=f"{demo_id}_candidate_preset",
            )
            candidate_cfg = next(
                item for item in candidate_profiles if item["id"] == selected_candidate_id
            )
            st.markdown(
                f"<div class='dte-note'>{candidate_cfg.get('description', '')}</div>",
                unsafe_allow_html=True,
            )

            control_adjustments = {name: 0.0 for name in spec.control_names}
            disturbance_adjustments = {name: 0.0 for name in spec.disturbance_names}
            with st.expander("Fine trim", expanded=False):
                st.caption("Apply small bounded offsets on top of the preset trajectories.")
                for control_name in spec.control_names:
                    low, high = spec.control_ranges[control_name]
                    delta_max = 0.15 * (high - low)
                    control_adjustments[control_name] = st.slider(
                        f"{control_name} trim",
                        min_value=float(-delta_max),
                        max_value=float(delta_max),
                        value=0.0,
                        key=f"{demo_id}_{control_name}_trim",
                    )
                for disturbance_name in spec.disturbance_names:
                    low, high = spec.disturbance_ranges[disturbance_name]
                    delta_max = 0.15 * (high - low)
                    disturbance_adjustments[disturbance_name] = st.slider(
                        f"{disturbance_name} trim",
                        min_value=float(-delta_max),
                        max_value=float(delta_max),
                        value=0.0,
                        key=f"{demo_id}_{disturbance_name}_trim",
                    )

            st.markdown("<div class='dte-rule'></div>", unsafe_allow_html=True)
            st.markdown("**Tracked target state**")
            for state_name in highlight_states:
                state_idx = spec.state_names.index(state_name)
                st.caption(f"{state_name}: {_format_metric(float(target_state[state_idx]))}")

            run_scenario = st.form_submit_button(
                demo_cfg.get("run_button_label", "Run Scenario"),
                type="primary",
                use_container_width=True,
            )
            optimize_clicked = st.form_submit_button(
                demo_cfg.get("optimize_button_label", "Recommend Control Sequence"),
                use_container_width=True,
            )

    disturbances = build_signal_sequence(
        spec,
        n_steps,
        signal_kind="disturbance",
        profile=disturbance_cfg.get("profile"),
    )
    disturbances = _clip_adjusted_sequence(
        spec,
        disturbances,
        disturbance_adjustments,
        kind="disturbance",
    )
    candidate_controls = build_signal_sequence(
        spec,
        n_steps,
        signal_kind="control",
        profile=candidate_cfg.get("profile"),
    )
    candidate_controls = _clip_adjusted_sequence(
        spec,
        candidate_controls,
        control_adjustments,
        kind="control",
    )

    try:
        comparison = compare_scenarios(
            spec,
            simulator,
            initial_state=initial_state,
            baseline_controls=baseline_controls,
            candidate_controls=candidate_controls,
            disturbances=disturbances,
            dt=dt,
            model=runtime,
            params=None,
            n_samples=20,
            seed=11,
        )
    except Exception as exc:
        st.error(f"Scenario comparison failed: {exc}")
        return

    # Persist optimization result across reruns
    opt_key = f"optimized_{demo_id}"
    if optimize_clicked:
        with st.spinner("Optimizing control sequence..."):
            try:
                optimized_result = optimize_control_sequence(
                    spec,
                    simulator,
                    initial_state=initial_state,
                    disturbances=disturbances,
                    dt=dt,
                    target_state=target_state,
                    tracked_state_names=list(highlight_states),
                    n_candidates=int(demo_cfg.get("optimization", {}).get("n_candidates", 64)),
                    seed=int(demo_cfg.get("optimization", {}).get("seed", 17)),
                )
                st.session_state[opt_key] = optimized_result
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")
                st.session_state.pop(opt_key, None)

    # Clear persisted optimization if scenario controls changed
    if run_scenario:
        st.session_state.pop(opt_key, None)

    stored_optimized = st.session_state.get(opt_key)
    optimized_rollout = {"mean": np.asarray(stored_optimized["predicted_states"])} if stored_optimized else None

    with right_col:
        st.plotly_chart(
            _trajectory_figure(spec, highlight_states, comparison, optimized_rollout),
            use_container_width=True,
        )
        metric_cols = st.columns(len(highlight_states) + 2)
        metric_cols[0].metric("Forecast source", comparison["candidate"]["source"].replace("_", " "))
        for index, state_name in enumerate(highlight_states, start=1):
            state_delta = comparison["summary"]["candidate_advantage"][state_name]
            metric_cols[index].metric(
                label=f"{state_name} final delta",
                value=_format_metric(float(state_delta)),
            )
        candidate_constraints = comparison["candidate"]["constraint_summary"]
        constraint_risk = (
            candidate_constraints["above_upper_bound_rate"]
            + candidate_constraints["below_lower_bound_rate"]
        )
        metric_cols[-1].metric(
            label="Constraint risk",
            value=_format_metric(constraint_risk),
            help="Fraction of time steps where state bounds are violated. Lower is better.",
        )
        if runtime is not None:
            st.markdown(
                "<div class='dte-note'>Uncertainty bands come from the shared checkpoint. "
                "The recommendation line is a lightweight search over the physical simulator.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='dte-note'>Release checkpoint unavailable — uncertainty bands come from a simulator ensemble.</div>",
                unsafe_allow_html=True,
            )
        if stored_optimized is not None:
            st.markdown("<div class='dte-rule'></div>", unsafe_allow_html=True)
            st.markdown("**Recommended sequence**")
            summary_cols = st.columns(spec.control_dim + 1)
            for idx, control_name in enumerate(spec.control_names):
                sequence = np.asarray(stored_optimized["control_sequence"])[:, idx]
                summary_cols[idx].metric(
                    label=f"{control_name} end",
                    value=_format_metric(float(sequence[-1])),
                )
            summary_cols[-1].metric(
                label="Objective",
                value=_format_metric(float(stored_optimized["objective"])),
                help="Lower objective = closer to target state",
            )


def _render_customer_story(snapshot: dict[str, Any]) -> None:
    st.markdown("<h2 class='dte-section-title'>Customer Adaptation</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-section-copy'>"
        "This release includes a full onboarding and adaptation pass on an ingested historian export. "
        "A shared checkpoint can be matched, adapted, and validated in a customer-facing workflow with minimal data."
        "</div>",
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Status", str(snapshot.get("customer_status", "unknown")).upper())
    metric_cols[1].metric(
        "Best template",
        str(snapshot.get("customer_best_unit_template", "n/a")).replace("_", " "),
    )
    metric_cols[2].metric(
        "Best validation loss",
        _format_metric(snapshot.get("customer_best_val_loss")),
    )
    metric_cols[3].metric(
        "Forecast RMSE",
        _format_metric(snapshot.get("customer_forecast_rmse")),
    )
    st.metric("Rollout RMSE", _format_metric(snapshot.get("customer_rollout_rmse")))

    report_text = _read_text_if_exists(snapshot.get("customer_report_path"))
    if report_text:
        with st.expander("Validation report", expanded=False):
            st.markdown(report_text)
    else:
        st.info("Customer validation report was not found in the configured release workspace.")


def _render_flowsheet_preview() -> None:
    st.markdown("<h2 class='dte-section-title'>Flowsheet Preview</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-section-copy'>"
        "V1 is unit-first. The flowsheet surface shows the graph direction — "
        "plant section modeling is the next frontier."
        "</div>",
        unsafe_allow_html=True,
    )
    preview_specs = [
        (
            "Exchanger → Reactor → Tank",
            ["Heat exchanger", "CSTR", "Storage tank"],
            "A temperature-conditioned reactor train with a downstream buffer and purge.",
        ),
        (
            "Reactor → Separator → Recycle",
            ["CSTR", "Separator", "Recycle loop"],
            "A recycle section where composition and thermal dynamics interact across the loop.",
        ),
    ]
    for title, nodes, description in preview_specs:
        st.markdown(f"**{title}**")
        st.markdown(f"<div class='dte-note'>{description}</div>", unsafe_allow_html=True)
        # Build flow grid with separate node/arrow divs so CSS grid handles layout
        cells = []
        for index, node in enumerate(nodes):
            cells.append(f"<div class='dte-node'><strong>{node}</strong></div>")
            if index < len(nodes) - 1:
                cells.append("<div class='dte-arrow'>→</div>")
        flow_html = "<div class='dte-flow'>" + "".join(cells) + "</div>"
        st.markdown(flow_html, unsafe_allow_html=True)
        st.markdown("<div class='dte-rule'></div>", unsafe_allow_html=True)


def main() -> None:
    _inject_css()
    demo_page_cfg = _load_demo_page_config()
    theme = demo_page_cfg.get("theme", {})
    snapshot = _load_release_snapshot()
    runtime = _load_release_runtime()

    # Only animate hero on first load per session
    hero_class = "dte-hero"
    if not st.session_state.get("_hero_shown"):
        hero_class += " dte-hero-first"
        st.session_state["_hero_shown"] = True

    n_systems = len(demo_page_cfg.get("demos", []))
    system_count_chip = f"{n_systems} unit {'family' if n_systems == 1 else 'families'}"

    st.markdown(
        f"""
        <section class="{hero_class}">
          <div class="dte-kicker">V1 Foundation Stack</div>
          <h1 class="dte-title">{theme.get('product_name', 'Digital Twin Engine')}</h1>
          <div class="dte-summary">{theme.get('headline', '')}<br><br>{theme.get('summary', '')}</div>
          <div class="dte-chiprow">
            <div class="dte-chip">{snapshot.get('release_label', 'V1 milestone release')}</div>
            <div class="dte-chip">{system_count_chip}</div>
            <div class="dte-chip">Shared checkpoint</div>
            <div class="dte-chip">Customer adaptation proof</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    _render_release_overview(snapshot, runtime)

    st.markdown("<h2 class='dte-section-title'>Interactive Demos</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-section-copy'>Each workspace starts from a fixed baseline policy. "
        "Select a disturbance regime and an alternate operating move, then see how the model "
        "forecasts the state trajectory and constraint risk profile.</div>",
        unsafe_allow_html=True,
    )

    demos = demo_page_cfg.get("demos", [])
    tab_labels = [demo["title"] for demo in demos]
    extra_tabs = ["Customer Adaptation", "Flowsheet"]
    all_labels = tab_labels + extra_tabs

    if not all_labels:
        st.info("No demos configured. Add entries to `configs/demo_app.yaml` to populate the tabs.")
        return

    tabs = st.tabs(all_labels)
    for tab, demo_cfg in zip(tabs[: len(demos)], demos):
        with tab:
            _render_unit_demo(demo_cfg, runtime)

    if len(tabs) >= 2:
        with tabs[-2]:
            _render_customer_story(snapshot)
        with tabs[-1]:
            _render_flowsheet_preview()


if __name__ == "__main__":
    main()
