"""Phase 6 interactive demo website for the Digital Twin Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dte.demo.engine import (
    compare_scenarios,
    default_control_sequence,
    default_disturbance_sequence,
    load_demo_config,
    optimize_control_sequence,
    rollout_scenario,
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
          min-height: 72vh;
          padding: 4.5rem 4rem 3rem 4rem;
          border-top: 1px solid var(--dte-line);
          border-bottom: 1px solid var(--dte-line);
          background:
            linear-gradient(135deg, rgba(33, 75, 69, 0.92), rgba(21, 33, 28, 0.96)),
            linear-gradient(135deg, rgba(176, 106, 43, 0.14), transparent 42%);
          color: #f5f1e7;
          animation: heroRise 700ms ease-out;
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
          font-family: "Trebuchet MS", "Gill Sans", sans-serif;
        }
        .dte-title {
          position: relative;
          z-index: 1;
          max-width: 8.5ch;
          font-size: clamp(3.2rem, 9vw, 7.6rem);
          line-height: 0.9;
          margin: 0;
          font-weight: 600;
          letter-spacing: -0.04em;
          font-family: Georgia, "Times New Roman", serif;
        }
        .dte-summary {
          position: relative;
          z-index: 1;
          max-width: 36rem;
          margin-top: 1.35rem;
          font-size: 1.05rem;
          line-height: 1.65;
          color: rgba(245, 241, 231, 0.86);
          font-family: "Trebuchet MS", "Gill Sans", sans-serif;
        }
        .dte-chiprow {
          position: relative;
          z-index: 1;
          display: flex;
          flex-wrap: wrap;
          gap: 0.6rem;
          margin-top: 1.8rem;
        }
        .dte-chip {
          border: 1px solid rgba(245, 241, 231, 0.22);
          padding: 0.55rem 0.85rem;
          font-size: 0.82rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          background: rgba(255,255,255,0.04);
          backdrop-filter: blur(4px);
          font-family: "Trebuchet MS", "Gill Sans", sans-serif;
        }
        .dte-section-title {
          font-family: Georgia, "Times New Roman", serif;
          font-size: 2rem;
          line-height: 1.05;
          color: var(--dte-ink);
          margin-top: 2rem;
          margin-bottom: 0.3rem;
        }
        .dte-section-copy {
          max-width: 46rem;
          color: rgba(21, 33, 28, 0.74);
          font-family: "Trebuchet MS", "Gill Sans", sans-serif;
          margin-bottom: 1.2rem;
        }
        .dte-rule {
          border-top: 1px solid var(--dte-line);
          margin: 1.6rem 0 1rem 0;
        }
        .dte-note {
          font-size: 0.86rem;
          color: rgba(21, 33, 28, 0.68);
          font-family: "Trebuchet MS", "Gill Sans", sans-serif;
        }
        .dte-flow {
          display: grid;
          grid-template-columns: repeat(3, minmax(120px, 1fr));
          gap: 1rem;
          align-items: center;
          margin: 1.2rem 0 0.5rem 0;
        }
        .dte-node {
          padding: 1.1rem 0.9rem;
          border-top: 1px solid var(--dte-line);
          border-bottom: 1px solid var(--dte-line);
          background: rgba(255,255,255,0.45);
          font-family: "Trebuchet MS", "Gill Sans", sans-serif;
        }
        .dte-arrow {
          text-align: center;
          color: var(--dte-brass);
          font-size: 1.4rem;
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


@st.cache_resource(show_spinner=False)
def _load_runtime(system_name: str):
    config_path = SYSTEM_CONFIGS[system_name]
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        system_config = yaml.safe_load(handle) or {}
    spec = get_system_spec(system_config)
    simulator = get_simulator(system_name, system_config)
    return spec, simulator


def _target_state_vector(spec, demo_cfg: dict[str, Any]) -> np.ndarray:
    target = np.asarray(spec.default_initial_state, dtype=np.float32).copy()
    for name, value in (demo_cfg.get("target_state") or {}).items():
        if name in spec.state_names:
            target[spec.state_names.index(name)] = float(value)
    return target


def _trajectory_figure(
    spec,
    highlight_states: list[str],
    comparison: dict[str, Any],
    optimized: dict[str, Any] | None,
) -> go.Figure:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    times = np.asarray(candidate["times"])
    fig = make_subplots(
        rows=len(highlight_states),
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
                name="Candidate band" if row_index == 1 else None,
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
                    line=dict(color="#15211C", width=2),
                ),
                row=row_index,
                col=1,
            )

    fig.update_layout(
        height=max(360, 230 * len(highlight_states)),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.7)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title_text="Time")
    return fig


def _format_metric(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.3f}"


def _render_unit_demo(demo_cfg: dict[str, Any]) -> None:
    spec, simulator = _load_runtime(demo_cfg["system"])
    n_steps = int(demo_cfg.get("n_steps", 25))
    dt = float(demo_cfg.get("dt", 0.1))
    highlight_states = list(demo_cfg.get("highlight_states", spec.state_names[:2]))
    target_state = _target_state_vector(spec, demo_cfg)
    initial_state = np.asarray(spec.default_initial_state, dtype=np.float32)
    baseline_controls = default_control_sequence(spec, n_steps)
    disturbances = default_disturbance_sequence(spec, n_steps)

    left_col, right_col = st.columns([0.85, 1.85], gap="large")

    with left_col:
        st.markdown(f"<div class='dte-section-title'>{demo_cfg['title']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='dte-section-copy'>{demo_cfg.get('description', '')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='dte-rule'></div>", unsafe_allow_html=True)
        st.markdown("**Candidate control**")
        candidate_control = baseline_controls[0].copy()
        for control_idx, control_name in enumerate(spec.control_names):
            bounds = spec.control_ranges[control_name]
            candidate_control[control_idx] = st.slider(
                control_name,
                min_value=float(bounds[0]),
                max_value=float(bounds[1]),
                value=float(baseline_controls[0, control_idx]),
                key=f"{demo_cfg['id']}_{control_name}",
            )

        st.markdown("**Shared disturbance**")
        disturbance_vector = disturbances[0].copy()
        for dist_idx, dist_name in enumerate(spec.disturbance_names):
            bounds = spec.disturbance_ranges[dist_name]
            disturbance_vector[dist_idx] = st.slider(
                dist_name,
                min_value=float(bounds[0]),
                max_value=float(bounds[1]),
                value=float(disturbances[0, dist_idx]),
                key=f"{demo_cfg['id']}_{dist_name}",
            )

        st.markdown("**Target state**")
        tracked_state_names = list(highlight_states)
        for state_name in tracked_state_names:
            st.caption(f"{state_name}: {_format_metric(target_state[spec.state_names.index(state_name)])}")

        optimize_clicked = st.button(
            "Recommend Control Sequence",
            key=f"{demo_cfg['id']}_optimize",
            use_container_width=True,
        )

    candidate_controls = np.tile(candidate_control[None, :], (n_steps, 1))
    disturbances = np.tile(disturbance_vector[None, :], (n_steps, 1))
    comparison = compare_scenarios(
        spec,
        simulator,
        initial_state=initial_state,
        baseline_controls=baseline_controls,
        candidate_controls=candidate_controls,
        disturbances=disturbances,
        dt=dt,
        model=None,
        n_samples=20,
        seed=11,
    )
    optimized_rollout = None
    optimized_result = None
    if optimize_clicked:
        optimized_result = optimize_control_sequence(
            spec,
            simulator,
            initial_state=initial_state,
            disturbances=disturbances,
            dt=dt,
            target_state=target_state,
            tracked_state_names=tracked_state_names,
            n_candidates=48,
            seed=17,
        )
        optimized_rollout = {
            "mean": np.asarray(optimized_result["predicted_states"]),
        }

    with right_col:
        st.plotly_chart(
            _trajectory_figure(spec, highlight_states, comparison, optimized_rollout),
            use_container_width=True,
        )
        metric_cols = st.columns(len(highlight_states) + 1)
        for index, state_name in enumerate(highlight_states):
            state_delta = comparison["summary"]["candidate_advantage"][state_name]
            metric_cols[index].metric(
                label=f"{state_name} final delta",
                value=_format_metric(state_delta),
            )
        candidate_constraints = comparison["candidate"]["constraint_summary"]
        metric_cols[-1].metric(
            label="Constraint risk",
            value=_format_metric(candidate_constraints["above_upper_bound_rate"] + candidate_constraints["below_lower_bound_rate"]),
        )
        st.markdown(
            "<div class='dte-note'>Bands come from a small simulator ensemble so the demo still works without a trained checkpoint. The FastAPI demo routes can use a loaded DigitalTwin when one is available.</div>",
            unsafe_allow_html=True,
        )
        if optimized_result is not None:
            st.markdown("<div class='dte-rule'></div>", unsafe_allow_html=True)
            st.markdown("**Recommended sequence**")
            summary_cols = st.columns(spec.control_dim + 1)
            for idx, control_name in enumerate(spec.control_names):
                sequence = np.asarray(optimized_result["control_sequence"])[:, idx]
                summary_cols[idx].metric(
                    label=f"{control_name} end",
                    value=_format_metric(float(sequence[-1])),
                )
            summary_cols[-1].metric(
                label="Objective",
                value=_format_metric(float(optimized_result["objective"])),
            )


def _render_flowsheet_preview() -> None:
    st.markdown("<div class='dte-section-title'>Flowsheet Preview</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-section-copy'>Phase 6 ships three live unit demos and a small-section preview to show how the product scales from one asset to a process section.</div>",
        unsafe_allow_html=True,
    )
    preview_specs = [
        (
            "Exchanger -> Reactor -> Tank",
            ["Heat exchanger", "CSTR", "Storage tank"],
            "A temperature-conditioned reactor train with a downstream buffer and purge.",
        ),
        (
            "Reactor -> Separator -> Recycle",
            ["CSTR", "Separator", "Recycle loop"],
            "A recycle section where composition and thermal dynamics interact across the loop.",
        ),
    ]
    for title, nodes, description in preview_specs:
        st.markdown(f"**{title}**")
        st.markdown(f"<div class='dte-note'>{description}</div>", unsafe_allow_html=True)
        flow_html = """
        <div class='dte-flow'>
        """
        for index, node in enumerate(nodes):
            flow_html += f"<div class='dte-node'><strong>{node}</strong></div>"
            if index < len(nodes) - 1:
                flow_html += "<div class='dte-arrow'>→</div>"
        flow_html += "</div>"
        st.markdown(flow_html, unsafe_allow_html=True)
        st.markdown("<div class='dte-rule'></div>", unsafe_allow_html=True)


def main() -> None:
    _inject_css()
    demo_page_cfg = _load_demo_page_config()
    theme = demo_page_cfg.get("theme", {})
    st.markdown(
        f"""
        <section class="dte-hero">
          <div class="dte-kicker">Phase 6 Demo Website</div>
          <h1 class="dte-title">{theme.get('product_name', 'Digital Twin Engine')}</h1>
          <div class="dte-summary">{theme.get('headline', '')}<br><br>{theme.get('summary', '')}</div>
          <div class="dte-chiprow">
            <div class="dte-chip">Live control sliders</div>
            <div class="dte-chip">Forecast uncertainty</div>
            <div class="dte-chip">Constraint visibility</div>
            <div class="dte-chip">Control recommendations</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='dte-section-title'>Interactive Demos</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-section-copy'>Each workspace below keeps one job: show what changes, how much it changes, and what risk travels with that control move.</div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([demo["title"] for demo in demo_page_cfg.get("demos", [])] + ["Flowsheet"])
    for tab, demo_cfg in zip(tabs[:-1], demo_page_cfg.get("demos", [])):
        with tab:
            _render_unit_demo(demo_cfg)
    with tabs[-1]:
        _render_flowsheet_preview()


if __name__ == "__main__":
    main()
