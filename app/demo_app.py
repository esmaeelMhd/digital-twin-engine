"""Presentation-ready marketing demo for the Digital Twin Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
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
    page_title="Digital Twin Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Design system
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

        :root {
          --ink: #0f1a14;
          --moss: #1a4a3e;
          --moss-l: #2d7a68;
          --brass: #c67a30;
          --brass-l: #d4923e;
          --cream: #faf7f0;
          --sand: #eee8db;
          --line: rgba(15,26,20,.08);
          --line-h: rgba(15,26,20,.15);
          --txt: #1a2b22;
          --txt-m: rgba(26,43,34,.58);
          --glass: rgba(255,255,255,.55);
          --glass-h: rgba(255,255,255,.72);
          --glass-b: rgba(255,255,255,.25);
          --sh-s: 0 2px 8px rgba(15,26,20,.06);
          --sh-m: 0 4px 20px rgba(15,26,20,.08);
          --sh-l: 0 12px 40px rgba(15,26,20,.10);
          --sh-xl: 0 20px 60px rgba(15,26,20,.14);
          --r-s: 8px;
          --r-m: 12px;
          --r-l: 20px;
          --r-xl: 28px;
          --sans: 'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          --serif: 'Playfair Display',Georgia,'Times New Roman',serif;
        }

        /* ── Base — force light regardless of OS/Streamlit theme ── */
        html, body { color-scheme: light !important; }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
          background: linear-gradient(180deg,#f9f5ec 0%,#faf7f0 40%,#f5f0e5 100%) !important;
          color: var(--txt) !important;
          font-family: var(--sans) !important;
        }
        /* Ensure every direct text node picks up the dark ink, not Streamlit's theme text */
        .stApp p, .stApp span, .stApp li, .stApp label,
        .stApp [data-testid="stMarkdownContainer"] {
          color: var(--txt) !important;
          font-family: var(--sans) !important;
        }
        .block-container {
          max-width: 1280px;
          padding: 0 2rem 4rem 2rem;
        }
        h1,h2,h3,h4 { font-family: var(--serif) !important; color: var(--ink) !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent !important; }

        /* ── Metric cards ── */
        [data-testid="stMetric"] {
          background: var(--glass);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border: 1px solid var(--glass-b);
          border-radius: var(--r-m);
          padding: 1rem 1.2rem;
          transition: all 200ms ease;
        }
        [data-testid="stMetric"]:hover {
          background: var(--glass-h);
          box-shadow: var(--sh-m);
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] * {
          font-family: var(--sans) !important;
          font-size: .82rem !important;
          font-weight: 500 !important;
          color: rgba(26,43,34,.62) !important;
          text-transform: uppercase;
          letter-spacing: .04em;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
          font-family: var(--serif) !important;
          font-size: 1.45rem !important;
          font-weight: 600 !important;
          color: #0f1a14 !important;
        }
        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] * {
          font-family: var(--sans) !important;
          font-size: .82rem !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
          gap: 0;
          border-bottom: 1px solid rgba(15,26,20,.15) !important;
          background: transparent !important;
        }
        .stTabs [data-baseweb="tab"] {
          font-family: var(--sans) !important;
          font-weight: 500 !important;
          font-size: .9rem !important;
          color: rgba(26,43,34,.58) !important;
          padding: .8rem 1.4rem !important;
          border-radius: var(--r-s) var(--r-s) 0 0 !important;
          background: transparent !important;
          transition: all 160ms ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
          color: #1a4a3e !important;
          background: rgba(26,74,62,.04) !important;
        }
        .stTabs [aria-selected="true"] {
          color: #1a4a3e !important;
          border-bottom: 2px solid #1a4a3e !important;
          font-weight: 600 !important;
        }
        /* Caption text */
        .stCaption, .stCaption * {
          color: rgba(26,43,34,.55) !important;
          font-family: var(--sans) !important;
        }
        /* Expander */
        .streamlit-expanderHeader, .streamlit-expanderHeader * {
          color: var(--txt) !important;
          font-family: var(--sans) !important;
        }
        /* Select / slider labels */
        .stSelectbox label, .stSlider label,
        .stCheckbox label, .stRadio label {
          color: var(--txt) !important;
          font-family: var(--sans) !important;
        }

        /* ── Buttons ── */
        .stFormSubmitButton > button[kind="primary"],
        .stButton > button[kind="primary"] {
          background: var(--moss) !important;
          color: #fff !important;
          border: none !important;
          border-radius: var(--r-s) !important;
          font-family: var(--sans) !important;
          font-weight: 600 !important;
          letter-spacing: .02em;
          transition: all 180ms ease;
        }
        .stFormSubmitButton > button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover {
          background: var(--moss-l) !important;
          box-shadow: var(--sh-m);
          transform: translateY(-1px);
        }
        .stFormSubmitButton > button:not([kind="primary"]) {
          background: transparent !important;
          color: var(--moss) !important;
          border: 1.5px solid var(--moss) !important;
          border-radius: var(--r-s) !important;
          font-family: var(--sans) !important;
          font-weight: 600 !important;
          transition: all 180ms ease;
        }
        .stFormSubmitButton > button:not([kind="primary"]):hover {
          background: rgba(26,74,62,.06) !important;
        }

        /* ── Sections ── */
        .dte-stitle {
          font-family: var(--serif);
          font-size: clamp(1.6rem,3vw,2.4rem);
          font-weight: 600;
          line-height: 1.1;
          color: var(--ink);
          margin-bottom: .6rem;
        }
        .dte-ssub {
          font-family: var(--sans);
          font-size: 1.05rem;
          line-height: 1.6;
          color: var(--txt-m);
          max-width: 44rem;
          margin-bottom: 2rem;
        }
        .dte-divider {
          height: 1px;
          background: linear-gradient(90deg,transparent,var(--line-h) 20%,var(--line-h) 80%,transparent);
          margin: 3.5rem 0;
        }

        /* ── Stats Bar ── */
        .dte-stats {
          display: grid;
          grid-template-columns: repeat(4,1fr);
          gap: 1rem;
          margin: -2.5rem 0 3rem 0;
          position: relative;
          z-index: 5;
        }
        @media(max-width:768px){ .dte-stats{ grid-template-columns: repeat(2,1fr); } }
        .dte-stat {
          background: var(--glass);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          border: 1px solid var(--glass-b);
          border-radius: var(--r-l);
          padding: 1.8rem 1.4rem;
          text-align: center;
          box-shadow: var(--sh-m);
          transition: all 250ms ease;
        }
        .dte-stat:hover {
          transform: translateY(-4px);
          box-shadow: var(--sh-l);
          background: var(--glass-h);
        }
        .dte-stat-n {
          font-family: var(--serif);
          font-size: 2.6rem;
          font-weight: 700;
          color: var(--moss);
          line-height: 1;
          margin-bottom: .35rem;
        }
        .dte-stat-l {
          font-family: var(--sans);
          font-size: .8rem;
          font-weight: 500;
          color: var(--txt-m);
          text-transform: uppercase;
          letter-spacing: .06em;
        }

        /* ── Industry Cards ── */
        .dte-ig {
          display: grid;
          grid-template-columns: repeat(5,1fr);
          gap: 1rem;
        }
        @media(max-width:1100px){ .dte-ig{ grid-template-columns: repeat(3,1fr); } }
        @media(max-width:700px){ .dte-ig{ grid-template-columns: repeat(2,1fr); } }
        .dte-ic {
          background: var(--glass);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid var(--glass-b);
          border-radius: var(--r-m);
          padding: 1.4rem 1.2rem;
          transition: all 280ms cubic-bezier(.23,1,.32,1);
          cursor: default;
          position: relative;
          overflow: hidden;
        }
        .dte-ic::before {
          content: '';
          position: absolute;
          top: 0; left: 0; right: 0;
          height: 3px;
          background: linear-gradient(90deg,var(--moss),var(--brass));
          opacity: 0;
          transition: opacity 250ms ease;
        }
        .dte-ic:hover {
          transform: translateY(-6px);
          box-shadow: var(--sh-l);
          background: var(--glass-h);
        }
        .dte-ic:hover::before { opacity: 1; }
        .dte-ic-i { font-size: 2rem; margin-bottom: .7rem; display: block; }
        .dte-ic-n {
          font-family: var(--sans);
          font-weight: 600;
          font-size: .92rem;
          color: var(--ink);
          margin-bottom: .3rem;
        }
        .dte-ic-d {
          font-family: var(--sans);
          font-size: .8rem;
          color: var(--txt-m);
          line-height: 1.45;
        }

        /* ── How It Works ── */
        .dte-hw {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 0;
          position: relative;
        }
        @media(max-width:800px){ .dte-hw{ grid-template-columns: 1fr; } }
        .dte-hwc {
          padding: 2.5rem 2rem;
          border: 1px solid var(--line);
          background: var(--glass);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          transition: background 200ms ease;
        }
        .dte-hwc:hover { background: var(--glass-h); }
        .dte-hwc:first-child { border-radius: var(--r-l) 0 0 var(--r-l); }
        .dte-hwc:last-child { border-radius: 0 var(--r-l) var(--r-l) 0; }
        @media(max-width:800px){
          .dte-hwc:first-child { border-radius: var(--r-l) var(--r-l) 0 0; }
          .dte-hwc:last-child { border-radius: 0 0 var(--r-l) var(--r-l); }
        }
        .dte-hwn {
          font-family: var(--serif);
          font-size: 3.5rem;
          font-weight: 700;
          line-height: 1;
          background: linear-gradient(135deg,var(--moss),var(--moss-l));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin-bottom: 1rem;
        }
        .dte-hwt {
          font-family: var(--sans);
          font-weight: 700;
          font-size: 1.05rem;
          color: var(--ink);
          margin-bottom: .6rem;
        }
        .dte-hwb {
          font-family: var(--sans);
          font-size: .88rem;
          line-height: 1.6;
          color: var(--txt-m);
        }

        /* ── Capabilities ── */
        .dte-cg {
          display: grid;
          grid-template-columns: repeat(4,1fr);
          gap: 1.2rem;
        }
        @media(max-width:900px){ .dte-cg{ grid-template-columns: repeat(2,1fr); } }
        .dte-cc {
          background: var(--glass);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid var(--glass-b);
          border-radius: var(--r-m);
          padding: 1.6rem 1.3rem;
          transition: all 250ms ease;
        }
        .dte-cc:hover {
          transform: translateY(-4px);
          box-shadow: var(--sh-l);
          background: var(--glass-h);
        }
        .dte-cci {
          width: 46px; height: 46px;
          border-radius: var(--r-s);
          background: linear-gradient(135deg,var(--moss),var(--moss-l));
          display: flex; align-items: center; justify-content: center;
          font-size: 1.3rem;
          margin-bottom: 1rem;
          color: #faf7f0;
        }
        .dte-cct {
          font-family: var(--sans);
          font-weight: 700;
          font-size: .95rem;
          color: var(--ink);
          margin-bottom: .4rem;
        }
        .dte-ccd {
          font-family: var(--sans);
          font-size: .84rem;
          color: var(--txt-m);
          line-height: 1.5;
        }

        /* ── CTA ── */
        .dte-cta {
          position: relative;
          overflow: hidden;
          margin-top: 5rem;
          padding: 4.5rem 4rem;
          background: linear-gradient(135deg,#0f1a14 0%,#1a4a3e 50%,#0f1a14 100%);
          border-radius: var(--r-xl);
          text-align: center;
        }
        .dte-cta * { color: #faf7f0 !important; }
        .dte-cta::before {
          content: '';
          position: absolute;
          inset: 0;
          background:
            radial-gradient(circle at 30% 40%,rgba(198,122,48,.15),transparent 50%),
            radial-gradient(circle at 70% 60%,rgba(45,122,104,.12),transparent 40%);
          pointer-events: none;
        }
        .dte-cta-t {
          position: relative;
          font-family: var(--serif);
          font-size: clamp(2rem,4.5vw,3.2rem);
          font-weight: 700;
          line-height: 1.1;
          max-width: 28rem;
          margin: 0 auto 1rem auto;
        }
        .dte-cta-d {
          position: relative;
          font-family: var(--sans);
          font-size: 1.1rem;
          line-height: 1.6;
          opacity: .8;
          max-width: 36rem;
          margin: 0 auto 2rem auto;
        }
        .dte-cta-a {
          position: relative;
          display: inline-block;
          padding: 1rem 2.8rem;
          background: var(--brass);
          color: #faf7f0 !important;
          font-family: var(--sans);
          font-size: 1.05rem;
          font-weight: 600;
          letter-spacing: .04em;
          text-decoration: none;
          border-radius: var(--r-s);
          transition: all 200ms ease;
        }
        .dte-cta-a:hover {
          background: var(--brass-l);
          transform: translateY(-2px);
          box-shadow: 0 8px 32px rgba(198,122,48,.3);
          color: #faf7f0 !important;
        }
        .dte-cta-f {
          position: relative;
          font-family: var(--sans);
          font-size: .82rem;
          opacity: .48;
          margin-top: 1rem;
        }

        /* ── Flowsheet ── */
        .dte-fl {
          display: grid;
          grid-template-columns: repeat(auto-fit,minmax(120px,1fr));
          gap: .8rem;
          align-items: center;
          margin: 1rem 0;
        }
        .dte-fln {
          background: var(--glass);
          backdrop-filter: blur(8px);
          border: 1px solid var(--glass-b);
          border-radius: var(--r-s);
          padding: 1rem;
          text-align: center;
          font-family: var(--sans);
          font-weight: 600;
          font-size: .88rem;
          color: var(--ink);
          transition: all 200ms ease;
        }
        .dte-fln:hover {
          background: var(--glass-h);
          box-shadow: var(--sh-m);
        }
        .dte-fla {
          text-align: center;
          color: var(--brass);
          font-size: 1.5rem;
        }
        .dte-note {
          font-size: .86rem;
          color: var(--txt-m);
          font-family: var(--sans);
        }
        .dte-rule {
          height: 1px;
          background: var(--line);
          margin: 1.4rem 0;
        }

        /* ── Animations ── */
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .dte-au { animation: fadeUp 600ms ease-out both; }
        .dte-au-1 { animation-delay: 80ms; }
        .dte-au-2 { animation-delay: 160ms; }
        .dte-au-3 { animation-delay: 240ms; }
        .dte-au-4 { animation-delay: 320ms; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Data loaders (business logic — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

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
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers (business logic — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  Plotly chart builder
# ─────────────────────────────────────────────────────────────────────────────

_PLOTLY_FONT = dict(family="Inter, -apple-system, sans-serif")
_C_MOSS = "#1a4a3e"
_C_BRASS = "#c67a30"
_C_INK = "#0f1a14"
_C_BAND = "rgba(26,74,62,.10)"


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

        fig.add_trace(
            go.Scatter(
                x=times, y=candidate_p95, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ),
            row=row_index, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times, y=candidate_p05, mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=_C_BAND,
                name="90 % forecast interval" if row_index == 1 else None,
                showlegend=row_index == 1, hoverinfo="skip",
            ),
            row=row_index, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times, y=baseline_mean, mode="lines",
                name="Baseline" if row_index == 1 else None,
                showlegend=row_index == 1,
                line=dict(color=_C_BRASS, width=2, dash="dash"),
                hovertemplate=f"{state_name} baseline: %{{y:.3f}}<br>t=%{{x:.2f}}<extra></extra>",
            ),
            row=row_index, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times, y=candidate_mean, mode="lines",
                name="Candidate" if row_index == 1 else None,
                showlegend=row_index == 1,
                line=dict(color=_C_MOSS, width=3),
                hovertemplate=f"{state_name} candidate: %{{y:.3f}}<br>t=%{{x:.2f}}<extra></extra>",
            ),
            row=row_index, col=1,
        )
        if optimized is not None:
            optimized_mean = np.asarray(optimized["mean"])[:, state_idx]
            fig.add_trace(
                go.Scatter(
                    x=times, y=optimized_mean, mode="lines",
                    name="Recommended" if row_index == 1 else None,
                    showlegend=row_index == 1,
                    line=dict(color=_C_INK, width=2, dash="dot"),
                    hovertemplate=f"{state_name} recommended: %{{y:.3f}}<br>t=%{{x:.2f}}<extra></extra>",
                ),
                row=row_index, col=1,
            )

    fig.update_layout(
        height=max(380, 240 * n_rows),
        template="plotly_white",
        font=_PLOTLY_FONT,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.5)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(size=12, family="Inter, sans-serif"),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="Time (min)", row=n_rows, col=1,
        gridcolor="rgba(15,26,20,.06)", zeroline=False,
    )
    fig.update_yaxes(
        title_text="Value",
        gridcolor="rgba(15,26,20,.06)", zeroline=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Hero  (canvas particle network via iframe)
# ─────────────────────────────────────────────────────────────────────────────

_HERO_HTML = """\
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%%;height:100%%;overflow:hidden}
body{
  background:linear-gradient(135deg,#0f1a14 0%%,#1a4a3e 50%%,#0f1a14 100%%);
  font-family:'Inter',-apple-system,sans-serif;color:#faf7f0;
}
canvas{position:fixed;top:0;left:0;width:100%%;height:100%%;pointer-events:none}
.h{position:relative;z-index:2;height:100%%;display:flex;flex-direction:column;justify-content:center;padding:3rem 5vw}
.badge{
  display:inline-block;width:max-content;
  font-size:.72rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;
  color:rgba(250,247,240,.55);border:1px solid rgba(250,247,240,.13);
  padding:.45rem 1rem;border-radius:100px;backdrop-filter:blur(8px);margin-bottom:2rem;
  animation:fu 800ms ease-out both;
}
h1{
  font-family:'Playfair Display',Georgia,serif;
  font-size:clamp(3rem,8vw,5.8rem);font-weight:700;line-height:.92;
  letter-spacing:-.03em;max-width:12ch;margin-bottom:1.4rem;
  animation:fu 800ms ease-out 120ms both;
}
h1 em{font-style:italic;color:#c67a30}
.sub{
  font-size:clamp(1rem,1.8vw,1.15rem);line-height:1.65;
  color:rgba(250,247,240,.68);max-width:38rem;margin-bottom:2rem;
  animation:fu 800ms ease-out 240ms both;
}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;animation:fu 800ms ease-out 360ms both}
.chip{
  font-size:.74rem;font-weight:500;letter-spacing:.06em;text-transform:uppercase;
  padding:.5rem 1rem;background:rgba(250,247,240,.05);
  border:1px solid rgba(250,247,240,.10);border-radius:6px;
  backdrop-filter:blur(6px);
}
.sc{
  position:absolute;bottom:1.5rem;left:50%%;transform:translateX(-50%%);
  font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;opacity:.35;
  animation:pulse 2.5s ease-in-out infinite;
}
@keyframes fu{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%%,100%%{opacity:.35;transform:translateX(-50%%) translateY(0)}50%%{opacity:.6;transform:translateX(-50%%) translateY(-4px)}}
</style></head><body>
<canvas id="c"></canvas>
<div class="h">
  <div class="badge">Physics-Informed AI for Industry</div>
  <h1>Your Plant<br>Deserves<br>a <em>Brain</em></h1>
  <div class="sub">
    A foundation model for industrial process digital twins.
    Connect your historian data. Adapt in minutes.
    Get live probabilistic forecasts and optimal control sequences.
  </div>
  <div class="chips">
    <span class="chip">Neural SDE</span>
    <span class="chip">Few-Shot Transfer</span>
    <span class="chip">Real-Time API</span>
    <span class="chip">10+ Industries</span>
    <span class="chip">Uncertainty Quantified</span>
  </div>
</div>
<div class="sc">Scroll to explore &#8595;</div>
<script>
!function(){
  var c=document.getElementById('c'),x=c.getContext('2d'),W,H;
  function sz(){W=c.width=innerWidth;H=c.height=innerHeight}sz();
  addEventListener('resize',sz);
  var N=65,P=[];
  for(var i=0;i<N;i++)P.push({x:Math.random()*W,y:Math.random()*H,
    vx:(Math.random()-.5)*.32,vy:(Math.random()-.5)*.32,r:Math.random()*1.6+.5});
  function f(){
    x.clearRect(0,0,W,H);
    for(var i=0;i<N;i++){var p=P[i];p.x+=p.vx;p.y+=p.vy;
      if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1}
    x.strokeStyle='rgba(250,247,240,.06)';x.lineWidth=.5;
    for(var i=0;i<N;i++)for(var j=i+1;j<N;j++){
      var dx=P[i].x-P[j].x,dy=P[i].y-P[j].y,d2=dx*dx+dy*dy;
      if(d2<20000){x.globalAlpha=1-Math.sqrt(d2)/141;
        x.beginPath();x.moveTo(P[i].x,P[i].y);x.lineTo(P[j].x,P[j].y);x.stroke()}}
    x.globalAlpha=1;x.fillStyle='rgba(250,247,240,.3)';
    for(var i=0;i<N;i++){var p=P[i];x.beginPath();x.arc(p.x,p.y,p.r,0,6.283);x.fill()}
    requestAnimationFrame(f)}f();
}();
</script></body></html>"""


def _render_hero() -> None:
    components.html(_HERO_HTML, height=620, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Stats bar
# ─────────────────────────────────────────────────────────────────────────────

def _render_stats_bar() -> None:
    st.markdown(
        """
        <div class="dte-stats dte-au">
          <div class="dte-stat dte-au dte-au-1">
            <div class="dte-stat-n">10+</div>
            <div class="dte-stat-l">Industries</div>
          </div>
          <div class="dte-stat dte-au dte-au-2">
            <div class="dte-stat-n">3</div>
            <div class="dte-stat-l">Proven Systems</div>
          </div>
          <div class="dte-stat dte-au dte-au-3">
            <div class="dte-stat-n">&lt; 5 min</div>
            <div class="dte-stat-l">Adaptation Time</div>
          </div>
          <div class="dte-stat dte-au dte-au-4">
            <div class="dte-stat-n">24/7</div>
            <div class="dte-stat-l">Real-Time API</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Industries
# ─────────────────────────────────────────────────────────────────────────────

_INDUSTRIES = [
    ("⚗️", "Chemicals", "Reactor temperature, selectivity &amp; yield optimisation"),
    ("⚡", "Energy", "Load-following, efficiency &amp; emissions reduction"),
    ("💊", "Pharma", "Batch consistency, CQA prediction &amp; deviation prevention"),
    ("💧", "Water &amp; Utilities", "Treatment optimisation, dosing &amp; demand forecast"),
    ("🌾", "Food &amp; Beverage", "Evaporation, drying &amp; fermentation control"),
    ("🔩", "Metals &amp; Materials", "Furnace dynamics, rolling mill &amp; alloy quality"),
    ("🏭", "Manufacturing", "Thermal, fluid &amp; mechanical process twins"),
    ("🌲", "Pulp &amp; Paper", "Digester, bleaching &amp; machine-section control"),
    ("⛏️", "Mining", "Flotation, leaching &amp; comminution optimisation"),
    ("🛢️", "Oil &amp; Gas", "Separator, compressor &amp; pipeline network twins"),
]


def _render_industries() -> None:
    st.markdown("<h2 class='dte-stitle'>One Engine, Every Industry</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-ssub'>"
        "The same physics-informed architecture adapts to any continuous process. "
        "Bring your historian data &mdash; we handle the rest."
        "</div>",
        unsafe_allow_html=True,
    )
    cards = "".join(
        f"<div class='dte-ic'>"
        f"<span class='dte-ic-i'>{icon}</span>"
        f"<div class='dte-ic-n'>{name}</div>"
        f"<div class='dte-ic-d'>{desc}</div>"
        f"</div>"
        for icon, name, desc in _INDUSTRIES
    )
    st.markdown(f"<div class='dte-ig'>{cards}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  How It Works
# ─────────────────────────────────────────────────────────────────────────────

def _render_how_it_works() -> None:
    st.markdown("<h2 class='dte-stitle'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-ssub'>"
        "From raw historian data to live uncertainty-aware forecasts in three steps."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="dte-hw">
          <div class="dte-hwc">
            <div class="dte-hwn">01</div>
            <div class="dte-hwt">Connect Your Data</div>
            <div class="dte-hwb">
              Ingest historian CSV or Parquet exports via a single CLI command.
              The engine normalises, validates, and builds a training-ready HDF5 dataset
              without touching your process network.
            </div>
          </div>
          <div class="dte-hwc">
            <div class="dte-hwn">02</div>
            <div class="dte-hwt">Adapt the Foundation Model</div>
            <div class="dte-hwb">
              A pre-trained physics-informed neural SDE is fine-tuned on your plant data
              in minutes, not weeks. Few-shot transfer learning means you need a fraction
              of the data a greenfield model would require.
            </div>
          </div>
          <div class="dte-hwc">
            <div class="dte-hwn">03</div>
            <div class="dte-hwt">Forecast &amp; Optimise Live</div>
            <div class="dte-hwb">
              The twin runs on a FastAPI service and streams probabilistic forecasts,
              constraint risk scores, and CEM-MPC recommended control sequences
              back to your DCS, SCADA, or dashboard.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Capabilities
# ─────────────────────────────────────────────────────────────────────────────

def _render_capabilities() -> None:
    st.markdown("<h2 class='dte-stitle'>What Makes It Different</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-ssub'>"
        "Not another black-box ML model. The engine embeds your process physics "
        "directly into the learning objective."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="dte-cg">
          <div class="dte-cc">
            <div class="dte-cci">🧬</div>
            <div class="dte-cct">Physics-Informed</div>
            <div class="dte-ccd">
              Mass, energy, and momentum residuals are baked into the loss function.
              The model respects thermodynamics even in unseen regimes.
            </div>
          </div>
          <div class="dte-cc">
            <div class="dte-cci">⚡</div>
            <div class="dte-cct">Few-Shot Transfer</div>
            <div class="dte-ccd">
              Pre-trained on process families, adapted to your plant with minimal data.
              No months-long modelling projects.
            </div>
          </div>
          <div class="dte-cc">
            <div class="dte-cci">📡</div>
            <div class="dte-cct">Real-Time API</div>
            <div class="dte-ccd">
              FastAPI service with probabilistic forecasts, constraint risk, and
              optimal control. Integrates with DCS, SCADA, or any dashboard.
            </div>
          </div>
          <div class="dte-cc">
            <div class="dte-cci">📊</div>
            <div class="dte-cct">Uncertainty Quantified</div>
            <div class="dte-ccd">
              Neural SDE diffusion gives calibrated confidence intervals on every
              forecast. Know when to trust the model and when to escalate.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Release / Performance overview
# ─────────────────────────────────────────────────────────────────────────────

def _render_release_overview(snapshot: dict[str, Any], runtime: UniversalDemoRuntime | None) -> None:
    st.markdown("<h2 class='dte-stitle'>Proven Performance</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-ssub'>"
        "The V1 shared checkpoint has been evaluated across multiple process unit families, "
        "including a customer adaptation pilot on a real historian dataset."
        "</div>",
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
            f"Shared checkpoint loaded from `{snapshot.get('model_path')}` "
            "— using universal-model rollouts for unit demos.",
        )
    else:
        st.warning(
            "Release checkpoint not loaded. Demos will fall back to simulator "
            "ensembles until the V1 model artifacts are available.",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive demo (business logic — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

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
        st.markdown(f"<h2 class='dte-stitle'>{demo_cfg['title']}</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='dte-ssub' style='margin-bottom:.8rem'>{demo_cfg.get('description', '')}</div>",
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
                "<div class='dte-note'>Release checkpoint unavailable — uncertainty bands "
                "come from a simulator ensemble.</div>",
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


# ─────────────────────────────────────────────────────────────────────────────
#  Customer story (reframed for marketing)
# ─────────────────────────────────────────────────────────────────────────────

def _render_customer_story(snapshot: dict[str, Any]) -> None:
    st.markdown("<h2 class='dte-stitle'>Case Study: Rapid Plant Adaptation</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-ssub'>"
        "A full onboarding and adaptation pass on a real historian export. "
        "The shared checkpoint was matched, adapted, and validated with minimal data "
        "in a single session."
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


# ─────────────────────────────────────────────────────────────────────────────
#  Flowsheet preview
# ─────────────────────────────────────────────────────────────────────────────

def _render_flowsheet_preview() -> None:
    st.markdown("<h2 class='dte-stitle'>What's Next: Multi-Unit Modelling</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-ssub'>"
        "V1 is unit-first. The flowsheet surface shows the direction &mdash; "
        "plant-section modelling across connected unit operations is the next frontier."
        "</div>",
        unsafe_allow_html=True,
    )
    preview_specs = [
        (
            "Exchanger \u2192 Reactor \u2192 Tank",
            ["Heat exchanger", "CSTR", "Storage tank"],
            "A temperature-conditioned reactor train with a downstream buffer and purge.",
        ),
        (
            "Reactor \u2192 Separator \u2192 Recycle",
            ["CSTR", "Separator", "Recycle loop"],
            "A recycle section where composition and thermal dynamics interact across the loop.",
        ),
    ]
    for title, nodes, description in preview_specs:
        st.markdown(f"**{title}**")
        st.markdown(f"<div class='dte-note'>{description}</div>", unsafe_allow_html=True)
        cells = []
        for index, node in enumerate(nodes):
            cells.append(f"<div class='dte-fln'>{node}</div>")
            if index < len(nodes) - 1:
                cells.append("<div class='dte-fla'>\u2192</div>")
        st.markdown("<div class='dte-fl'>" + "".join(cells) + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='dte-rule'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CTA footer
# ─────────────────────────────────────────────────────────────────────────────

def _render_cta_footer() -> None:
    st.markdown(
        """
        <div class="dte-cta">
          <div class="dte-cta-t">Ready to twin your plant?</div>
          <div class="dte-cta-d">
            Tell me about your process &mdash; industry, unit operations, data availability &mdash;
            and I'll show you exactly how the engine maps onto your system.
          </div>
          <a class="dte-cta-a" href="mailto:s.mohammadi.rl@gmail.com?subject=Digital%20Twin%20Engine%20enquiry">
            Get in touch &nearr;
          </a>
          <div class="dte-cta-f">No commitment. Response within 24 hours.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Page
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    demo_page_cfg = _load_demo_page_config()
    snapshot = _load_release_snapshot()
    runtime = _load_release_runtime()

    # ── Hero ──
    _render_hero()

    # ── Stats ──
    _render_stats_bar()

    # ── Industries ──
    _render_industries()

    # ── Divider ──
    st.markdown("<div class='dte-divider'></div>", unsafe_allow_html=True)

    # ── How It Works ──
    _render_how_it_works()

    # ── Divider ──
    st.markdown("<div class='dte-divider'></div>", unsafe_allow_html=True)

    # ── Capabilities ──
    _render_capabilities()

    # ── Divider ──
    st.markdown("<div class='dte-divider'></div>", unsafe_allow_html=True)

    # ── Live Demos ──
    st.markdown("<h2 class='dte-stitle'>Try It Live</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dte-ssub'>"
        "Each workspace starts from a fixed baseline policy. Select a disturbance regime "
        "and an alternate operating move, then watch the model forecast the state trajectory "
        "and constraint risk profile in real time."
        "</div>",
        unsafe_allow_html=True,
    )

    demos = demo_page_cfg.get("demos", [])
    tab_labels = [demo["title"] for demo in demos]
    extra_tabs = ["Case Study", "Roadmap"]
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

    # ── Performance ──
    st.markdown("<div class='dte-divider'></div>", unsafe_allow_html=True)
    _render_release_overview(snapshot, runtime)

    # ── CTA ──
    _render_cta_footer()


if __name__ == "__main__":
    main()
