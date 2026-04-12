"""Shared visual theme for all Digital Twin Engine Streamlit apps."""

from __future__ import annotations

import streamlit as st


# DTE design tokens — kept in sync with demo_app.py CSS variables
DTE_COLORS = {
    "ink": "#15211C",
    "moss": "#214B45",
    "brass": "#B06A2B",
    "cream": "#FBF8F1",
    "sand": "#F0EBE0",
    "state": "#0F766E",
    "control": "#1D4ED8",
    "setpoint": "#B91C1C",
    "success": "#16a34a",
    "warning": "#D97706",
    "error": "#DC2626",
}

# Plotly chart template aligned with DTE palette
PLOTLY_TEMPLATE = "plotly_white"


def inject_theme() -> None:
    """Inject the shared DTE CSS theme into the current Streamlit page."""
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
            radial-gradient(circle at 20% 15%, rgba(176, 106, 43, 0.07), transparent 30%),
            linear-gradient(180deg, #f8f3e8 0%, #fbf8f1 55%, #f4eee1 100%);
          color: var(--dte-ink);
        }
        .block-container {
          max-width: 1400px;
          padding-top: 1.5rem;
          padding-bottom: 3rem;
        }
        h1, h2, h3 {
          font-family: Georgia, "Times New Roman", serif;
          color: var(--dte-ink);
        }
        /* Hero always stays light regardless of heading overrides */
        .dte-hero, .dte-hero * {
          color: #f5f1e7 !important;
        }
        .dte-section-title {
          font-family: Georgia, "Times New Roman", serif;
          font-size: 1.75rem;
          line-height: 1.1;
          color: var(--dte-ink);
          margin-top: 1.6rem;
          margin-bottom: 0.3rem;
        }
        .dte-section-copy {
          max-width: 52rem;
          color: rgba(21, 33, 28, 0.72);
          font-family: "Trebuchet MS", "Gill Sans", Arial, sans-serif;
          margin-bottom: 1rem;
        }
        .dte-note {
          font-size: 0.86rem;
          color: rgba(21, 33, 28, 0.65);
          font-family: "Trebuchet MS", "Gill Sans", Arial, sans-serif;
        }
        .dte-rule {
          border-top: 1px solid var(--dte-line);
          margin: 1.4rem 0 0.8rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
