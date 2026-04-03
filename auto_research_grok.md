# Digital Twin Engine — Grok Moonshot Ideas

This file provides context for the moonshot ideas defined in `auto_research_grok_ideas.yaml`.
These are deliberately high-variance experiments that could either transform this repository
into a true **foundational model for physical systems** or cause training collapse.

The ideas were generated to explore the extreme edges of what a physics-informed latent
neural SDE architecture can achieve.

---

## Campaign Goal

Find architectural or mathematical breakthroughs that enable:
- **Zero-shot / few-shot transfer** to new process systems
- **Universal latent representations** of industrial physics
- **Strong inductive biases** that generalize across CSTRs, heat exchangers, reactors, etc.
- **Dramatically better** long-horizon stability and uncertainty quantification

Primary success metric remains `best_val_loss`. Experiments that timeout with a finite
loss are considered valid. NaN losses or complete divergence are discards.

---

## Ideas Overview

**Tier 1** (priorities 10-40): Single-file changes suitable for the autoresearch harness.
Focus on `dte/models/latent_sde.py` with config-gated fallbacks.

**Tier 2** (priorities 50-70): More structural changes, some touching training code.

**Tier 3** (priorities 80-100): Transformative ideas that may require multi-file changes
or external dependencies. Documented for future manual exploration.

---

## Core Constraints (Must Be Respected)

- **SystemSpec abstraction** must remain intact. No hardcoded dimensions or system-specific branches in model code.
- All changes must be **config-driven** with graceful fallbacks to existing behavior.
- `DigitalTwin.from_config()` and `DigitalTwin.load()` interfaces must continue to work.
- Physics losses must stay in the `PhysicsLoss` registry system.
- Changes should be **minimal and reversible** — one bold idea per experiment.

---

## The 10 Moonshot Ideas

### Tier 1

1. **Universal SystemSpec-conditioned Backbone** (`dte/models/latent_sde.py`)
2. **Port-Hamiltonian Neural SDE** (`dte/models/latent_sde.py`)
3. **Trajectory-level Score-Based Diffusion** (`dte/models/latent_sde.py`)
4. **Differentiable Simulator Residual** (`dte/models/latent_sde.py`)

### Tier 2

5. **Manifold-Constrained Latent Space** (`dte/models/latent_sde.py`)
6. **Neural CDE + Koopman Hybrid** (`dte/models/latent_sde.py`)
7. **Self-Supervised Masked Pretraining** (`dte/training/trainer.py`)

### Tier 3

8. **Multi-Modal Text Grounding** (`dte/models/latent_sde.py`)
9. **Meta-Learned Adaptive Solver** (`dte/training/trainer.py`)
10. **LLM-Driven Symbolic Physics Discovery** (`dte/training/trainer.py`)

---

## Usage

These ideas are meant to be used with:
- `auto_research_grok_ideas.yaml` (the structured experiment definitions)
- `configs/autoresearch_grok_stage1.yaml` (conservative hyperparameters)
- `configs/autoresearch_grok_stage2.yaml` (more aggressive settings)

Run experiments using the autoresearch harness, starting with Tier 1 ideas.
Each experiment should make **one minimal change** to the specified `target_file`.

See `auto_research_claude_ideas.yaml` and `auto_research_gemini_ideas.yaml` for
similarly structured campaigns.
