# Digital Twin Engine — GPT Moonshot Ideas

This file provides context for the moonshot ideas defined in `autoresearch_ideas_gpt.yaml`.
These are deliberately high-variance experiments that could either move this repository
toward a true **foundational model for physical systems** or make training materially worse.

The ideas focus on cross-system transfer, stronger physical inductive bias, and more
general-purpose latent representations rather than incremental loss tuning.

---

## Campaign Goal

Find architectural or mathematical breakthroughs that enable:
- **Cross-system transfer** with minimal fine-tuning
- **Shared latent structure** across multiple physical processes
- **Better invariance** to units, operating scales, and topology changes
- **Long-horizon stability** without relying only on manually tuned penalties

Primary success metric remains `best_val_loss`. Experiments that timeout with a finite
loss are still useful. NaN losses, collapse, or clearly worse transfer behavior are discards.

---

## Ideas Overview

**Tier 1** (priorities 10-40): Single-file ideas suitable for the autoresearch harness.
These focus on `dte/models/encoder.py`, `dte/models/latent_sde.py`, and `dte/training/losses.py`.

**Tier 2** (priorities 50-70): Higher-reward structural ideas that are still plausible
within the repo's current training architecture.

**Tier 3** (priorities 80-100): Foundational-model bets that likely require multi-file
changes or multi-system data plumbing. They are included here so the campaign has a
clear long-range direction.

---

## Core Constraints (Must Be Respected)

- **SystemSpec abstraction** must remain intact. No hardcoded dimensions or per-system branching in model code.
- All changes must be **config-gated** with a safe fallback to current behavior.
- `DigitalTwin.from_config()` and `DigitalTwin.load()` must continue to work.
- Physics losses should stay compatible with the existing `PhysicsLoss` registry.
- Each experiment should test **one bold change at a time** so the autoresearch loop can attribute outcomes cleanly.

---

## The 10 Moonshot Ideas

### Tier 1

1. **Dimensionless / Buckingham-Pi Frontend** (`dte/models/encoder.py`)
2. **Contrastive Cross-System Latent Alignment** (`dte/training/losses.py`)
3. **Koopman + Residual Hybrid Drift** (`dte/models/latent_sde.py`)
4. **Neural CDE Control Integration** (`dte/models/latent_sde.py`)

### Tier 2

5. **Simulator + Residual Dynamics** (`dte/models/latent_sde.py`)
6. **Port-Hamiltonian Dissipative Latent Drift** (`dte/models/latent_sde.py`)
7. **Cross-System Masked Pretraining** (`dte/training/trainer.py`)

### Tier 3

8. **Universal SystemSpec-Conditioned Backbone** (`dte/models/digital_twin.py`)
9. **Compositional Process-Graph Hypernetwork** (`dte/models/digital_twin.py`)
10. **Latent Trajectory Diffusion Foundation Model** (`dte/models/latent_sde.py`)

---

## Usage

These ideas are meant to be used with:
- `autoresearch_ideas_gpt.yaml` (structured experiment definitions)
- `configs/autoresearch_gpt_stage1.yaml` (conservative starting point)
- `configs/autoresearch_gpt_stage2.yaml` (more aggressive settings for survivors)

Start with Tier 1 ideas and keep each experiment minimal. The highest-upside direction
in this set is whether the repo can learn a **shared cross-system backbone** without
catastrophic negative transfer.

See `autoresearch_ideas_grok.yaml`, `autoresearch_ideas_grok.md`, and
`auto_research_claude_ideas.yaml` for related campaigns already in the repo.
