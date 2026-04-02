# Digital Twin Engine — Encoder Campaign Context

This file is injected into the autoresearch prompt for the focused
`dte/models/encoder.py` campaign. It is intended to steer the search toward
representation and posterior-geometry ideas.

---

## Campaign Goal

Search for high-upside encoder ideas that improve:

- inference of useful latent states from physical observations
- conditioning on controls and params
- posterior stability
- latent geometry for downstream rollout quality

This campaign is not for routine MLP tuning.

---

## File Boundary

You may modify only:

- `dte/models/encoder.py`

One file, one idea, minimal coherent patch.

Do not modify:

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `program.md`

---

## Repo/Architecture Facts

- the encoder maps state + params + control to `z_mean`, `z_logvar`
- normalization comes from `SystemSpec`, not from hardcoded constants
- the encoder must remain generic across systems
- better encoder structure can improve the whole digital twin without changing the decoder or latent SDE

---

## Highest-Priority Ideas

Prioritize these families in roughly this order:

1. **Deeper conditioning mechanisms**
   - FiLM-like conditioning from control and params
   - gated conditioning pathways
   - stronger use of operating context than plain input concatenation

2. **Posterior geometry improvements**
   - more disciplined mean/logvar pathway separation
   - structures that reduce unstable variance behavior
   - architectures that make latent inference smoother and more reliable

3. **Residual and multi-branch inference structure**
   - linear plus nonlinear encoder paths
   - skip or highway pathways
   - representation structures that help preserve state information

4. **Robust latent inference**
   - better behavior under partial or noisy observations
   - architectural robustness, not data-pipeline hacks

5. **Generic regime-aware encoding**
   - make latent inference depend more intelligently on controls/params
   - encourage representations that can generalize across systems and regimes

---

## Lower-Priority Ideas

These are allowed only if they support a larger representation idea:

- pure activation swaps
- pure width/depth changes
- pure latent-dim changes
- simple LayerNorm additions with no broader rationale

Avoid spending experiments mainly on:

- "just make it wider"
- "just add one normalization layer"
- "just change GELU to SiLU"

---

## Search Heuristics

Prefer changes that:

- change how the encoder uses control and parameter context
- improve posterior quality in a principled way
- preserve the generic `SystemSpec` design
- can plausibly support a future foundational industrial model

Be cautious with changes that:

- make `z_logvar` unstable
- inject hardcoded system-specific assumptions
- introduce config-like constants directly in generic model code

---

## Good Examples Of "Crazy But Worth Trying"

- FiLM-conditioned encoder hidden layers
- gated branch for control/param context
- separate structured heads for mean and log-variance
- linear-plus-nonlinear encoder pathway
- residual inference blocks designed to preserve physical state information

## Bad Examples For This Campaign

- only changing hidden size
- only changing number of layers
- only changing activation

---

## Output Style Reminder

Propose one coherent representation idea at a time.

If choosing between routine MLP tuning and a stronger conditioning/posterior
architecture, prefer the stronger conditioning/posterior architecture.
