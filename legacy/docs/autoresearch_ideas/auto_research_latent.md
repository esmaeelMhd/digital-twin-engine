# Digital Twin Engine — Latent SDE Campaign Context

This file is injected into the autoresearch prompt for the focused
`dte/models/unit/latent_sde.py` campaign. It is intentionally narrower than the
default `auto_research.md`.

---

## Campaign Goal

Search for **high-upside latent-dynamics ideas** that could materially improve
long-horizon rollout quality, stochastic stability, and generalisation.

This campaign is not for routine tuning. It is specifically for trying
architectural and mathematical ideas inside `dte/models/unit/latent_sde.py`.

The model should move closer to a **general industrial dynamics backbone**, not
just a narrowly tuned CSTR latent model.

---

## File Boundary

You may modify only:

- `dte/models/unit/latent_sde.py`

One file, one idea, minimal coherent patch.

Do not modify:

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `program.md`

---

## Repo/Architecture Facts

- `DigitalTwin` is `Encoder -> LatentSDE -> Decoder`
- `LatentSDE` currently contains:
  - `LatentDrift`
  - `LatentDiffusion`
- Both are conditioned on latent state, controls, disturbances, and params
- The architecture must remain generic across systems via `SystemSpec`
- Do not introduce system-specific branches or hardcoded CSTR assumptions
- Do not hardcode config-like physical constants, bounds, normalization values,
  or fixed dimensions into `dte/models/unit/latent_sde.py`

The autoresearch metric is still `best_val_loss`, but the intended search style
for this campaign is **bold latent architecture** rather than small tuning.

---

## Highest-Priority Ideas

Prioritise these families in roughly this order:

1. **Regime-switching or gated latent dynamics**
   - mixture-of-experts drift
   - gated residual branches
   - operating-regime-dependent latent flow

2. **Diffusion reparameterisation**
   - diffusion floor plus adaptive term
   - safer diffusion parameterisation that avoids collapse or explosion
   - structure that preserves uncertainty without destabilising training

3. **Learnable correction pathways**
   - zero-init residual drift correction
   - small corrective branch that only activates when useful
   - generic “missing physics” style latent correction, but kept disciplined

4. **Stronger conditioning pathways**
   - inject conditioning information deeper into hidden layers
   - FiLM-like or gated conditioning
   - structures that reduce forgetting of control/disturbance/param context

5. **Latent numerical robustness via architecture**
   - residual scaling
   - skip structures
   - controlled latent update parameterisations
   - mathematically motivated stabilisation, not arbitrary knobs

---

## Lower-Priority Ideas

These are allowed, but only if they clearly support a larger architectural idea:

- activation swaps by themselves
- solver swaps by themselves
- tolerance changes by themselves
- step-size changes by themselves
- tiny scalar tuning with no architectural meaning

In particular, avoid spending experiments mainly on:

- “just use a better solver”
- “just tighten solver tolerances”
- “just change GELU to SiLU”
- “just change one scalar scale”

Those can be support details inside a bigger latent-architecture change, but
they should not be the whole idea unless there is a very strong mathematical
reason.

---

## Search Heuristics

Prefer changes that:

- alter how latent dynamics are represented, gated, corrected, or conditioned
- could plausibly generalise across multiple industrial systems
- improve stability without collapsing stochasticity
- remain maintainable and coherent in one file

Be cautious with changes that:

- make diffusion too large
- destroy conditioning on disturbances or params
- secretly hardcode a system-specific assumption
- rely only on numerical solver tricks instead of improving the latent model

---

## Good Examples Of “Crazy But Worth Trying”

- two drift experts with a learned gate
- diffusion = floor + bounded adaptive component
- zero-initialised correction head added to the drift output
- hidden-layer conditioning injection beyond plain input concatenation
- gated residual branch that lets the model interpolate between linear and nonlinear latent flow

## Bad Examples For This Campaign

- only changing `dt`
- only changing solver tolerances
- only changing activations
- only changing hidden size or layer count
- only changing a scalar hyperparameter in config

---

## Output Style Reminder

Propose one coherent change at a time.

If you are choosing between a boring safe tweak and a structured latent
architecture idea, prefer the structured latent architecture idea.
