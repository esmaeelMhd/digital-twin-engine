# Digital Twin Engine — Moonshot Campaign Context

This file is injected into the autoresearch prompt for the moonshot campaign
defined by `auto_research_claude_ideas.yaml`. It covers ideas that are
deliberately high-variance: any one of them could either become a game-changer
for using this repo as a foundational model for physical systems, or cause
complete training collapse.

This campaign is intentionally different from the incremental tuning campaigns
(latent, encoder, decoder, losses). Those search for 1–5% gains. This one
searches for 10–100% gains and accepts the corresponding risk of discard.

---

## Campaign Goal

Search for **moonshot architectural and mathematical ideas** that could
materially shift this repo from a CSTR-tuned surrogate toward a
**universal foundational model for industrial physical systems**.

The primary metric is still `best_val_loss`. Timed-out runs with a finite
`best_val_loss` are valid. Crashes or NaN losses are discards.

The intended search style is: **one bold idea per experiment, minimal patch,
run it, keep or discard, move to the next.**

---

## File Boundary

The agent may modify only the single file listed in each idea's `target_file`.
Modifiable files across this campaign are:

- `dte/models/unit/latent_sde.py`
- `dte/training/shared/losses.py`
- `dte/training/unit/trainer.py`

One file, one idea, minimal coherent patch.

Do not modify:

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `program.md`
- `auto_research.md` (base context file)

---

## Architecture Invariants

These must be preserved regardless of the idea being explored:

- The `SystemSpec` / `ProcessSimulator` / `PhysicsLoss` abstraction must remain
  intact. Do not introduce system-specific branches (`if system == "cstr"`) or
  hardcoded CSTR dimensions, normalization constants, or physical parameters
  anywhere in `dte/models/` or `dte/training/`.
- The `DigitalTwin.from_config(config, key, system_spec)` interface must remain
  callable with any registered system.
- `TrajectoryDataset.sample_batch` must remain the only data-access path in the
  training loop.
- Physics losses must remain in physical units via the `PhysicsLoss` registry.

---

## Ideas in Priority Order

### Tier 1 — Single-file, feasible within the time budget

**1. Koopman Linear Drift** (`dte/models/unit/latent_sde.py`, priority 10)

Replace the unstructured MLP drift with a Koopman-style linear map
`A(u,c)·z + B(u,c)·w`. If the encoder learns the right lifted embedding, latent
dynamics become globally linear at any given operating point. Benefits: potential
analytical prediction via matrix exponential, convex MPC formulation, stability
via eigenvalue constraints. Risk: 32-dimensional latent may be too small for a
faithful Koopman embedding; training may collapse to trivial `A ≈ 0`.

Key constraint: zero-initialise off-diagonal A; start near identity dynamics.

**2. Wasserstein / Sinkhorn KL Replacement** (`dte/training/shared/losses.py`, priority 20)

Replace `KL(q(z|x) || N(0,I))` with Sinkhorn divergence on aggregate
mini-batch posteriors. Prevents posterior collapse, allows sharper multi-modal
latent clusters for distinct operating regimes. Risk: Sinkhorn is noisy on small
batches; ε-tuning is sensitive; removes the carefully tuned KL annealing warm-up.

Key constraint: keep KL annealing weight schedule plumbing; apply it to the
Sinkhorn weight as a drop-in replacement. Fall back to standard KL if the
`sinkhorn_reg` config key is absent.

**3. Lagrange Multiplier Physics Constraints** (`dte/training/shared/losses.py`, priority 30)

Replace fixed physics weights with learned dual variables λ_mass, λ_energy that
enforce residual constraints via primal-dual gradient ascent/descent. Dual
variables increase when constraints are violated and shrink when satisfied,
automatically finding the tightest constraint the model can satisfy. Risk:
dual variables can diverge during the initial unstable epochs — must clip them.

Key constraint: clip λ ≤ 10.0; use dual_lr ≈ 1e-3; fall back to legacy fixed
weights if `physics_constraints` config block is absent.

### Tier 2 — Higher reward, higher structural risk

**4. Port-Hamiltonian Drift** (`dte/models/unit/latent_sde.py`, priority 40)

Structure the drift as `dz = (J(z) - R(z))∇H(z) dt + B(z)u dt` where:
- `J` is skew-symmetric (energy-conserving coupling between latent modes)
- `R` is positive-definite (irreversible dissipation)
- `H(z)` is a scalar Hamiltonian network
- `B(z)·u` is the actuated input channel

Dissipativity is baked in by construction. This prior generalises across every
thermodynamic system (CSTR, heat exchangers, reactors, distillation). Risk:
computing `∇H` requires `jax.grad` inside the SDE solve — potential second-order
AD instability and 5–10× slowdown.

Key constraint: parameterise J via W - Wᵀ (guaranteed skew-symmetric); R via
LᵀL with jnp.tril (guaranteed PSD). Initialise both small.

**5. Spectral Multi-Scale Drift** (`dte/models/unit/latent_sde.py`, priority 50)

Split latent_dim into slow and fast sub-spaces with separate drift networks.
Slow sub-space is deterministic (zero diffusion); fast sub-space uses the
existing stochastic SDE. Improves long-horizon accuracy by separating thermal
equilibrium dynamics from control-response transients. Risk: scale assignment
may collapse to all-fast; cross-scale coupling is hard to learn with separate
networks.

Key constraint: slow_dim_fraction = 0.5 default; make it configurable under
`model.slow_dim_fraction`. Diffusion applies to fast sub-space only.

**6. Neural CDE Control Integration** (`dte/models/unit/latent_sde.py`, priority 60)

Replace linear-interpolation control injection with proper Neural CDE:
`dz = f_θ(z) dX` where X = (t, u, d) is the control path and f_θ outputs a
`(latent_dim × path_dim)` matrix multiplying the path derivative. Handles
piecewise-constant actuators and irregular sampling correctly. Risk: Dirac-delta
path derivatives at switching points require careful numerics; overhead with no
benefit on smooth simulated data.

Key constraint: add CDE term additively on top of existing autonomous drift
so the model degrades gracefully when controls are constant.

### Tier 3 — Transformative; document here, run manually outside harness if needed

**7. Self-Supervised Masked Pre-Training** (`dte/training/unit/trainer.py`, priority 70)

Add a BERT/MAE-style pre-training phase: randomly mask 40% of trajectory
timesteps, train encoder+decoder to reconstruct masked states from unmasked
context via mean_trajectory rollout. Seeds the representation with physics-
consistent interpolation knowledge before SDE training begins. Risk: pre-trained
features may conflict with sequential SDE prediction objective.

**8. Universal Joint Training Across Systems** (`dte/training/unit/trainer.py`, priority 80)

Train a shared LatentSDE on CSTR + heat exchanger data simultaneously with
system-specific encoder/decoder heads. If successful, the latent dynamics
backbone becomes a physics foundation model. Risk: negative transfer between
systems with incompatible dynamics could make both worse.

**9. Score-Based Trajectory Diffusion** (`dte/models/unit/latent_sde.py`, priority 90)

Replace forward SDE rollout with DDPM denoising over full latent trajectory
tensors. Captures multi-modal trajectory distributions; no stiff ODE solve.
Risk: loses continuous-time formulation entirely; requires rewriting the
inference path; denoising is iterative and may be slower than a single SDE
solve. Full infrastructure change.

---

## Idea Selection Heuristics

**Prefer ideas that:**

- make a single coherent architectural change to one file
- preserve the generic `SystemSpec`-driven interface
- improve stability or accuracy without sacrificing physical plausibility
- could plausibly generalise to systems beyond CSTR

**Be cautious with ideas that:**

- compute gradients-of-gradients inside a Diffrax solve (Hessian instability)
- increase diffusion scale or remove diffusion floor (documented NaN trigger)
- weaken decoder output constraints (unphysical predictions)
- require large physics loss weight increases (documented NaN trigger)
- silently introduce CSTR-specific dimensions or assumptions

**Order of attempted experiments:**

Start with Tier 1 ideas (Koopman → Wasserstein → Lagrange) before moving to
Tier 2. Tier 3 ideas require manual multi-file patches outside the harness.

---

## Risk/Reward Reference

| ID | Reward Ceiling | Catastrophe Risk | Files |
|----|---------------|-------------------|-------|
| koopman_linear_drift | Analytical prediction, convex MPC | Trivial A≈0 minimum | latent_sde.py |
| wasserstein_kl_replacement | Multi-modal latent, no collapse | Sinkhorn noise, ε sensitivity | losses.py |
| lagrange_multiplier_physics | Auto-balanced physics constraints | Dual divergence early in training | losses.py |
| port_hamiltonian_drift | Universal dissipativity guarantee | ∇H instability, 5–10× slowdown | latent_sde.py |
| spectral_multiscale_drift | Long-horizon accuracy | Cross-scale coupling failure | latent_sde.py |
| neural_cde_drift | Correct irregular-signal handling | Delta-function numerics | latent_sde.py |
| self_supervised_pretraining | 5–10× faster SDE convergence | Pre-train/fine-tune conflict | trainer.py |
| universal_joint_training | Physics foundation model | Negative transfer | trainer.py |
| score_based_trajectory_diffusion | Multi-modal UQ, parallel inference | Full rewrite, loses continuity | latent_sde.py |

---

## Known Failure Patterns (from repo history)

These are the documented failure modes in `auto_research.md` and `AGENTS.md`.
They apply to this campaign with extra force because the ideas here are more
aggressive:

- **Latent variance instability**: diffusion scale too large or floor removed
  → numerical explosion in Diffrax SDE solve
- **Physics loss NaN**: large residual weights with poor initial predictions
  → unbounded gradients; Lagrange dual must be clipped
- **Decoder constraint weakening**: removing `softplus` / `sigmoid_range`
  → unphysical Ca < 0 or T outside physical range
- **`eqx.field(static=True)` on JAX arrays**: breaks gradient flow silently
- **Python list indexing on JAX arrays**: use `jnp.array(idx_list)` instead
- **`LossComputer` constructor**: signature is
  `LossComputer(config, normalization_stats, physics_loss, state_names)`;
  `config` must be the **full** config dict with `loss_weights` at top level
- **HDF5 key**: dataset time key is `"time"` not `"t"`

---

## Output Style Reminder

Propose one coherent idea per experiment.

If choosing between a safe incremental tweak and a structured moonshot change,
this campaign prefers the **moonshot**. The incremental campaigns (latent,
encoder, decoder, losses) cover the safe ground.

A discard result here is expected and acceptable. The goal is to find the one
idea that delivers a step-change improvement in the physical realism and
generalisability of the latent dynamics model.
