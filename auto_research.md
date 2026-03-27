# Digital Twin Engine - Autoresearch Context

This file is read by `scripts/agent.py` and injected into the LLM prompt.
It is a compact repo briefing, not the source of truth. The main source-of-truth
files are `README.md`, `WORKFLOW.md`, `program.md`, `configs/*.yaml`, and the code.

## Repo Purpose

Digital Twin Engine is a JAX/Equinox project for learning a fast, physics-aware
surrogate model of a chemical process system. The current benchmark system is a
non-isothermal CSTR with a cooling jacket and a first-order irreversible
reaction `A -> B`.

The product goal is not just forecasting. The learned model should be useful for:

- accurate trajectory prediction
- physically plausible rollouts
- uncertainty-aware simulation
- downstream control, especially MPC

The repo and README describe this as an AI-powered digital twin that aims to be
much faster than first-principles simulation while remaining useful for control.

## Current Physical System

The current modeled plant is the CSTR defined in `configs/cstr_default.yaml`
and implemented in `dte/simulators/cstr.py`.

State vector:

- `Ca`: concentration of reactant A
- `Cb`: concentration of product B
- `T`: reactor temperature
- `Tc`: coolant jacket temperature

Controls:

- `F_in`: inlet volumetric flow rate
- `Tc_in`: coolant inlet temperature

Disturbances present in the dataset:

- `Ca_in`: inlet feed concentration
- `T_in`: inlet feed temperature

Default operating ranges from `configs/cstr_default.yaml`:

- `F_in`: `[10.0, 100.0]`
- `Tc_in`: `[280.0, 320.0]`
- `Ca_in`: `[0.5, 2.0]`
- `T_in`: `[290.0, 350.0]`

The simulator is the physical reference model. Training data is synthetic and
comes from that simulator, not from real plant data.

## Current Implemented Architecture

The broad conceptual pipeline is:

`physical state -> encoder -> latent state -> latent dynamics -> decoder -> predicted state`

That is true in the repo, but the exact implementation details matter:

- `dte/models/encoder.py`
  - encodes `state + params + control`
  - outputs `z_mean` and `z_logvar`
  - uses VAE-style reparameterization when sampling
- `dte/models/decoder.py`
  - decodes `latent + params + control`
  - constrains `Ca` and `Cb` with `softplus`
  - constrains `T` and `Tc` to `200 + 300 * sigmoid(raw)`
- `dte/models/latent_sde.py`
  - drift and diffusion are separate MLPs
  - latent dynamics are conditioned on `z + control + params`
  - disturbances are not yet part of the learned latent dynamics
- `dte/models/digital_twin.py`
  - exposes both stochastic sampling and deterministic rollout APIs

Important current-truth detail:

- training currently uses `model.latent_sde.mean_trajectory(...)` in
  `dte/training/trainer.py`
- so the optimization path is effectively deterministic during training
- the model supports stochastic sampling for ensemble prediction, but the main
  training loop is not currently using stochastic rollout samples

This distinction matters. Do not describe the current training setup as if it
is fully stochastic end-to-end.

## Current Training Setup

The baseline config lives in `configs/training_default.yaml`.

Current defaults:

- `latent_dim: 16`
- `hidden_dim: 128`
- `n_layers: 3`
- `drift_layers: 3`
- `diffusion_layers: 2`
- `diffusion_hidden_dim: 64`
- `initial_diffusion_scale: 0.1`
- `solver: euler_heun`
- `dt_ratio: 0.5`
- `batch_size: 64`
- `seq_len: 20`
- `stride: 10`
- `gradient_clip: 0.5`
- `peak_lr: 3e-4`

Current loss weights:

- reconstruction: `1.0`
- trajectory: `10.0`
- KL: `0.001`
- mass balance: `0.01`
- energy balance: `0.01`

KL annealing is implemented in `dte/training/losses.py`.

The trainer now stops early on non-finite train or validation losses and records
`failure_reason` and `non_finite_detected` in the training summary.

## What Is Actually Optimized

For autoresearch, the primary comparison metric is:

- `best_val_loss`
- lower is better

This is a bounded-run search. A run does not need to finish all epochs to be
useful. These interpretations are important:

- `timed_out: true` with finite `best_val_loss` is a valid result
- `best_val_loss: null` is usually the real failure signal
- a run that finishes but collapses to `NaN` is worse than a timed-out run that
  still improves validation

Early useful progress matters more than full convergence inside autoresearch.

## Existing Files The Agent Should Rely On

Use these as the main context sources:

- `README.md`: product purpose, architecture, pipeline, and control use case
- `WORKFLOW.md`: operational flow for data generation, training, evaluation, and autoresearch
- `program.md`: autonomous experimentation rules and keep/discard behavior
- `configs/training_default.yaml`: baseline architecture and optimizer settings
- `configs/autoresearch_default.yaml`: current bounded experiment setup
- `scripts/train.py`: CLI training harness and summary generation
- `dte/training/trainer.py`: training loop and optimizer behavior

This file should summarize those, not contradict them.

## Data Truths The Agent Should Remember

Training data is stored in HDF5 and loaded through `dte/data/dataset.py`.
The main arrays are:

- `states`: `(N, n_steps, 4)`
- `controls`: `(N, n_steps, 2)`
- `disturbances`: `(N, n_steps, 2)`
- `params`: `(N, n_params)`
- `time`: `(N, n_steps)`

Normalization statistics are also stored:

- `state_mean`, `state_std`
- `control_mean`, `control_std`
- `disturbance_mean`, `disturbance_std`
- `param_mean`, `param_std`

Important current-truth detail:

- dataset batches are stored in physical units
- the trainer normalizes predicted and true states internally before computing
  reconstruction and trajectory losses
- physics losses are computed in physical units

Do not assume the model is trained directly on pre-normalized HDF5 tensors.

## Experiment Boundaries

The autoresearch agent should usually modify only files from
`configs/autoresearch_default.yaml -> agent.modifiable_files`.

Defaults currently include:

- `configs/training_default.yaml`
- `scripts/train.py`
- `dte/models/encoder.py`
- `dte/models/decoder.py`
- `dte/models/latent_sde.py`
- `dte/models/digital_twin.py`
- `dte/training/trainer.py`
- `dte/training/losses.py`

Preferred experiment style:

- one file
- one idea
- minimal patch

Do not modify the measurement harness during search:

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `program.md`
- this file, unless the human explicitly asks for agent-context changes

## Physics And Losses

Physics consistency matters in this repo, but it is a regularizer, not the only
goal. The practical balance is:

1. improve predictive validation performance
2. keep rollouts physically plausible
3. avoid numerical instability
4. avoid adding code complexity that is hard to maintain

Mass and energy residuals are implemented in:

- `dte/physics/conservation.py`
- `dte/training/losses.py`

The most important engineering mistake to avoid is claiming a physics change is
good when it improves one residual but hurts the main validation objective or
causes instability.

## What Is In Sync With The Repo

These ideas are aligned with the current repo:

- digital twin for chemical process systems
- CSTR benchmark with states `[Ca, Cb, T, Tc]`
- encoder / latent dynamics / decoder structure
- VAE-style latent encoding
- constrained decoder outputs
- physics residual losses for mass and energy balance
- control-focused downstream motivation, especially MPC
- JAX + Equinox + Diffrax stack

## What Should Be Framed As Research Guidance, Not Current Fact

These are good ideas, but they are not current repo behavior unless implemented
explicitly in code:

- curriculum training over sequence lengths
- teacher forcing schedules
- two-phase training where encoder/decoder are pretrained separately
- disturbance-conditioned latent dynamics
- strong claims about calibrated uncertainty quality
- multi-unit foundation-model behavior
- hard target thresholds presented as already achieved

If these appear in planning text, label them as:

- research directions
- candidate experiments
- desired future behavior

Do not phrase them as if the current repo already does them.

## Search Heuristics

Prefer experiments that:

- improve stability early in training
- improve `best_val_loss` within the bounded budget
- preserve or improve physically plausible outputs
- make minimal, single-file changes
- avoid fragile complexity

Be cautious with:

- latent variance changes
- solver and timestep changes
- decoder output constraints
- large loss-weight changes
- anything likely to create NaNs or unrealistic states

## Failure Patterns Already Seen

Observed failure modes in this repo include:

- latent variance instability causing numerical explosion
- aggressive timestep or solver changes causing collapse
- decoder edits that weaken physical constraints
- loss-weight changes that produce `NaN` training
- proposals that sound reasonable conceptually but do not help bounded runs

Stable, modest improvements are better than ambitious unstable ideas.

## Current Search Context

- The default autoresearch config currently points to `data/cstr/`.
- The search is based on bounded runs, not full-convergence comparisons.
- Stable partial training is more valuable than ambitious edits that collapse numerically.

## Practical Output Expectations

- Propose exactly one experiment.
- Modify only allowed files.
- Prefer clear, repo-specific descriptions over generic ML wording.
- Avoid noisy or speculative changes when a smaller, cleaner edit would test the same idea.

## Final Practical Guidance

- Use this file as quick orientation, not as the source of truth.
- Cross-check assumptions against the config and code before proposing changes.
- Prefer repo-true language over generic ML wording.
- When in doubt, optimize for bounded-run validation quality plus stability.
