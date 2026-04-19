# Digital Twin Engine — Trainer Campaign Context

This file is injected into the autoresearch prompt for the focused
`dte/training/unit/trainer.py` campaign. It is narrower than the default
`auto_research.md` and is meant to steer the search toward training-loop and
rollout-logic ideas.

---

## Campaign Goal

Search for high-upside changes in the training algorithm itself:

- how trajectories are rolled out
- how stochastic and deterministic paths interact
- how teacher forcing is used
- how latent consistency is enforced
- how robustness is learned from partial or corrupted observations

This campaign is not for routine optimizer tuning.

---

## File Boundary

You may modify only:

- `dte/training/unit/trainer.py`

One file, one idea, minimal coherent patch.

Do not modify:

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `program.md`

---

## Repo/Architecture Facts

- `Trainer.compute_loss(...)` is the core place where rollout behavior and loss
  interaction are defined
- `Trainer` already supports deterministic rollout vs stochastic SDE rollout
- curriculum and teacher-forcing annealing already exist in config
- `LossComputer` is generic and physics-aware; do not hardcode system-specific logic
- the generic `SystemSpec` / registry-driven design must be preserved

Changes here should affect actual training behavior, not just add dead helpers.

---

## Highest-Priority Ideas

Prioritize these families in roughly this order:

1. **Latent rollout consistency**
   - re-encode predicted states and align them with rollout latents
   - penalize latent drift from the encoder manifold
   - encourage coherent latent trajectories across rollout and reconstruction

2. **Deterministic/stochastic path coordination**
   - dual-path consistency between sampled and mean latent rollout
   - staged stochasticity instead of abrupt on/off behavior
   - structured agreement penalties that stabilize SDE training

3. **Teacher-forcing redesign**
   - smarter transition between teacher-forced and free rollout behavior
   - hybrid rollout objectives that better match downstream prediction quality
   - algorithmic changes to reduce exposure bias

4. **Masked or corrupted observation robustness**
   - sensor dropout in the training loop
   - masked-state robustness objectives
   - partial-observation training that encourages better latent inference

5. **Multi-timescale rollout supervision**
   - enforce consistency over short and long rollout horizons
   - combine local accuracy with long-horizon stability in a principled way

---

## Lower-Priority Ideas

These are allowed only if they support a larger training-logic idea:

- pure learning-rate schedule tweaks
- pure gradient-clip tweaks
- pure batch-size changes
- pure epoch-count changes
- pure scalar loss-weight changes

Avoid spending experiments mainly on:

- optimizer cosmetics
- standalone warmup changes
- standalone clip changes
- routine hyperparameter sweeps

---

## Search Heuristics

Prefer changes that:

- alter the algorithm used to train trajectories, not just the optimizer around it
- could generalize across multiple industrial systems
- improve early stability without neutering stochasticity
- are coherent inside one file and easy to test

Be cautious with changes that:

- make training dramatically slower per step
- duplicate loss terms without a clear role
- silently convert stochastic training back into deterministic training
- add logic that only works for one system or one sequence length

---

## Good Examples Of "Crazy But Worth Trying"

- latent manifold consistency between rollout latents and re-encoded predicted states
- mean-vs-sampled rollout agreement penalty with a schedule
- structured sensor masking during training
- algorithmic teacher-forcing redesign tied to rollout confidence or timestep
- multi-horizon supervision inside one batch

## Bad Examples For This Campaign

- only changing `peak_lr`
- only changing `gradient_clip`
- only changing a scalar loss weight
- only changing validation cadence

---

## Output Style Reminder

Propose one coherent training-logic change at a time.

If choosing between a routine optimizer tweak and a deeper rollout/training
algorithm idea, prefer the rollout/training algorithm idea.
