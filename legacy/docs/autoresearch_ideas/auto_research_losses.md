# Digital Twin Engine — Losses Campaign Context

This file is injected into the autoresearch prompt for the focused
`dte/training/shared/losses.py` campaign. It is intended to steer the search toward
objective-design ideas rather than routine scalar tuning.

---

## Campaign Goal

Search for high-upside loss-function ideas that could materially improve:

- long-horizon trajectory quality
- latent-dynamics faithfulness
- physically plausible behavior
- robustness to rollout instability

This campaign is about better objectives, not simply bigger or smaller weights.

---

## File Boundary

You may modify only:

- `dte/training/shared/losses.py`

One file, one idea, minimal coherent patch.

Do not modify:

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `program.md`

---

## Repo/Architecture Facts

- `LossComputer` is generic and delegates physics terms through `PhysicsLoss`
- state losses are computed in normalized space
- physics residuals are computed in physical units
- trajectory weighting already supports a few simple schedules
- changes must remain generic across systems and not hardcode CSTR assumptions

---

## Highest-Priority Ideas

Prioritize these families in roughly this order:

1. **Derivative-aware trajectory supervision**
   - finite-difference loss on state derivatives
   - slope/velocity matching alongside trajectory matching
   - objective terms that better capture dynamics, not just state snapshots

2. **Multi-scale trajectory losses**
   - short-horizon plus long-horizon supervision in one objective
   - coarse/fine temporal consistency
   - losses that capture both local and global rollout quality

3. **Consistency losses tied to dynamics**
   - latent-state consistency proxies expressible in `LossComputer`
   - one-step vs rollout agreement objectives
   - penalties that align decoder behavior with trajectory evolution

4. **Physics-aware shaping without brute-force weights**
   - adaptive or structure-aware use of physics residuals
   - better balancing of residual contributions without mere scalar nudges
   - objectives that emphasize physically meaningful failure modes

5. **Robustness-aware reconstruction/trajectory objectives**
   - alternatives to plain Huber usage when justified by dynamics
   - objectives that reduce sensitivity to rare rollout spikes

---

## Lower-Priority Ideas

These are allowed only if they support a larger objective-design idea:

- only changing Huber delta
- only changing scalar weights
- only changing one existing weighting schedule parameter

Avoid spending experiments mainly on:

- "make trajectory weight 12 instead of 10"
- "make KL smaller"
- "make physics losses bigger"

Those are sweepable hyperparameters, not the target of this campaign.

---

## Search Heuristics

Prefer changes that:

- change what the model is rewarded for, not only how strongly
- better reflect industrial dynamics and rollout fidelity
- are mathematically interpretable
- remain generic across systems

Be cautious with changes that:

- duplicate existing losses with no distinct role
- overweight noisy finite-difference signals
- rely on state-name-specific hacks
- sneak system-specific residual assumptions into generic code

---

## Good Examples Of "Crazy But Worth Trying"

- derivative-matching loss
- multi-scale trajectory loss
- structured one-step vs rollout consistency loss
- adaptive physics residual aggregation with a principled rule
- robustness-aware dynamic loss shaping tied to temporal error structure

## Bad Examples For This Campaign

- only changing Huber delta
- only changing weight magnitudes
- only changing a trajectory schedule from linear to exponential with no broader idea

---

## Output Style Reminder

Propose one coherent objective change at a time.

If choosing between a scalar weight tweak and a better loss formulation, prefer
the better loss formulation.
