# Digital Twin Engine — Decoder Campaign Context

This file is injected into the autoresearch prompt for the focused
`dte/models/decoder.py` campaign. It is intended to steer the search toward
better physical-state reconstruction structure.

---

## Campaign Goal

Search for high-upside decoder ideas that improve:

- reconstruction quality from latent states
- conditioning on controls and params
- physically plausible outputs
- long-horizon state decoding from latent rollouts

This campaign is not for routine MLP tuning.

---

## File Boundary

You may modify only:

- `dte/models/decoder.py`

One file, one idea, minimal coherent patch.

Do not modify:

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `program.md`

---

## Repo/Architecture Facts

- the decoder maps latent + params + control to physical state
- physical output constraints are already generic and driven by `SystemSpec`
- do not weaken generic physical constraints casually
- do not hardcode system-specific bounds or channel semantics

The best decoder changes should improve how latent information is translated
back into structured physical states.

---

## Highest-Priority Ideas

Prioritize these families in roughly this order:

1. **Stronger conditioning mechanisms**
   - FiLM-like conditioning from control and params
   - gated conditioning pathways
   - deeper use of operating context than plain concatenation

2. **Structured residual decoding**
   - linear plus nonlinear decoder paths
   - predict correction terms around a baseline decode
   - residual structures that help preserve stable physical mappings

3. **Latent-to-physical coupling improvements**
   - architectures that better preserve regime information
   - structured pathways for different latent subspaces or physical modes

4. **Constraint-friendly decoder architecture**
   - designs that work with the existing constraint application instead of against it
   - better pre-constraint parameterization of outputs

5. **Robust long-horizon decoding**
   - decoder structures that remain stable when latent rollouts drift
   - architectures less brittle to small latent perturbations

---

## Lower-Priority Ideas

These are allowed only if they support a larger decoder-structure idea:

- pure activation swaps
- pure width/depth changes
- simple normalization additions with no broader rationale

Avoid spending experiments mainly on:

- "just change GELU to SiLU"
- "just add LayerNorm"
- "just make the decoder deeper"

---

## Search Heuristics

Prefer changes that:

- improve how latent information is translated into physical states
- preserve generic constraint handling
- could generalize across different industrial systems
- remain coherent inside one file

Be cautious with changes that:

- weaken or bypass output constraints
- hardcode per-state assumptions
- add architecture that conflicts with generic `SystemSpec`-driven constraints

---

## Good Examples Of "Crazy But Worth Trying"

- FiLM-conditioned decoder hidden layers
- linear-plus-residual decoder path
- correction-style decode around a simple baseline map
- gated decoder branches for different operating regimes
- improved latent-to-physical coupling before constraint application

## Bad Examples For This Campaign

- only changing activation
- only changing hidden size
- only adding a normalization layer without a stronger idea

---

## Output Style Reminder

Propose one coherent decoder-structure idea at a time.

If choosing between routine MLP tuning and a stronger conditioning or residual
decoder architecture, prefer the stronger architecture.
