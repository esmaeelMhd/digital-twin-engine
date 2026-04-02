# Digital Twin Engine — DigitalTwin Campaign Context

This file is injected into the autoresearch prompt for the focused
`dte/models/digital_twin.py` campaign. It is intended to steer the search toward
full-model coupling ideas that can be expressed within `DigitalTwin` itself.

---

## Campaign Goal

Search for high-upside ideas in how the full model composes:

- encoding
- latent rollout
- decoding
- latent sampling behavior
- deterministic/stochastic interaction

This campaign is for model-level coupling ideas, not routine tuning.

---

## File Boundary

You may modify only:

- `dte/models/digital_twin.py`

One file, one idea, minimal coherent patch.

Do not modify:

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `program.md`

---

## Repo/Architecture Facts

- `DigitalTwin` composes `Encoder`, `Decoder`, and `LatentSDE`
- `from_config(...)` builds all three from `SystemSpec`
- `predict(...)` and `predict_ensemble(...)` define the forward behavior used at inference
- changes here must materially affect methods that are already used during training or inference
- do not add dead helper methods that never execute
- preserve the generic `SystemSpec` / registry-driven architecture

---

## Highest-Priority Ideas

Prioritize these families in roughly this order:

1. **Smarter latent sampling / hybrid inference**
   - structured blending between `z_mean` and sampled `z`
   - regime-aware deterministic/stochastic behavior
   - coupling that improves robustness without killing uncertainty

2. **Model-level latent/decoder/encoder coordination**
   - better interaction between encoded latent state and rollout initialization
   - architectural coupling that reduces mismatch between encode and rollout phases
   - coherent full-model pathways rather than isolated module tweaks

3. **Generic prior-like or initialization structure**
   - model-level logic that makes latent initialization more disciplined
   - safer sampling or initialization parameterization without system-specific hacks

4. **Inference-time robustness structure**
   - full-model behavior that is more stable under long-horizon rollout
   - better deterministic vs stochastic prediction behavior

5. **Foundation-model-style genericity**
   - changes that help the composed model behave more like a reusable industrial dynamics backbone
   - generic across systems, not CSTR-specific

---

## Lower-Priority Ideas

These are allowed only if they support a larger full-model coupling idea:

- pure refactors with no behavioral effect
- helper methods that are never called
- tiny scalar tweaks with no architectural meaning

Avoid spending experiments mainly on:

- cosmetic cleanup
- dead API extensions
- helper functions that do not affect `encode`, `predict`, or training-time behavior

---

## Search Heuristics

Prefer changes that:

- alter real encode-rollout-decode behavior
- improve full-model coherence
- remain generic across systems
- plausibly support future foundational industrial modeling

Be cautious with changes that:

- silently change model semantics in ways training losses do not support
- reduce uncertainty modeling to zero
- inject hardcoded system assumptions

---

## Good Examples Of "Crazy But Worth Trying"

- structured hybrid of sampled and mean latent initialization
- model-level gating between deterministic and stochastic latent use
- disciplined latent initialization logic that stabilizes the overall encode-rollout-decode chain
- stronger full-model coupling between encoded context and rollout start state

## Bad Examples For This Campaign

- dead helpers with no effect
- pure renaming or refactoring
- tiny scalar changes without architectural meaning

---

## Output Style Reminder

Propose one coherent full-model coupling idea at a time.

If choosing between a cosmetic refactor and a real encode-rollout-decode
behavior change, prefer the real behavior change.
