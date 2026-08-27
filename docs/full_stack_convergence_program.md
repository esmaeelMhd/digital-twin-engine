# Full-Stack Convergence Program

> **Historical planning document.** The `legacy/` migration described below
> was not carried out. Single-system `DigitalTwin` training remains a
> first-class path (see `docs/architecture.md`). Physics-informed losses
> penalise conservation residuals; they do not guarantee conservation.

This document defines the shortest credible path from the current repo to the
target platform:

- pretrained unit foundation models
- flowsheet graph composition
- mechanistic law grounding
- customer adaptation
- state correction and control-facing runtime
- deployable API and demo surfaces

This is not a backlog. It is a forced-convergence program.

## Repository Rule

The end state must be one unified system.

This repo should not preserve multiple active generations of the same idea just
because they existed at different points in development.

That means:

- one primary training path
- one primary adaptation path
- one primary flowsheet path
- one primary control/runtime path
- one primary documentation path
- one primary test surface

If an older file, config, script, doc, test, output layout, or data layout is
superseded, it should be:

- removed, if it has no future value
- moved to `legacy/`, if it is worth preserving for historical reference

The active repo surface should stay small and coherent.

Backward compatibility is not a convergence goal.

If compatibility aliases, shims, fallback file names, or legacy code paths keep
multiple repo generations alive, they should be removed instead of preserved.
During convergence, it is better to rewrite a function, script, or workflow for
the target architecture than to carry an old path forward beside it.

Surgical cleanup is expected.

If a file, helper, config, test, or script is clearly superseded inside this
repo, it should be deleted instead of left in place as clutter. If there is any
real doubt about whether it may still be useful for historical reference, move
it to `legacy/` rather than keeping it in the active surface.

## Goal

Build a physics-aware foundation stack for industrial process simulation,
optimization, and control that can be:

- pretrained across unit and process families
- composed into flowsheets and plant sections
- adapted to customer plants with limited data
- validated on rollout and control metrics
- deployed as a simulator, optimizer, and interactive digital twin

Autoresearch should only accelerate this stack after the architecture and
evaluation surfaces are complete enough to be trustworthy.

## Hard Position

If the repo keeps treating the legacy single-system path, universal path,
flowsheet path, and control path as parallel optional experiments, it will
never become the target platform.

The repo must converge on one primary architecture:

1. Process spec layer
2. Unit foundation model
3. Law modules
4. Customer adaptation layer
5. Flowsheet graph composition layer
6. State correction and estimation hooks
7. Control runtime layer
8. API, demo, and deployment layer

## Single-Path Policy

This policy applies to code, scripts, configs, docs, tests, outputs, and data.

### 1. No parallel active versions

There should not be multiple active ways to do the same thing because of repo
history.

Examples of what must eventually collapse to one path:

- multiple training entrypoints for the same platform layer
- multiple docs describing different generations of the architecture
- multiple smoke or phase runners that survive after the capability becomes a
  normal platform feature
- multiple baseline configs for the same primary use case

### 2. Legacy is not active

Anything under `legacy/` is archival only.

It must not remain part of:

- the primary docs
- the primary tests
- the primary CLI workflow
- the primary architecture narrative

Historical references are allowed, but active workflows should not depend on
legacy assets.

### 3. Deprecation should be decisive

When a new canonical path lands, the old path should not remain indefinitely as
another "temporary" option.

Preferred behavior:

1. land the new canonical path
2. move or remove the superseded path in the same change or immediately after
3. repair references
4. update tests to only validate the canonical surface
5. **Strict Artifact Cleanup:** Upon completion of any phase, all temporary configs, debug training scripts, intermediate evaluation harnesses, and ad hoc outputs specifically created for that phase (e.g., `phase0_smoke.yaml`, `train_debug_phase1.py`) must be deleted. They must not remain in the active repository.

This includes CLI aliases, saved artifact aliases, fallback config names, and
compatibility-only resolution modes.

It also includes one-off experiment files, temporary smoke scaffolding, and
intermediate helpers that only existed to bridge between repo generations.

### 4. Tests should converge too

Tests must validate the target architecture, not preserve every transitional
layer forever.

Keep:

- tests for the canonical platform behavior
- tests for compatibility only when compatibility is still intentionally supported

Remove or archive:

- tests whose only purpose is to preserve an old repo generation
- tests for deprecated scripts or deprecated config layouts

### 5. Generated state should not clutter the repo

Generated outputs and transient data should not shape the active repository
surface.

If they are useful:

- keep them in ignored paths
- or archive them under `legacy/`

They should not sit beside the active architecture as if they were source.

## Non-Negotiable Architecture Decisions

### 1. Universal is Primary; Single-System Remains First-Class

`UniversalDigitalTwin` is the primary model path.

The original plan treated `DigitalTwin` and single-system training as
deprecated and proposed moving them to `legacy/` during Phase 0. That
migration was not carried out. Single-system training remains a first-class,
tested path; the universal checkpoint is an additional shared-backbone track.

### 1.b. Residual Physics is the Standard

The Universal decoder should predict *residuals* on top of mechanistic base
laws (derived from `dte/laws/`), rather than predicting the full state
directly. Residual physics penalises conservation violations; it does not
guarantee conservation at inference.

### 2. Flowsheet is no longer experimental

Flowsheet work must move from "thin slice" status to a first-class product track.

The next goal is not yet a full flowsheet foundation checkpoint.
The next goal is a stable, composable, customer-usable flowsheet graph layer.

### 3. Control is a gate, not a feature

The system is not "good enough" because validation loss improved.
Every major milestone must clear rollout and control-facing acceptance gates.

### 4. Customer adaptation is canonical

Customer onboarding, family matching, adapters, calibration, and validation
must become one supported path, not a loose collection of scripts.

### 5. Docs and config sprawl stop now

No new active configs, docs, or scripts should be added unless they fit the
current taxonomy and primary architecture.

## Program Shape

Target duration: 42 calendar days

This is already aggressive.
If one person is doing all of it serially, it will take longer.
A 42-day target only works if implementation is run as parallel streams with
hard weekly gates.

## Exit Condition

The program is complete only when the repo has all of the following:

- one canonical unit foundation training path
- one canonical customer adaptation path
- one canonical flowsheet assembly and rollout path
- one canonical control-facing validation path
- one API/demo path that exposes the current architecture instead of legacy shortcuts
- one benchmark suite that decides whether the platform is actually ready

## What Must Stop

These are active sources of drift and should be treated as anti-goals during the program:

- extending the single-system path except for compatibility or bug fixes
- creating new one-off moonshot configs or ad hoc experiment trees
- adding UI/demo features before backend evaluation gates exist
- doing naming refactors without capability changes
- adding new legacy-style scripts instead of strengthening the primary path
- treating val loss as the main success metric
- keeping multiple active repo generations alive in parallel
- preserving transitional phase/smoke/experiment surfaces after a canonical
  replacement exists

## Workstreams

The program should run as four parallel workstreams with one integration owner:

### Stream A: Unit Foundation

Scope:

- universal data
- universal model (using MoE or regime-conditioned routing for diverse physics)
- stiffness-aware ODE/SDE solvers for fast kinetic regimes
- mechanistic base + residual prediction decoder
- universal training
- transfer and adaptation benchmarking

Primary code areas:

- `dte/data/datasets/universal_unit_dataset.py`
- `dte/models/universal/`
- `dte/training/universal/`
- `configs/generation_phase1_regime.yaml`
- `configs/training_universal_phase1_regime*.yaml`
- `scripts/train_universal.py`
- `scripts/evaluate_universal.py`

### Stream B: Customer Adaptation

Scope:

- schema mapping
- template and family selection
- adapter tuning
- parameter calibration
- validation reporting

Primary code areas:

- `dte/customer/`
- `dte/calibration/`
- `dte/api/onboarding.py`
- customer reporting and validation surfaces

### Stream C: Flowsheet Composition

Scope:

- unit graph assembly
- stream semantics
- recycle handling
- flowsheet rollout
- flowsheet evaluation

Primary code areas:

- `dte/flowsheet/`
- `dte/models/flowsheet/`
- `dte/training/flowsheet/`
- `scripts/phases/smoke_phase3.py`

### Stream D: Control Runtime and Deployment

Scope:

- state correction
- model-backed and simulator-backed MPC-facing runtime
- uncertainty-quantification (UQ) penalized control cost functions to prevent MPC exploitation
- RL-facing environment wrapper
- API and demo cutover to current architecture

Primary code areas:

- `dte/control/`
- `dte/evaluation/control_metrics.py`
- `scripts/phases/smoke_phase7.py`
- `dte/api/service.py`
- `app/demo_app.py`
- `frontend/`

## Phase Plan

## Phase 0: Lock The Target

Duration: 2 days

Goal:

- freeze the target architecture
- freeze the primary evaluation gates
- freeze the canonical training and deployment paths

Required outputs:

- architecture docs reflect target stack
- config taxonomy is explicit
- legacy path is identified as compatibility-only
- milestone scorecard is written down

Acceptance gate:

- there is no ambiguity about which model path, adaptation path, and control path are primary
- all transitional Phase 0 planning artifacts are removed from the active tree.

## Phase 1: Unit Foundation V1

Duration: 10 days

Goal:

- make universal pretraining the center of the repo

Required outcomes:

- regime corpus generation is reproducible
- regime training configs are clean and benchmarked
- universal evaluation reports rollout quality, transfer quality, and uncertainty behavior
- unit-family descriptors and state-role conditioning are first-class and validated

Must build:

- larger and better-curated regime corpus
- stronger universal evaluation harness
- canonical benchmark config for a full baseline
- few-shot transfer benchmark against new unit variants

Acceptance gate:

- one shared checkpoint trains reproducibly across the regime corpus
- universal adaptation beats scratch or equal-budget warm starts on target variants
- rollout stability is acceptable on held-out regime variants
- control-response fidelity is measured, not assumed

Current execution plan:

- [docs/phase1_completion_plan.md](phase1_completion_plan.md)

## Phase 2: Customer Adaptation V1

Duration: 8 days

Goal:

- turn customer adaptation into one supported workflow

Required outcomes:

- process description maps into internal schema
- family/template retrieval is canonical
- adapter and parameter calibration are combined into one pipeline
- validation report decides whether a twin is usable

Must build:

- unit family retrieval and scoring
- adapter tuning plus calibration orchestration
- rollout and control validation report
- API/onboarding surface that produces a usable adaptation package

Acceptance gate:

- a new customer unit can be onboarded from schema plus data
- the repo produces a structured validation report
- the result can be promoted or rejected based on rollout and control metrics

## Phase 3: Flowsheet Composition V1

Duration: 10 days

Goal:

- make multi-unit plant-section modeling operational

Required outcomes:

- connected unit graphs roll out stably
- stream semantics are explicit and reusable
- recycle loops have one supported handling path
- adapted unit modules can be assembled into flowsheets

Must build:

- canonical flowsheet schema and builder path
- stream update and recycle policy
- flowsheet dataset and evaluation harness
- small-plant scenario rollouts

Acceptance gate:

- at least one nontrivial flowsheet with recycle loop runs end to end
- rollout is stable enough for scenario evaluation
- flowsheet evaluation includes constraint and mismatch summaries

## Phase 4: Control Readiness V1

Duration: 7 days

Goal:

- make the platform usable for control-facing workflows

Required outcomes:

- model-backed `ProcessMPCInterface` path is stable
- simulator-backed control baseline remains available
- RL environment wrapper is usable
- state correction hooks operate during closed loop

Must build:

- one canonical control baseline script on the new architecture
- disturbance-response benchmark
- mismatch-robustness benchmark
- closed-loop validation summary

Acceptance gate:

- model-backed control rollout works on at least one representative unit
- RL environment rollouts are stable
- state correction is integrated into the runtime path
- control metrics are part of the acceptance decision

## Phase 5: API / Demo Cutover V1

Duration: 5 days

Goal:

- expose the current architecture cleanly through product surfaces

Required outcomes:

- demo and API surfaces reflect universal, adaptation, flowsheet, and control paths
- no misleading legacy-only UI path is treated as the main product
- documentation shows the canonical end-to-end path

Must build:

- API routes for current adaptation and flowsheet scenarios
- control-facing demo hooks where appropriate
- deployment/config examples for the current architecture

Acceptance gate:

- a user can follow one path from data to adapted twin to validation to control/demo surface

## Phase 6: Final Cutover

Duration: 5 days

Goal:

- make the target architecture the repo default

Required outcomes:

- default docs point to universal-first training
- customer adaptation path is documented as first-class
- flowsheet path is documented as active, not experimental
- control runtime path is documented as the primary modern surface

Must do:

- de-emphasize or label legacy paths clearly
- remove or archive any remaining misleading intermediate program artifacts
- write one master benchmark document and one release gate checklist

Acceptance gate:

- a new contributor sees one coherent platform, not a stack of parallel experiments

## Scorecard

The program is not complete unless these metrics exist and are used:

### Rollout

- one-step error
- multi-step rollout error
- long-horizon drift
- stability / finite-rate success

### Transfer

- few-shot adaptation improvement over scratch
- adaptation data efficiency
- family-selection correctness or ranking quality

### Flowsheet

- node-level rollout fidelity
- stream consistency
- recycle convergence
- graph-level scenario robustness

### Control

- disturbance-response fidelity
- tracking cost
- control effort cost
- constraint violations
- mismatch robustness

### Product

- onboarding completion rate
- validation report completeness
- API/demo scenario coverage

## Repository-Level Deliverables

By the end of the program, the repo should have:

- one canonical universal baseline command
- one canonical customer adaptation command or orchestrated API path
- one canonical flowsheet assembly and evaluation path
- one canonical control baseline on the new runtime layer
- one benchmark suite with pass/fail criteria
- one architecture document consistent with the code
- one active config taxonomy with no duplicate historical branches in the live surface
- one active documentation surface with deprecated material moved to `legacy/`
- one active test surface centered on the canonical platform paths
- **Zero Phase Artifacts:** All temporary, phase-specific scripts, configs, and outputs (`phase1_debug`, `smoke_phase3.py`, etc.) are completely removed from the repository.

## Recommended Sequence Of Execution

If there is only one integration owner, the order should be:

1. lock architecture and scorecard (move single-system to legacy)
2. finish unit foundation (must clear stiff kinetics and transfer gates)
3. finish customer adaptation
4. finish flowsheet composition
5. finish control readiness
6. cut over API/demo/docs

**Hard Block:** Do not start Phase 3 (Flowsheet) or Phase 4 (Control) until Phase 1 (Unit Foundation) rollout metrics are strictly met. Scaling unstable nodes leads to exponential error compounding in graphs and catastrophic MPC exploitation.

Do not do flowsheet pretraining before flowsheet composition is stable.
Do not do autoresearch-first optimization before the scorecard and primary paths are in place.

## What "Full" Means Before Autoresearch

Before autoresearch becomes the main optimization engine, the repo should already have:

- the right primary architecture
- the right primary data programs
- the right acceptance metrics
- the right product surfaces
- the old parallel versions removed from the active surface

Autoresearch should improve a coherent platform.
It should not be used to search over an architecture that is still undecided.
