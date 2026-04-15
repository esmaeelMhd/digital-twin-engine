# Foundation Process Modeling Platform Implementation Plan

Status note: this file is now a historical roadmap draft.
Use the live codebase and `WORKFLOW.md` as the operational source of truth.

## Purpose

This file is the execution plan for the coding agent. The agent must first inspect and understand the current repository in full, then implement the roadmap step by step without breaking the existing working system.

The long-term target is a **physics-aware foundation stack for industrial process modeling and control** with these layers:

1. **Unit foundation model** for lumped and compartmental process units
2. **Flowsheet graph model** for connecting units into plant sections and small plants
3. **Modular law layers** for chemistry, thermodynamics, and microbiology
4. **Optional transport-aware submodels** later for selected units where internal spatial structure matters

The immediate goal is **not** to jump to CFD or field/operator learning. The next practical goal is:

- strengthen the current universal multi-system latent model
- add typed process structure
- add graph/flowsheet abstractions
- support customer-specific plant sections with light fine-tuning
- make the simulator robust enough for MPC and DRL

---

## Rules for the Agent

### Rule 1: read the whole current repository first
Before changing anything, inspect and summarize:

- project structure
- training entry points
- current universal model path
- current single-system path
- dataset format
- config system
- loss functions
- uncertainty handling
- auto-research loop
- evaluation scripts
- demo / web app pieces if any

Do not start implementing until this audit is done.

### Rule 2: preserve backward compatibility
The current repo already supports:

- single-system latent models
- universal multi-system checkpointing
- control-conditioned rollouts
- uncertainty-aware latent dynamics
- physics losses per system

Do not break current training scripts or inference paths. Add new abstractions incrementally.

### Rule 3: work in thin vertical slices
Each phase must end in a usable state with:

- code compiling
- training runnable
- at least one test or smoke test
- docs updated

### Rule 4: prefer explicit process structure over generic MLP growth
Do not solve new requirements by only increasing MLP width/depth. Prefer:

- typed state groups
- unit descriptors
- topology-aware interfaces
- conservation-aware updates
- adapters

### Rule 5: optimize for control usefulness
The final simulator is meant for:

- forecasting
- MPC
- DRL
- what-if simulation
- digital twins

So evaluate not only one-step loss but also:

- rollout stability
- action sensitivity
- constraint violations
- uncertainty calibration

---

# Phase 0 — Repository Audit

## Objective
Understand the current codebase thoroughly and write a short audit before implementation.

## Required outputs
Create a markdown audit note in the repo, for example:

- `docs/repo_audit.md`

It must contain:

### A. Current architecture map
Identify files/classes/functions for:

- encoder
- decoder
- latent dynamics / latent SDE
- universal model
- dataset loading
- normalization
- losses
- rollout logic
- uncertainty path
- training loops
- evaluation
- checkpointing

### B. Data model map
Document:

- current tensor shapes
- how systems are identified
- how controls/disturbances/parameters are represented
- masking/padding behavior

### C. System abstraction map
Document the current abstractions such as:

- `SystemSpec`
- system descriptors
- config objects
- dataset metadata

### D. Technical debt list
List blockers for the roadmap, especially:

- concatenated vector-only assumptions
- no typed states
- no topology abstraction
- no stream abstraction
- no graph model
- no adapter mechanism
- no customer calibration flow

## Acceptance criteria
- audit note exists
- no functional changes yet
- agent can explain current repo in its own words in the audit

---

# Phase 1 — Stabilize and Refactor the Universal Unit Model

## Objective
Turn the current universal multi-system model into a strong, extensible **unit foundation model** for lumped systems.

## Why this phase matters
This is the layer most useful in practice for real industrial optimization and control.

## Key design move
Move from:

- one generic padded state vector

to:

- typed process states
- richer descriptors
- stronger physics constraints
- better rollout training

## Implementation tasks

### 1. Introduce typed state groups
Add a schema that classifies each state channel into roles such as:

- inventory / level
- temperature
- concentration / composition
- pressure
- flow
- actuator internal state
- biological state
- energy-like state

Possible new file:

- `core/state_schema.py`

Add a structure like:

```python
@dataclass
class StateChannel:
    name: str
    role: str
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    conserved_group: str | None = None
```

Then extend system specifications to include ordered channel metadata.

### 2. Extend SystemSpec into ProcessUnitSpec
Current `SystemSpec` appears too minimal. Introduce a richer abstraction, for example:

- `ProcessUnitSpec`

It should include:

- state channels
- control channels
- disturbance channels
- parameter descriptors
- unit type
- law tags
- optional topology ports
- normalization metadata
- constraints metadata

Possible file:

- `core/process_unit_spec.py`

### 3. Add typed encoders instead of one flat encoder only
Keep current flat fallback, but add an option for grouped encoding:

- thermal channels encoder
- concentration channels encoder
- inventory channels encoder
- pressure/flow channels encoder
- optional biological channels encoder

Then fuse them into the latent state.

This can still be MLP-based, but grouped.

### 4. Add explicit physics constraint utilities
Create reusable functions for:

- positivity penalties
- bound penalties
- mass-balance residuals
- energy-balance residuals
- monotonicity / physically meaningful direction constraints where applicable

Possible file:

- `physics/constraints.py`

Do not hardcode everything inside per-system losses.

### 5. Add multi-horizon training support
Current repo likely focuses too much on one-step or standard rollout loss. Add explicit training options for:

- 1-step
- K-step
- long rollout window

The trainer should support weighted combinations.

### 6. Add action sensitivity evaluation
For control usefulness, add metrics that evaluate whether predicted next states and key outputs change correctly under control perturbations.

Possible metrics:

- finite-difference sensitivity mismatch
- local control gain error

### 7. Improve uncertainty calibration
Preserve the current variational/stochastic path, but add diagnostics for:

- calibration
- variance collapse
- overconfidence during long rollout

## Deliverables
- richer unit spec abstraction
- typed state schema
- grouped encoder path
- reusable physics constraints
- multi-horizon training
- control sensitivity metrics

## Acceptance criteria
- existing systems still train
- at least cstr / heat_exchanger / two_tank run through the new abstractions
- universal model still works
- docs updated

---

# Phase 2 — Add Unit Adapters and Family-Level Conditioning

## Objective
Make the model reusable across many unit families and customer variants with small fine-tuning.

## Key design move
Base weights stay shared; small adapters specialize.

## Implementation tasks

### 1. Add unit-family taxonomy
Introduce a taxonomy such as:

- reactor
- thermal
- hydraulic
- separator
- bioprocess
- column_like

Each system gets:

- family
- subtype
- law tags

### 2. Add adapter layers
Implement optional lightweight adapters for:

- encoder
- latent dynamics
- decoder

This allows customer-specific tuning without retraining the entire backbone.

Possible approaches:

- residual MLP adapters
- FiLM conditioning
- low-rank adapters

### 3. Add parameter and law embeddings
Introduce embeddings for:

- reaction class
- thermodynamic regime assumptions
- biological model family
- operating regime tags

Keep them structured, not free-text.

### 4. Add calibration pipeline
Create a small calibration module for adapting a pretrained unit model to a new customer unit using limited data.

Possible file:

- `calibration/unit_calibration.py`

This should support:

- freezing backbone
- tuning adapters
- tuning normalization offsets/scales if needed
- tuning selected physics parameters

## Deliverables
- adapter architecture
- family-level conditioning
- calibration entry point

## Acceptance criteria
- can pretrain on multiple systems and adapt to one held-out variant with fewer trainable parameters
- no regression in base use case

---

# Phase 3 — Introduce Flowsheet Graph Modeling

## Objective
Add the plant/section level: connect multiple units by streams and simulate a process section or small plant.

## Why this phase matters
This is the layer closest to Aspen-style process simulation and practical industrial deployment.

## Key design move
Represent plant sections as graphs:

- node = unit
- edge = stream

## Implementation tasks

### 1. Define flowsheet schema
Create a graph schema with:

- units
- streams
- source/sink nodes
- recycle links
- utility links

Possible files:

- `flowsheet/schema.py`
- `flowsheet/types.py`

Suggested abstractions:

```python
@dataclass
class StreamSpec:
    name: str
    source_unit: str
    target_unit: str
    variables: list[str]
    delay: float | None = None

@dataclass
class FlowsheetSpec:
    units: dict[str, ProcessUnitSpec]
    streams: list[StreamSpec]
    global_controls: list[str]
    global_disturbances: list[str]
```

### 2. Add graph dataset format
Create a dataset format for small flowsheets with trajectories containing:

- per-unit states
- per-stream values
- controls
- disturbances
- topology metadata

### 3. Build a flowsheet simulator model
First version can be simple:

- shared unit foundation model updates each node
- stream message passing between nodes
- graph-level update block for consistency

Do not overcomplicate the first version.

Possible file:

- `models/flowsheet_model.py`

### 4. Support recycle loops and multi-unit rollouts
Ensure rollout logic can handle feedback connections and does not assume an acyclic graph only.

### 5. Add plant-level losses and metrics
Metrics should include:

- plant-wide mass/energy residuals where possible
- stream consistency
- unit-output consistency
- rollout stability

## Initial target demos
Implement at least 2 small flowsheets:

1. exchanger -> reactor -> tank
2. reactor -> separator -> recycle -> reactor

## Deliverables
- flowsheet schema
- graph dataset
- first graph simulator
- example flowsheets

## Acceptance criteria
- can train and rollout a small multi-unit process section
- graph simulation works with controls
- recycle loop example runs

---

# Phase 4 — Add Modular Law Layers

## Objective
Ground the simulator in reusable law modules instead of relying only on implicit learning.

## Why this phase matters
This addresses earlier data-driven simulator problems and improves transfer.

## Modules to add

### 1. Chemistry module
Support:

- stoichiometric maps
- reaction rate helpers
- heat of reaction hooks
- parameterized kinetic families

Possible file:

- `laws/chemistry.py`

### 2. Thermodynamics module
Do not attempt full Aspen-grade property packages initially.
Start with light abstractions for:

- enthalpy-like transforms
- heat-capacity approximations
- simple phase/equilibrium placeholders
- property correlations

Possible file:

- `laws/thermo.py`

### 3. Microbiology module
For treatment plants / bioprocesses support:

- growth/decay templates
- substrate uptake templates
- oxygen transfer terms
- inhibition terms

Possible file:

- `laws/biology.py`

### 4. Law interface
Each unit spec can point to law modules and parameters.

The learned model can use these as:

- residual correction targets
- explicit features
- constraint generators
- partial mechanistic update functions

## Deliverables
- chemistry law API
- thermo law API
- biology law API
- unit integration hooks

## Acceptance criteria
- at least one chemistry-driven example and one biology-driven example use the new interfaces

---

# Phase 5 — Build Customer Adaptation Workflow

## Objective
Make the system deployable to a customer who wants a model of one unit or one process section.

## Workflow design

### Step 1. Customer process specification
Create a simple schema for customer onboarding data:

- unit list
- stream list
- controls
- disturbances
- measurements
- known laws
- operating ranges

Possible file:

- `customer/onboarding_schema.py`

### Step 2. Template matching
Map the customer problem to:

- nearest unit families
- nearest flowsheet template

### Step 3. Fine-tuning pipeline
Automate:

- initialize pretrained weights
- freeze backbone optionally
- train adapters
- calibrate normalizers / selected parameters

### Step 4. Validation report
Generate a report with:

- forecast metrics
- rollout metrics
- control sensitivity metrics
- uncertainty summary
- constraints summary

## Deliverables
- customer schema
- adaptation script
- validation report generator

## Acceptance criteria
- can adapt a pretrained model to a new synthetic held-out customer variant
- report is generated automatically

---

# Phase 6 — Website / Demo App

## Objective
Create a polished demo website showing simulation and control effects for representative units and small flowsheets.

## Demo principles
The website is for:

- showing interactivity
- demonstrating future trajectory under control changes
- showing uncertainty and constraints
- making the product understandable to customers

## Suggested demos

### Demo 1 — CSTR
Controls:
- feed concentration
- coolant action
- flow rate

Show:
- temperature
- concentration
- future trajectories
- optimization target

### Demo 2 — Heat Exchanger
Controls:
- hot-side flow
- cold-side flow

Show:
- outlet temperatures
- duty
- future trajectories

### Demo 3 — Two/Three Tank System
Controls:
- inlet flow
- valve positions

Show:
- levels
- overflow risk
- control effect

### Demo 4 — Small Flowsheet
Example:
- exchanger -> reactor -> tank

Show:
- graph layout
- unit states
- stream states
- plant-wide trajectories

## Technical suggestion
If the repo uses Python backend plus web frontend, expose:

- `simulate`
- `rollout`
- `optimize_control`
- `compare_scenarios`

API endpoints.

## Deliverables
- demo backend endpoints
- simple frontend
- example configs

## Acceptance criteria
- at least 3 interactive demos work locally

---

# Phase 7 — MPC and DRL Readiness

## Objective
Make the simulator usable as an environment/model for control algorithms.

## Implementation tasks

### 1. MPC interface
Add a simple interface that exposes:

- current state estimate
- rollout under candidate actions
- cost function hook
- constraint hook

Possible file:

- `control/mpc_interface.py`

### 2. DRL environment wrapper
Provide Gymnasium-style wrappers if appropriate.

Possible file:

- `control/rl_env.py`

### 3. State correction / filtering
Add online correction hooks:

- measurement assimilation
- latent update from observations
- optional filter-based correction

### 4. Control-oriented metrics
Add automated checks for:

- closed-loop cost
- constraint violations
- sensitivity to disturbances
- robustness under mismatch

## Deliverables
- MPC wrapper
- RL env wrapper
- state correction utilities

## Acceptance criteria
- one example MPC loop runs
- one example RL environment runs

---

# Phase 8 — Later Extension: Distributed / Transport-Aware Units

## Objective
Only after the previous layers are stable, add structured distributed units.

## Important note
This is a later extension. Do not start here.

## First targets
Before any full field model, use low-dimensional distributed representations for:

- plug-flow reactor discretized into cells
- axial heat exchanger
- treatment basin zone chain
- stratified tank

This bridges lumped models and transport-aware models.

Only later consider:

- neural operators
- graph operators
- field decoders
- geometry-aware models

## Acceptance criteria
- first distributed unit can be trained and connected into flowsheet graph without breaking the rest of the stack

---

# Suggested Repository Changes

## New directories to add

- `core/`
- `flowsheet/`
- `laws/`
- `calibration/`
- `customer/`
- `control/`
- `docs/`

## Likely file additions

- `core/state_schema.py`
- `core/process_unit_spec.py`
- `physics/constraints.py`
- `flowsheet/schema.py`
- `flowsheet/types.py`
- `models/flowsheet_model.py`
- `laws/chemistry.py`
- `laws/thermo.py`
- `laws/biology.py`
- `calibration/unit_calibration.py`
- `customer/onboarding_schema.py`
- `control/mpc_interface.py`
- `control/rl_env.py`
- `docs/repo_audit.md`

---

# Prioritization Order

Implement in this exact order:

1. Phase 0 — repository audit
2. Phase 1 — strengthen universal unit model
3. Phase 2 — adapters and family conditioning
4. Phase 3 — flowsheet graph modeling
5. Phase 4 — modular law layers
6. Phase 5 — customer adaptation workflow
7. Phase 6 — website/demo app
8. Phase 7 — MPC and DRL readiness
9. Phase 8 — later transport-aware extension

Do not jump ahead unless earlier layers are stable.

---

# Definition of Done for the Whole Project

The project reaches the first major milestone when all of the following are true:

- pretrained shared unit model works across multiple unit families
- small flowsheet graph simulation works
- chemistry or biology law hooks are integrated
- customer adaptation script exists
- at least 3 website demos run
- simulator can be used by MPC and RL wrappers
- rollout stability is significantly improved compared with the old purely data-driven simulator

---

# Immediate First Tasks

The coding agent should start with these exact steps:

1. Read the full repository and create `docs/repo_audit.md`
2. Propose the minimum set of abstractions needed for `StateChannel`, `ProcessUnitSpec`, and typed state groups
3. Refactor current system specs to the richer spec format while keeping backward compatibility
4. Add grouped/typed encoder support without removing the old path
5. Add reusable physics constraints utility module
6. Add multi-horizon training and rollout metrics
7. Run smoke tests on cstr, heat_exchanger, and two_tank

Only after that should it continue to the next phases.

---

# Notes to the Agent

- Favor clean interfaces and incremental integration.
- Keep old checkpoints loadable where possible, or document migration clearly.
- Write small tests for every new abstraction.
- Prefer explicit process semantics over generic neural complexity.
- Keep the stack useful for practical plant optimization, not only academic elegance.
