# Current Technical Specification

## Scope

This document summarizes the current implemented technical interfaces for the
main repo paths. It replaces older conceptual class sketches with the actual
runtime boundaries used by the codebase.

---

# 1. Core Specifications

## 1.1 Typed Channel Metadata

Implemented in `dte/core/state_schema.py`.

Primary dataclasses:

- `StateChannel`
- `SignalChannel`
- `ParameterDescriptor`
- `TopologyPort`

`StateChannel` carries semantic role and optional bounds:

```python
@dataclass
class StateChannel:
    name: str
    role: str
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    conserved_group: str | None = None
    description: str | None = None
```

Controls and disturbances use the shared `SignalChannel` type rather than
separate `ControlChannel` and `DisturbanceChannel` classes.

## 1.2 System-Level Specification

Implemented in `dte/simulators/base.py`.

Base process specification:

```python
@dataclass
class SystemSpec:
    name: str
    state_dim: int
    control_dim: int
    disturbance_dim: int
    param_dim: int
    state_names: list[str]
    control_names: list[str]
    disturbance_names: list[str]
    decoder_constraints: list[DecoderConstraint]
    normalization: NormalizationSpec
    default_initial_state: list[float]
    default_nominal_disturbance: list[float]
    control_ranges: dict[str, list[float]]
    disturbance_ranges: dict[str, list[float]]
    state_groups: list[StateGroupSpec]
```

Typed extension used across most newer paths:

```python
@dataclass
class ProcessUnitSpec(SystemSpec):
    state_channels: list[StateChannel]
    control_channels: list[SignalChannel]
    disturbance_channels: list[SignalChannel]
    parameter_descriptors: list[ParameterDescriptor]
    unit_type: str
    family: str
    subtype: str | None
    law_tags: list[str]
    conditioning_tags: dict[str, str]
    topology_ports: list[TopologyPort]
    constraints_metadata: dict[str, Any]
    law_feature_names: list[str]
    law_feature_defaults: list[float]
```

## 1.3 Flowsheet Specification

Implemented in `dte/flowsheet/schema.py`.

```python
@dataclass(frozen=True)
class StreamSpec:
    name: str
    source_unit: str
    target_unit: str
    variables: list[str]
    target_variables: list[str] | None = None
    kind: str = "process"
    delay: float | None = None
    description: str | None = None


@dataclass(frozen=True)
class FlowsheetSpec:
    name: str
    units: dict[str, ProcessUnitSpec]
    streams: list[StreamSpec]
    global_controls: list[str]
    global_disturbances: list[str]
    description: str | None = None
```

---

# 2. Simulation And Registry Layer

Implemented in:

- `dte/simulators/base.py`
- `dte/simulators/registry.py`

Canonical simulator interface:

```python
class ProcessSimulator(ABC):
    @property
    def spec(self) -> SystemSpec: ...

    def dynamics(self, t, state, control, disturbance): ...
    def simulate(self, initial_state, control_trajectory, disturbance_trajectory, t_span, dt=0.1, n_steps=1000): ...
    def steady_state(self, nominal_control, nominal_disturbance): ...
```

Registry entrypoints:

- `get_system_spec(config)`
- `get_simulator(name, config)`
- `list_systems()`

The API and training code depend on registry resolution, not on hardcoded
system-specific branching in model classes.

---

# 3. Single-System Model Path

Implemented in:

- `dte/models/encoder.py`
- `dte/models/grouped_encoder.py`
- `dte/models/decoder.py`
- `dte/models/latent_sde.py`
- `dte/models/digital_twin.py`

## 3.1 Encoder Path

There are two supported encoder choices:

- `Encoder`
- `GroupedStateEncoder`

Both are constructed from `SystemSpec`-derived dimensions and normalization.
The grouped path additionally uses state-group metadata and optional
channel/law conditioning.

Representative interfaces:

```python
class Encoder(eqx.Module):
    def encode(self, state, params, control) -> tuple[z_mean, z_logvar]: ...
    def __call__(self, state, params, control, key) -> tuple[z, z_mean, z_logvar]: ...


class GroupedStateEncoder(eqx.Module):
    def encode(self, state, params, control) -> tuple[z_mean, z_logvar]: ...
```

## 3.2 Latent Dynamics

The repo does not use a single `LatentDynamics` class. It uses `LatentSDE`,
which contains:

- `LatentDrift`
- `LatentDiffusion`

This supports deterministic mean trajectories and stochastic SDE rollouts.

## 3.3 Decoder

The decoder maps latent state plus conditioning inputs back to physical state:

```python
class Decoder(eqx.Module):
    def __call__(self, z, params, control) -> state
```

Physical constraints are enforced through decoder constraint metadata derived
from `SystemSpec`.

Uncertainty is not returned directly by the decoder. It is produced by
stochastic latent rollout and ensemble aggregation.

## 3.4 Digital Twin Wrapper

The implemented wrapper is `DigitalTwin`, not `UnitModel`.

```python
class DigitalTwin(eqx.Module):
    encoder: Encoder | GroupedStateEncoder
    decoder: Decoder
    latent_sde: LatentSDE
    simulator: ProcessSimulator | None

    @classmethod
    def from_config(cls, config, key, system_spec, system_config=None): ...

    def encode(self, state, params, control, key=None): ...
    def decode(self, z, params, control): ...
    def rollout_latent(self, ts, z0, controls, params, disturbances, key=None, stochastic=False): ...
    def predict(self, initial_state, controls, disturbances, params, ts, key): ...
    def predict_ensemble(self, initial_state, controls, disturbances, params, ts, key, n_samples=20): ...
```

---

# 4. Universal Shared-Checkpoint Path

Implemented in:

- `dte/data/multi_system_dataset.py`
- `dte/models/universal_digital_twin.py`
- `dte/training/universal_trainer.py`
- `dte/calibration/unit_calibration.py`

## 4.1 Universal Metadata

The universal path is driven by padded metadata tables in
`UniversalSystemMetadata`, including:

- normalization tables
- masks for state/control/disturbance/param dimensions
- state-group metadata
- role and channel-name ids
- family/subtype/law-tag tables
- law feature defaults

## 4.2 Universal Model

The implemented model is `UniversalDigitalTwin`.

It includes:

- shared grouped encoder and decoder subnets
- descriptor and embedding tables
- optional channel and law conditioning
- optional Neural CDE path
- residual bottleneck adapters for calibration
- normalization and parameter calibration tables

The repo also includes an explicit adapter module:

```python
class BottleneckAdapter(eqx.Module):
    def __call__(self, x, context) -> x_plus_residual
```

This is the actual calibration adapter path used today.

## 4.3 Universal Training

`UniversalTrainer` supports:

- reconstruction loss
- trajectory loss
- teacher-forced one-step loss
- configurable multi-horizon `k`-step losses
- bound and positivity penalties
- targeted derivative losses for selected state roles

This is a real implemented path and should be treated as first-class.

---

# 5. Physics And Law Layer

Implemented in:

- `dte/physics/base.py`
- `dte/physics/registry.py`
- `dte/laws/base.py`
- `dte/laws/integration.py`

## 5.1 Physics Loss Interface

```python
class PhysicsLoss(ABC):
    def compute_residuals(self, states, controls, disturbances, dt, params_batch=None): ...
    def residual_names(self) -> list[str]: ...
```

Physics implementations are registered per system and resolved dynamically.

## 5.2 Law Module Interface

The actual law interface is richer than older sketches:

```python
class LawModule(ABC):
    def feature_names(self) -> tuple[str, ...]: ...
    def residual_names(self) -> tuple[str, ...]: ...
    def feature_vector(self, state, control, disturbance, params, dt): ...
    def mechanistic_delta(self, state, control, disturbance, params, dt): ...
    def trajectory_residuals(self, states, controls, disturbances, dt, params): ...
```

Current end-to-end usage is strongest in:

- law-augmented physics losses
- law-augmented diagnostics
- law conditioning for grouped/universal encoders

Deeper direct mechanistic replacement of learned dynamics is not yet the main
runtime path.

---

# 6. Data And Training Pipelines

## 6.1 Dataset Classes

Implemented in:

- `TrajectoryDataset`
- `MultiSystemTrajectoryDataset`
- `FlowsheetTrajectoryDataset`
- `RealDataIngestion`

The repo keeps batches in physical units at dataset boundaries. Model training
normalizes internally where appropriate.

## 6.2 Training Entrypoints

Main scripts:

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/train_universal.py`
- `scripts/evaluate_universal.py`
- `scripts/adapt_customer.py`

There is no single mandatory pipeline of:

`unit training -> flowsheet training -> fine-tuning`

Instead, the repo currently supports parallel paths:

- single-system training
- universal mixed-system training
- unit-level customer calibration
- thin-slice flowsheet experiments

## 6.3 Flowsheet Training

Implemented in:

- `dte/data/flowsheet_dataset.py`
- `dte/models/flowsheet_model.py`
- `dte/training/flowsheet_trainer.py`

This is a real path, but still thin-slice:

- synthetic datasets
- small graph examples
- smoke-style workflows
- no mainline customer adaptation path

---

# 7. State Estimation And Control Interfaces

Implemented in:

- `dte/control/state_correction.py`
- `dte/control/mpc_interface.py`
- `dte/control/rl_env.py`

## 7.1 State Correction

The implemented class is `StateCorrectionHook`, not `StateEstimator`.

It supports:

- measurement filtering
- measurement assimilation
- latent re-encoding
- one-step latent prediction

## 7.2 MPC-Facing Runtime

The implemented class is `ProcessMPCInterface`, not a minimal `MPCInterface`.

It supports:

- candidate rollout
- candidate evaluation
- measurement assimilation
- random-shooting style optimization helpers
- simulator- or model-based rollouts

## 7.3 RL Environment

The implemented class is `ProcessControlEnv`.

It exposes Gymnasium-style reset/step behavior without a hard dependency on
Gymnasium.

---

# 8. Serving Surfaces

Implemented in:

- `dte/api/models.py`
- `dte/api/service.py`
- `dte/api/onboarding.py`
- `dte/demo/engine.py`
- `frontend/`
- `app/demo_app.py`
- `app/dashboard.py`

## 8.1 FastAPI Surface

The FastAPI service now exposes:

- `/health`
- `/predict`
- `/ensemble`
- `/steady_state`
- `/demo/*`
- `/onboarding/*`

This is materially broader than a simple simulation-only API.

## 8.2 Demo / Planning Surface

The repo includes:

- browser demo bootstrap and scenario APIs
- control recommendation endpoints
- scenario comparison endpoints
- customer onboarding and adaptation job handling

---

# 9. Support Boundaries

## First-Class

- single-system digital twin path
- universal shared-checkpoint path
- unit-level customer adaptation
- FastAPI inference and demo APIs
- generic state-correction and control interfaces

## Thin-Slice

- flowsheet graph training and evaluation
- direct law-driven representation learning

## Legacy / Compatibility

- older MPC/PID path in `dte/control/mpc.py` and `scripts/run_mpc.py`

---

# 10. Summary

The current repo is not best described by conceptual placeholder classes such as
`UnitModel`, `LatentDynamics`, `StateEstimator`, or `Simulator`.

The implemented technical core is:

- `ProcessUnitSpec` and `ProcessSimulator`
- `DigitalTwin`
- `UniversalDigitalTwin`
- registry-driven physics and law layers
- `StateCorrectionHook`
- `ProcessMPCInterface`
- `ProcessControlEnv`
- FastAPI and demo/onboarding serving layers

Any future changes should prefer aligning docs and contracts with these real
interfaces rather than reshaping the code to match older conceptual diagrams.
