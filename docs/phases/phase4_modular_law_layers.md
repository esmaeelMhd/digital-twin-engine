# Phase 4: Modular Law Layers

Date: 2026-04-10

Phase 4 adds reusable mechanistic law modules on top of the existing system-specific physics residual path. The goal is not to replace the current handwritten physics immediately, but to introduce a reusable layer that can provide:

- explicit mechanistic features
- partial mechanistic state updates
- reusable residual/constraint terms
- config-driven composition by unit

## What Landed

Core package:

- `dte/laws/base.py`
  - `LawModule`
  - `UnitLawBundle`
- `dte/laws/chemistry.py`
  - `arrhenius_rate_constant`
  - `power_law_rate`
  - `ChemistryLaw`
- `dte/laws/thermo.py`
  - `linear_heat_capacity`
  - `enthalpy_like_transform`
  - `ThermoLaw`
- `dte/laws/biology.py`
  - `monod_growth_rate`
  - `inhibition_factor`
  - `BiologyLaw`
- `dte/laws/integration.py`
  - config parsing for `laws`
  - `build_law_bundle(...)`
  - `LawAugmentedPhysicsLoss`
  - diagnostic augmentation hooks
- `dte/laws/examples.py`
  - chemistry-driven CSTR example
  - biology-driven bioreactor example

## Integration Model

Law modules are opt-in through config:

```yaml
laws:
  enabled: true
  chemistry:
    - name: primary_reaction
      kind: arrhenius_reaction
      ...
  thermo:
    - name: liquid_cp
      kind: constant_heat_capacity
      ...
```

When present:

1. `build_law_bundle(spec, system_config)` creates a `UnitLawBundle`
2. `dte/physics/registry.py` wraps the base `PhysicsLoss`
3. training and evaluation receive the extra law residual names through the existing physics interface

This preserves backward compatibility because:

- default configs do not enable law bundles
- existing physics modules still run unchanged
- `LossComputer` already supports arbitrary residual names dynamically

## Example Entry Points

Chemistry example:

- `configs/cstr_law_example.yaml`
- `dte/laws/examples.py::build_cstr_law_example_config`

Biology example:

- `configs/bioreactor_law_example.yaml`
- `dte/laws/examples.py::build_bioreactor_process_unit_spec`
- `dte/laws/examples.py::build_bioreactor_law_bundle_example`

## Example Usage

```python
import jax.numpy as jnp

from dte.laws.examples import build_cstr_law_bundle_example

spec, config, bundle = build_cstr_law_bundle_example()

state = jnp.asarray([0.8, 0.2, 330.0, 300.0], dtype=jnp.float32)
control = jnp.asarray([50.0, 300.0], dtype=jnp.float32)
disturbance = jnp.asarray([1.0, 320.0], dtype=jnp.float32)
params = jnp.asarray([100.0, 8750.0, -50000.0, 50000.0, 15.0, 0.239], dtype=jnp.float32)

features = bundle.feature_vector(state, control, disturbance, params, 0.1)
delta = bundle.mechanistic_delta(state, control, disturbance, params, 0.1)
```

The same bundle can also generate residual series for diagnostics or be attached to the registered physics path via `get_physics_loss(...)`.

## Current Scope

This is a first modular layer, not a full mechanistic process framework.

What it does now:

- reusable chemistry, thermo, and biology APIs
- config-driven law composition
- law-derived residuals through the physics registry
- chemistry and biology examples

What it does not do yet:

- direct law-feature injection into model encoders/decoders
- rigorous property packages
- large reaction-network tooling
- full process-family libraries

## Verification

Phase 4 verification currently includes:

- `tests/test_law_modules.py`
- `tests/test_physics_registry.py`

These cover:

- low-level chemistry/thermo/biology helpers
- law-bundle feature and mechanistic-delta surfaces
- chemistry-driven CSTR example
- biology-driven bioreactor example
- law-augmented physics-loss and diagnostic registry paths

## Smoke Runner

Reusable smoke coverage for this phase now lives in:

- `scripts/phases/smoke_phase4.py`

Default usage:

```bash
source .venv/bin/activate
python scripts/phases/smoke_phase4.py
```

Useful variants:

```bash
python scripts/phases/smoke_phase4.py --dry_run
python scripts/phases/smoke_phase4.py --workspace_dir outputs/phase4_smoke/manual_run
python scripts/phases/smoke_phase4.py --examples chemistry
python scripts/phases/smoke_phase4.py --n_steps 8 --batch_size 3
```
