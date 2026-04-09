# Phase 1 Unit Foundation Model

Date: 2026-04-09

This note documents the Phase 1 implementation that turns the current universal/unit-model stack into a richer, typed unit foundation layer while keeping the legacy paths intact.

## What Landed

- `StateChannel`, `SignalChannel`, `ParameterDescriptor`, and `TopologyPort` schemas in `dte/core/state_schema.py`
- `ProcessUnitSpec` as a backward-compatible extension of `SystemSpec`
- typed channel metadata for the default `cstr`, `heat_exchanger`, and `two_tank` configs
- optional grouped single-system encoder in `dte/models/grouped_encoder.py`
- reusable physical constraint utilities in `dte/physics/constraints.py`
- multi-horizon universal training support (`loss_weights.k_step` + `multi_horizon.k_steps`)
- universal rollout regularizers for `state_bounds` and `positivity`
- control-sensitivity and uncertainty-calibration evaluation utilities
- `scripts/evaluate_universal.py` summaries for uncertainty and local control-gain mismatch

## Backward Compatibility

- `get_system_spec(...)` still works for existing callers. It now returns `ProcessUnitSpec`, which is a subclass of `SystemSpec`.
- Existing single-system training stays on the original flat encoder unless `model.grouped_encoder.enabled: true`.
- Existing universal configs still run without changes. New loss terms default to zero when not configured.
- Existing registry/system names and script entrypoints are unchanged.

## New Spec Layer

`ProcessUnitSpec` keeps all `SystemSpec` fields and adds:

- `state_channels`
- `control_channels`
- `disturbance_channels`
- `parameter_descriptors`
- `unit_type`
- `family`
- `subtype`
- `law_tags`
- `topology_ports`
- `constraints_metadata`

If channel metadata is omitted in YAML, the spec infers a usable default from:

- `state_groups`
- decoder constraints
- control/disturbance operating ranges

## Typed Encoder Path

The single-system model now supports:

- flat encoder: legacy default
- grouped encoder: optional Phase 1 path driven by `state_groups`

Config shape:

```yaml
model:
  grouped_encoder:
    enabled: true
    group_token_dim: 64
    group_kind_dim: 8
    group_encoder_layers: 2
    group_mixer_layers: 2
```

The grouped encoder preserves the same `encode/sample` interface as the legacy `Encoder`, so `DigitalTwin` does not need a new inference API.

## Universal Training Extensions

`UniversalTrainer` now supports:

- 1-step loss: existing
- full trajectory loss: existing
- K-step teacher-forced losses: new
- state bound regularization: new
- positivity regularization: new

Config shape:

```yaml
loss_weights:
  reconstruction: 1.0
  trajectory: 1.0
  one_step: 0.5
  k_step: 0.25
  state_bounds: 0.01
  positivity: 0.0

multi_horizon:
  k_steps: [3, 5]
```

## Evaluation Additions

`scripts/evaluate_universal.py` can now report:

- empirical coverage at 1 sigma / 2 sigma
- calibration gap
- Gaussian NLL
- variance-collapse rate
- local control-sensitivity mismatch against the registered simulator

These are controlled through the universal evaluation config:

```yaml
evaluation:
  uncertainty_samples: 8
  uncertainty_batches: 2
  sensitivity_batches: 2
```

## Tests Added

- richer spec metadata tests
- grouped encoder compatibility test
- reusable constraint utility tests
- control-sensitivity and uncertainty utility tests
- universal trainer multi-horizon loss test

## Remaining Phase 1 Follow-Up

The thin-slice implementation is in place, but future iterations can still deepen:

- role-specific encoder banks beyond state-group conditioning
- tighter integration of reusable mass/energy utilities into per-system physics losses
- stronger long-horizon stochastic uncertainty propagation in the universal path
- richer topology-port usage ahead of flowsheet work
