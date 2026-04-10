# Phase 2 Adapters And Calibration

Date: 2026-04-09

This note documents the Phase 2 implementation for unit-family conditioning, lightweight adapters, and customer-unit calibration.

## What Landed

- structured family/subtype/law-tag metadata is now propagated into `UniversalSystemMetadata`
- system YAMLs now expose structured `conditioning_tags` for:
  - `reaction_class`
  - `thermo_regime`
  - `bio_model_family`
  - `operating_regime`
- `UniversalDigitalTwin` now conditions on:
  - system embeddings
  - family embeddings
  - subtype embeddings
  - pooled law-tag embeddings
  - structured conditioning-tag embeddings
  - parameter-law summaries
- optional bottleneck residual adapters now exist for:
  - encoder features
  - latent drift / neural-CDE features
  - decoder features
- calibration tables were added for:
  - state/control/disturbance normalization offsets and scales
  - selected physical-parameter bias terms
- a reusable calibration module now exists in `dte/calibration/unit_calibration.py`
- a CLI entry point now exists in `scripts/calibrate_unit.py`

## Universal Model Changes

The shared universal model still uses one backbone across systems, but its context vector is now richer than Phase 1:

- static context:
  - `system_id`
  - `family`
  - `subtype`
  - system-level `law_tags`
  - structured conditioning tags
  - optional numeric `SystemSpec` descriptor projection
- runtime context:
  - pooled parameter-law embeddings modulated by the scaled parameter values

This conditioning is injected into:

- state-group encoding
- encoder mean/logvar heads
- decoder path
- latent drift / neural-CDE path

## Adapter Path

Adapters are configured through:

```yaml
model:
  adapters:
    enabled: true
    bottleneck_dim: 16
    residual_scale: 0.1
    encoder: true
    drift: true
    decoder: true
```

Two filter modes are now available on the universal model:

- `mode="full"`: standard shared-backbone training
- `mode="adapters"`: adapter-only fine-tuning with the shared backbone frozen

Optional calibration tables can be unfrozen alongside adapters.

## Calibration Workflow

Use the calibration CLI to adapt a pretrained universal checkpoint to a target unit:

```bash
source .venv/bin/activate

python scripts/calibrate_unit.py \
  --model_path outputs/universal_model/best_model.eqx \
  --config configs/training_universal.yaml \
  --system_config configs/cstr_default.yaml \
  --data_dir data/cstr_variant/ \
  --system_name cstr_variant \
  --output_dir outputs/cstr_variant_calibration/ \
  --trainable_mode adapters \
  --tune_normalization \
  --tune_physics_params \
  --param_indices 1,3
```

The calibration path does three things:

1. loads the pretrained multi-system checkpoint using its original metadata
2. initializes a target-only model by copying shared weights and matching named embedding rows
3. runs few-shot calibration on the target dataset with the requested trainable subset

## Verification

Targeted verification for Phase 2:

- `pytest tests/test_universal_digital_twin.py tests/test_universal_trainer_phase1.py tests/test_unit_calibration.py tests/test_simulator_registry.py -q`
- `pytest tests/test_digital_twin.py tests/test_grouped_encoder.py tests/test_phase1_process_unit_spec.py -q`
- `python -m py_compile dte/data/multi_system_dataset.py dte/models/universal_digital_twin.py dte/training/universal_trainer.py dte/calibration/unit_calibration.py scripts/calibrate_unit.py`

## Reusable Smoke Script

Use [scripts/smoke_phase2.py](/home/ismayil/digital-twin-engine/scripts/smoke_phase2.py) to rerun the small end-to-end Phase 2 matrix:

```bash
source .venv/bin/activate
python scripts/smoke_phase2.py
```

Useful overrides:

- `--workspace_dir outputs/phase2_smoke/manual_run`
- `--target_base_system heat_exchanger --target_system_name hx_variant`
- `--target_n_trajectories 12 --n_steps 20`
- `--skip_data_generation` to reuse an existing workspace
- `--dry_run` to inspect the generated configs and planned commands

## Notes

- the single-system `DigitalTwin` path is unchanged
- the old transfer utilities in `dte/training/transfer.py` still exist for single-system fine-tuning
- the new calibration path is universal-model specific
