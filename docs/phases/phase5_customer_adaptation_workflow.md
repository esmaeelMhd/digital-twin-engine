# Phase 5 Customer Adaptation Workflow

Date: 2026-04-10

Phase 5 adds a customer-facing layer on top of the existing universal calibration stack. The goal is to make adaptation look like one workflow instead of a set of internal research utilities.

## What Landed

Core package:

- `dte/customer/onboarding_schema.py`
  - customer onboarding schema for:
    - unit list
    - stream list
    - controls
    - disturbances
    - measurements
    - known laws
    - operating ranges
- `dte/customer/template_matching.py`
  - ranked unit-family template matching against registered unit specs
  - ranked flowsheet-template matching against Phase 3 example graphs
- `dte/customer/reporting.py`
  - forecast metrics
  - rollout metrics
  - control-sensitivity metrics
  - uncertainty summary
  - constraint summary
- `dte/customer/adaptation.py`
  - end-to-end orchestration:
    - load onboarding
    - rank templates
    - initialize pretrained weights
    - calibrate adapters / calibration tables
    - generate JSON + Markdown validation report
- `scripts/adapt_customer.py`
  - CLI entry point for the full workflow

## Workflow Shape

The unit adaptation path now looks like:

1. customer onboarding payload is loaded and validated
2. unit and flowsheet templates are ranked
3. the pretrained universal checkpoint is loaded with its original metadata
4. a target-only calibration model is initialized through Phase 2 weight transfer
5. adapters and optional calibration tables are fine-tuned on the customer dataset
6. a validation report is generated automatically

## CLI Usage

```bash
source .venv/bin/activate

python scripts/adapt_customer.py \
  --onboarding /path/to/customer_onboarding.yaml \
  --model_path outputs/universal_model/best_model.eqx \
  --config configs/training_universal.yaml \
  --system_config configs/cstr_default.yaml \
  --data_dir data/cstr_variant/ \
  --output_dir outputs/customer_cstr_variant/ \
  --trainable_mode adapters \
  --tune_normalization
```

Key outputs:

- `onboarding.json`
- `template_matches.json`
- `config.yaml`
- `summary.json`
- `validation_report.json`
- `validation_report.md`

## Report Contents

The validation report is generated from the calibrated model and includes:

- one-step physical forecast metrics
- multi-step rollout metrics
- local control-sensitivity mismatch against the matched simulator
- uncertainty calibration metrics from encoder initial-latent sampling (the universal runtime has no stochastic diffusion path)
- bound and positivity summaries from predicted trajectories

## Current Scope

What is covered end-to-end now:

- customer unit onboarding
- unit-family template matching
- automatic adapter-based unit adaptation
- automatic validation report generation

What is only partially covered:

- flowsheet onboarding and flowsheet-template matching are implemented
- flowsheet calibration/orchestration is not yet wired through `scripts/adapt_customer.py`

## Verification

Targeted verification for this phase:

- `JAX_PLATFORMS=cpu pytest tests/test_customer_workflow.py -q`
- `python -m py_compile dte/customer/__init__.py dte/customer/onboarding_schema.py dte/customer/template_matching.py dte/customer/reporting.py dte/customer/adaptation.py scripts/adapt_customer.py`

The test slice includes:

- schema loading and validation
- unit-template matching
- flowsheet-template matching
- an end-to-end synthetic customer adaptation run with automatic report generation

## Smoke Runner

Reusable end-to-end smoke coverage for this phase now lives in:

- `scripts/phases/smoke_phase5.py`

Default usage:

```bash
source .venv/bin/activate
python scripts/phases/smoke_phase5.py
```

Useful variants:

```bash
python scripts/phases/smoke_phase5.py --dry_run
python scripts/phases/smoke_phase5.py --workspace_dir outputs/phase5_smoke/manual_run
python scripts/phases/smoke_phase5.py --target_base_system heat_exchanger --target_system_name customer_hx_variant
python scripts/phases/smoke_phase5.py --skip_data_generation
python scripts/phases/smoke_phase5.py --jax_platform gpu
```

The smoke runner forces `JAX_PLATFORMS=cpu` by default so the small universal pretrain and customer adaptation path stay quiet and reproducible on ordinary dev machines.
