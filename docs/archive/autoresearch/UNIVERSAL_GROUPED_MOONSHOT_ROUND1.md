# Universal Grouped Moonshot Round 1

This round targets the new shared grouped-state universal backbone only.

## Goal

Improve the shared-checkpoint universal baseline trained with:
- [configs/training_universal_round1.yaml](configs/training_universal_round1.yaml)
- [scripts/train_universal.py](scripts/train_universal.py)
- [scripts/evaluate_universal.py](scripts/evaluate_universal.py)

Primary comparison metric:
- `aggregate_metric_value` in `summary.json`
- lower is better

## Baseline

Establish a fresh baseline from current code:

```bash
source .venv/bin/activate

python scripts/train_universal.py \
  --config configs/training_universal_round1.yaml \
  --output_dir outputs/autoresearch_universal_grouped_round1/baseline \
  --seed 42

python scripts/evaluate_universal.py \
  --model_path outputs/autoresearch_universal_grouped_round1/baseline/best_model.eqx \
  --config outputs/autoresearch_universal_grouped_round1/baseline/config.yaml \
  --output_dir outputs/autoresearch_universal_grouped_round1/baseline/eval
```

Baseline record:
- save `outputs/autoresearch_universal_grouped_round1/baseline/summary.json`
- save `outputs/autoresearch_universal_grouped_round1/baseline/eval/summary.json`

## Idea Backlog

Use:
- [auto_research_universal_grouped_round1_ideas.yaml](auto_research_universal_grouped_round1_ideas.yaml)

Suggested execution order:
1. `grouped_systemspec_film_conditioning`
2. `attention_pool_over_state_groups`
3. `per_group_kind_decoder_heads`
4. `control_conditioned_group_gates`
5. `grouped_masked_pretraining`
6. `contrastive_cross_system_latent_alignment`

## Manual Keep/Discard Rule

For each idea:
1. make the smallest coherent patch
2. train with `configs/training_universal_round1.yaml`
3. evaluate the resulting checkpoint
4. compare `aggregate_metric_value` against the current promoted baseline
5. keep only if it is strictly lower
6. discard and revert otherwise

## Recommended Experiment Output Layout

For idea `<idea_id>`:

```bash
outputs/autoresearch_universal_grouped_round1/runs/<idea_id>/
```

Run command:

```bash
python scripts/train_universal.py \
  --config configs/training_universal_round1.yaml \
  --output_dir outputs/autoresearch_universal_grouped_round1/runs/<idea_id> \
  --seed 42

python scripts/evaluate_universal.py \
  --model_path outputs/autoresearch_universal_grouped_round1/runs/<idea_id>/best_model.eqx \
  --config outputs/autoresearch_universal_grouped_round1/runs/<idea_id>/config.yaml \
  --output_dir outputs/autoresearch_universal_grouped_round1/runs/<idea_id>/eval
```

## Scope Guardrails

Allowed surfaces:
- [dte/models/universal/digital_twin.py](dte/models/universal/digital_twin.py)
- [dte/training/universal/trainer.py](dte/training/universal/trainer.py)
- [dte/data/datasets/universal_unit_dataset.py](dte/data/datasets/universal_unit_dataset.py)
- [configs/training_universal_round1.yaml](configs/training_universal_round1.yaml)

Do not touch in round 1:
- single-system model path
- single-system trainer
- physics modules
- simulator implementations
- autoresearch harness

## Why This Round Exists

The repo now has:
- one shared checkpoint across all systems
- typed state groups
- grouped universal encode/decode

Round 1 should exploit those changes locally before moving to bigger universal ideas
like graph hypernetworks or diffusion.
