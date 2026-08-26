# Config Taxonomy

This file classifies the current `configs/` surface by role.
It is intended to reduce ambiguity before any future config-path refactor.

The main rule is:

- keep active runtime, training, and tested experiment configs under `configs/`
- keep historical or one-off material under `configs/archive/` or `docs/archive/`
- do not move active config files unless scripts, docs, tests, and environment defaults are updated in the same pass

## Current Active Groups

### 1. System Specs

These define system structure, simulator parameters, normalization, decoder constraints,
and typed metadata. They are first-class runtime inputs.

- `configs/cstr_default.yaml`
- `configs/heat_exchanger_default.yaml`
- `configs/two_tank_default.yaml`
- `configs/isothermal_cstr_default.yaml`
- `configs/isothermal_cstr_slow_kinetics.yaml`
- `configs/cstr_fast_kinetics_hot_feed.yaml`
- `configs/heat_exchanger_high_ua.yaml`
- `configs/two_tank_high_throughput.yaml`
- `configs/storage_tank_default.yaml`
- `configs/storage_tank_high_holdup.yaml`
- `configs/separator_default.yaml`
- `configs/separator_sharp_split.yaml`
- `configs/bioreactor_compartment_default.yaml`
- `configs/bioreactor_compartment_high_transfer.yaml`

Status: keep active.

### 2. Core Training And Runtime Defaults

These are the main configs a normal user or script is expected to touch.

- `configs/training_default.yaml`
- `configs/training_universal.yaml`
- `configs/training_universal_baseline_fast.yaml`
- `configs/heat_exchanger_training.yaml`
- `configs/two_tank_training.yaml`
- `configs/demo_app.yaml`
- `configs/mpc_default.yaml`

Status: keep active.

### 3. Universal Corpus And Regime Experiments

These support the universal shared-checkpoint path and are still validated by tests.
They are not random leftovers. They are the repo's structured "bigger than smoke, smaller than full foundation" configs.

Generation manifests:

- `configs/generation_phase1_corpus.yaml`
- `configs/generation_phase1_regime.yaml`

Training configs:

- `configs/training_universal_phase1_regime.yaml`
- `configs/training_universal_phase1_regime_rebalanced.yaml`
- `configs/training_universal_phase1_regime_rebalanced_gpu.yaml`
- `configs/training_universal_phase1_regime_storage_focus.yaml`
- `configs/training_universal_phase1_regime_storage_dynamics.yaml`

Meaning:

- `corpus` = broadened synthetic unit corpus
- `regime` = 14-system variant-heavy universal corpus
- `rebalanced` = same corpus, changed mixed-batch weights
- `rebalanced_gpu` = same weighting idea, larger GPU budget
- `storage_focus` = only storage-family upweighting
- `storage_dynamics` = targeted derivative loss on storage inventory states

Status: keep active.

### 4. Law Examples

These are small example configs for law-layer workflows.

- `configs/cstr_law_example.yaml`
- `configs/bioreactor_law_example.yaml`

Status: keep active, but treat as examples rather than primary runtime defaults.

### 5. Autoresearch Harness

These are the main autoresearch entry configs.

- `configs/autoresearch_default.yaml`
- `configs/autoresearch_stage1.yaml`
- `configs/autoresearch_stage2.yaml`

Status: keep active.

### 6. Focused Autoresearch Campaigns

These are narrower stage configs still used by the maintained autoresearch surface.

- `configs/autoresearch/autoresearch_claude_stage1.yaml`
- `configs/autoresearch/autoresearch_claude_stage2.yaml`
- `configs/autoresearch/autoresearch_decoder_stage1.yaml`
- `configs/autoresearch/autoresearch_decoder_stage2.yaml`
- `configs/autoresearch/autoresearch_digital_twin_stage1.yaml`
- `configs/autoresearch/autoresearch_digital_twin_stage2.yaml`
- `configs/autoresearch/autoresearch_encoder_stage1.yaml`
- `configs/autoresearch/autoresearch_encoder_stage2.yaml`
- `configs/autoresearch/autoresearch_gemini_stage1.yaml`
- `configs/autoresearch/autoresearch_gemini_stage2.yaml`
- `configs/autoresearch/autoresearch_latent_stage1.yaml`
- `configs/autoresearch/autoresearch_latent_stage2.yaml`
- `configs/autoresearch/autoresearch_losses_stage1.yaml`
- `configs/autoresearch/autoresearch_losses_stage2.yaml`
- `configs/autoresearch/autoresearch_trainer_stage1.yaml`
- `configs/autoresearch/autoresearch_trainer_stage2.yaml`

Status: keep active.

## Archive Group

Historical campaign notes and old experiment configs live under:

- `configs/archive/`
- `docs/archive/`

They are not part of the active validation surface. There is no `legacy/` tree in this repository.

Status: archival.

## Recommended Mental Model

If you are looking at `configs/`, read it in this order:

1. `*_default.yaml` = system specs and main simulator configs
2. `training_default.yaml` / `training_universal.yaml` = primary training entrypoints
3. `demo_app.yaml` / `mpc_default.yaml` = runtime surfaces
4. `generation_phase1_*` and `training_universal_phase1_regime*` = structured universal experiments
5. `autoresearch*.yaml` = bounded research harness configs
6. `configs/archive/` = historical material

## Future Target Layout

If the repo ever does a config-path refactor, the target structure should look like this:

```text
configs/
├── systems/
│   ├── cstr_default.yaml
│   ├── heat_exchanger_default.yaml
│   └── ...
├── training/
│   ├── training_default.yaml
│   ├── training_universal.yaml
│   ├── training_universal_baseline_fast.yaml
│   ├── heat_exchanger_training.yaml
│   └── two_tank_training.yaml
├── universal/
│   ├── generation_phase1_corpus.yaml
│   ├── generation_phase1_regime.yaml
│   ├── training_phase1_regime.yaml
│   ├── training_phase1_regime_rebalanced.yaml
│   ├── training_phase1_regime_rebalanced_gpu.yaml
│   ├── training_phase1_regime_storage_focus.yaml
│   └── training_phase1_regime_storage_dynamics.yaml
├── runtime/
│   ├── demo_app.yaml
│   └── mpc_default.yaml
├── laws/
│   ├── cstr_law_example.yaml
│   └── bioreactor_law_example.yaml
├── autoresearch/
│   ├── autoresearch_default.yaml
│   ├── autoresearch_stage1.yaml
│   ├── autoresearch_stage2.yaml
│   └── focused campaign configs...
└── legacy/
```

That layout is reasonable long-term, but it should only be done as an explicit refactor pass.
Too many current scripts, tests, env defaults, and docs still point at the flat `configs/*.yaml` paths.

## Keep / Move Rule

Use this rule for housekeeping:

- keep files in `configs/` if they are named in scripts, tests, docs, env defaults, or active workflows
- move files to `configs/archive/` only if they are historical/manual-only and no longer part of the active validation surface
- do not move regime configs just because their names look experimental; they are still part of the tested universal path
