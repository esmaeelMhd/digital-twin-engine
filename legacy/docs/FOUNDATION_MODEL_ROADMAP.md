# Foundation Model Roadmap

## Current State

- Active branch pushed: `autoresearch/foundation-cgg`
- Current remote/local head: `23a40f98095eafb6223f230472bd5f0bdb52543e`
- Current promoted foundation moonshot winner: `neural_cde_drift`
- Winning model/code commit: `6e70c70`
- Promoted aggregate metric:
  - `aggregate_relative_best_val_loss = 0.8759446184791675`

Relevant artifacts:
- Baseline metadata: [legacy/outputs/autoresearch_foundation_moonshot/baseline/metadata.json](../outputs/autoresearch_foundation_moonshot/baseline/metadata.json)
- Winning run summary: [legacy/outputs/autoresearch_foundation_moonshot/runs/20260406-111717-neural-cde-drift-manual/summary.json](../outputs/autoresearch_foundation_moonshot/runs/20260406-111717-neural-cde-drift-manual/summary.json)
- Campaign ledger: [legacy/outputs/autoresearch_foundation_moonshot/results.tsv](../outputs/autoresearch_foundation_moonshot/results.tsv)

## Next Moonshot Round

Round 2 should start from the promoted Neural CDE baseline, not from the pre-moonshot baseline.

### Setup

1. Create a fresh branch.
   ```bash
   git switch autoresearch/foundation-cgg
   git switch -c autoresearch/foundation-cgg-round2
   ```

2. Create a fresh workspace config from [legacy/configs/autoresearch_foundation_moonshot.yaml](../configs/autoresearch_foundation_moonshot.yaml).
   ```bash
   cp legacy/configs/autoresearch_foundation_moonshot.yaml \
      legacy/configs/autoresearch_foundation_moonshot_round2.yaml
   ```

3. In the new config, change only:
   - `workspace_dir`
   - optional run tags/descriptions if needed

4. Establish a fresh baseline from current code.
   ```bash
   source .venv/bin/activate
   python scripts/autoresearch.py \
     --config legacy/configs/autoresearch_foundation_moonshot_round2.yaml \
     --description baseline
   ```

### Best Round-2 Search Directions

Keep round 2 focused on local variants of the winning Neural CDE change in:
- [dte/models/unit/latent_sde.py](dte/models/unit/latent_sde.py)
- [dte/models/unit/digital_twin.py](dte/models/unit/digital_twin.py)
- [configs/training_default.yaml](configs/training_default.yaml)
- [configs/heat_exchanger_training.yaml](configs/heat_exchanger_training.yaml)
- [configs/two_tank_training.yaml](configs/two_tank_training.yaml)

Recommended experiments:

1. Add a scalar `cde_gain` initialized near `0`.
2. Add `neural_cde.hidden_dim` and `neural_cde.n_layers` instead of tying them to drift width/depth.
3. Clip or `tanh` the path derivative before the matrix multiply.
4. Try `path_terms = [dt, du, dd]` vs pure derivative scaling.
5. Delay Neural CDE activation until after curriculum warmup.
6. Add a small context-conditioned gate on the control-path term.

Do not spend round 2 on the ideas that already lost unless they are combined with the Neural CDE winner.

## Ideas That Need A New Universal Harness

This repo now has the first real shared-checkpoint universal baseline:
- [dte/data/datasets/universal_unit_dataset.py](dte/data/datasets/universal_unit_dataset.py)
- [configs/training_universal.yaml](configs/training_universal.yaml)
- [scripts/train_universal.py](scripts/train_universal.py)
- [scripts/evaluate_universal.py](scripts/evaluate_universal.py)
- [dte/models/universal/digital_twin.py](dte/models/universal/digital_twin.py)

The new universal model also includes typed grouped state semantics:
- `cstr`: concentration + thermal
- `heat_exchanger`: thermal
- `two_tank`: inventory

The bounded `scripts/autoresearch.py` harness still does not operate on one shared
checkpoint, so universal moonshots remain a manual loop for now.

These ideas are not comparable inside the current single-system bounded moonshot harness because that harness still trains one model per target system.

Ideas in this bucket:
- `universal_joint_training`
- `universal_systemspec_conditioned_backbone`
- `contrastive_cross_system_latent_alignment`

### Required New Pieces

Build a shared-checkpoint universal training path:

- `dte/data/datasets/universal_unit_dataset.py`
- `configs/training_universal.yaml`
- `configs/archive/autoresearch/autoresearch_universal.yaml`
- `scripts/train_universal.py`
- `scripts/evaluate_universal.py`

### Universal Training Requirements

- mixed batches from `cstr`, `heat_exchanger`, and `two_tank`
- `system_id` in every batch
- padding masks for state/control/disturbance dimensions
- shared latent backbone
- per-system input adapters or padded masked inputs
- per-system output heads or masked decoder behavior
- explicit system conditioning from `SystemSpec` or `system_id`

### Recommended Order

1. `universal_joint_training`
   - one shared checkpoint
   - mixed-system batches
   - no alignment loss yet

2. `universal_systemspec_conditioned_backbone`
   - add system embedding or `SystemSpec` embedding
   - condition encoder, latent drift, decoder

3. `contrastive_cross_system_latent_alignment`
   - add contrastive loss only after the universal backbone exists
   - positives should be based on operating-regime similarity or augmentations, not random cross-system pairings

### New Round-1 Grouped Universal Search

Before the broader universal ideas above, run a local search around the new grouped
shared backbone:

- [legacy/configs/training_universal_round1.yaml](../configs/training_universal_round1.yaml)
- [configs/archive/autoresearch/autoresearch_universal_grouped_round1.yaml](configs/archive/autoresearch/autoresearch_universal_grouped_round1.yaml)
- [docs/archive/autoresearch/auto_research_universal_grouped_round1_ideas.yaml](docs/archive/autoresearch/auto_research_universal_grouped_round1_ideas.yaml)
- [docs/archive/autoresearch/UNIVERSAL_GROUPED_MOONSHOT_ROUND1.md](docs/archive/autoresearch/UNIVERSAL_GROUPED_MOONSHOT_ROUND1.md)

Round-1 focus:
1. descriptor-conditioned modulation of grouped tokens
2. better pooling over grouped state tokens
3. lightweight kind-specific decoder heads
4. grouped control gating
5. optional grouped masked pretraining

### Metric For Universal Runs

Use a new baseline and new metric:
- one shared checkpoint
- evaluated across all systems
- aggregate geometric mean of per-system relative validation losses
- no independent per-system checkpoints during comparison

## Ideas That Need Separate Research Tracks

These are not “small patches” to the current latent SDE path and should not be forced into the bounded moonshot loop.

### Diffusion Family

Ideas:
- `score_based_trajectory_diffusion`
- `latent_trajectory_diffusion_foundation_model`
- `trajectory_level_score_based_diffusion`

Required work:
- add a new model family, not a patch to `LatentSDE`
- start with latent-trajectory diffusion, not raw-state diffusion
- create:
  - `dte/models/latent_trajectory_diffusion.py`
  - `scripts/train_diffusion.py`
  - `scripts/evaluate_diffusion.py`
  - `configs/training_diffusion.yaml`
- run on longer budgets, likely GPU-only

### Multi-Modal Text Grounding

Idea:
- `multi_modal_text_grounding`

Blocked until text data exists.

Needed:
- per-system and/or per-trajectory text metadata
- text or text embeddings stored alongside datasets
- a text-conditioning path into the encoder or latent core

### LLM Symbolic Physics Discovery

Idea:
- `llm_symbolic_physics_discovery`

This should be an outer-loop system:
- generate candidate residual formulas
- fit constants on train data
- validate on held-out data
- only then integrate winning formulas into [dte/physics](dte/physics)

Suggested script:
- `scripts/discover_symbolic_physics.py`

### Compositional Process Graph Hypernetwork

Idea:
- `compositional_process_graph_hypernetwork`

Needed:
- add structural graphs to system configs
- define units, ports, and edges
- graph encoder + hypernetwork that generates adapters or drift weights

This belongs in the universal-harness family, not the current single-system-per-run harness.

### Manifold-Constrained Latent Space

Idea:
- `manifold_constrained_latent_space`

This is feasible but should be run as a dedicated longer-budget campaign:
- add manifold regularization or projection in latent space
- likely modify [dte/models/unit/latent_sde.py](dte/models/unit/latent_sde.py) and training losses
- evaluate separately after the second Neural CDE-focused moonshot round

## Recommended Global Order

1. Round 2 local search around the Neural CDE winner
2. Universal harness
3. `universal_joint_training`
4. `universal_systemspec_conditioned_backbone`
5. `contrastive_cross_system_latent_alignment`
6. Diffusion track
7. Graph-hypernetwork track
8. Text-grounding track
9. Symbolic-physics track
