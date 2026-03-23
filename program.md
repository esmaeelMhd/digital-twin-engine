# Digital Twin Engine Autoresearch Program

This repo supports an `autoresearch` workflow modeled after Karpathy's `autoresearch`, but adapted to the digital twin training stack.

## Setup

Work with the human to:

1. Pick a run tag and create a branch like `autoresearch/<tag>` from the current default branch.
2. Read the in-scope files for context:
   - `README.md` for product and architecture context.
   - `WORKFLOW.md` for the standard training/evaluation flow.
   - `configs/training_default.yaml` for the baseline model and optimizer settings.
   - `scripts/train.py` and `dte/training/trainer.py` for the experiment harness and metric generation.
   - `program.md` for the actual autonomous research rules.
3. Verify the dataset exists. At minimum, the configured `data_dir` must contain `train_data.h5`.
4. Confirm the baseline settings in `configs/autoresearch_default.yaml`.
5. Start with a baseline run before changing anything:

   ```bash
   python scripts/autoresearch.py \
     --config configs/autoresearch_default.yaml \
     --description baseline \
     --data_dir data/test/
   ```

## Experimentation

Each experiment runs through the same harness:

```bash
python scripts/autoresearch.py \
  --config configs/autoresearch_default.yaml \
  --description "short description of the idea" \
  --data_dir data/test/
```

The harness will:

- run `scripts/train.py` with a fixed wall-clock budget
- write logs and artifacts to `outputs/autoresearch/runs/<run_id>/`
- compare the configured metric to the current promoted baseline
- append the result to `outputs/autoresearch/results.tsv`
- promote the run into `outputs/autoresearch/baseline/` only if it improves the metric

## Optimization Target

- Primary metric: `best_val_loss`
- Lower is better
- Validation is forced regularly through the harness so every run produces a comparable summary

## What You Can Modify

- `configs/training_default.yaml`
- `scripts/train.py`
- `dte/models/*.py`
- `dte/training/*.py`
- Other model/training code that directly affects the learned digital twin

## What You Should Not Modify During Experiments

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `program.md`

Those files are the measurement harness. If you change them during the loop, you break comparability.

## Keep / Discard Rule

1. Commit the candidate code change.
2. Run the autoresearch harness with a concise description.
3. If the result status is `keep`, continue from that commit.
4. If the result status is `discard`, revert to the previous kept commit.
5. If the result status is `crash`, inspect the run log, fix obvious issues, and rerun. If the idea is fundamentally broken, log the crash and move on.

## Simplicity Criterion

Do not keep complexity for tiny gains. A small improvement that adds hard-to-maintain code is usually not worth it. A comparable result with simpler code is a win.

## Persistence

After the baseline is established, keep iterating until the human interrupts you. The point of this workflow is autonomous experimentation, not a single manual trial.
