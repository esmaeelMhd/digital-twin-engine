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
   - `auto_research.md` for repo-specific agent guidance and recent operating knowledge.
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

When modifying `dte/` code during autoresearch:
- keep the architecture generic across systems
- in `dte/models` and `dte/training`, do not hardcode config-like constraints, normalization constants, default states, or fixed dimensions
- ordinary generic numeric algorithmic tweaks are okay
- prefer config-driven or `SystemSpec`-driven values for numeric bounds/scales

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

---

## Autonomous Agent

`scripts/agent.py` runs the loop fully automatically. It calls an LLM to propose each
change, applies the patch, runs the harness, keeps improvements, reverts failures, and
displays a Rich TUI dashboard.

### Launch commands

```bash
# Default: Gemini 3.1 Pro, 100 experiments, Rich dashboard
python scripts/agent.py

# Resume an existing branch
python scripts/agent.py --resume

# Named branch tag (creates autoresearch/<tag>)
python scripts/agent.py --tag mar26

# Cap total experiments
python scripts/agent.py --max-runs 50

# Text-only output (no Rich TUI)
python scripts/agent.py --no-dashboard

# Restrict the LLM to one file only
python scripts/agent.py --file dte/training/trainer.py

# LLM provider options
python scripts/agent.py --opus                  # Claude Opus 4.6 (32k extended thinking)
python scripts/agent.py --openai o3             # OpenAI o3
python scripts/agent.py --openai gpt-5.1        # OpenAI GPT-5.1
python scripts/agent.py --grok                  # xAI Grok 3
python scripts/agent.py --local                 # Local LM Studio at 127.0.0.1:1234
```

### Required environment variables

| Provider | Variable |
|----------|----------|
| Gemini (default) | `GEMINI_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| xAI Grok | `XAI_API_KEY` |
| Local LM Studio | none (uses localhost:1234) |

### What the agent modifies (one file per experiment)

See `configs/autoresearch_default.yaml` → `agent.modifiable_files` for the full list.
Defaults:

- `configs/training_default.yaml`
- `scripts/train.py`
- `dte/models/encoder.py`, `decoder.py`, `latent_sde.py`, `digital_twin.py`
- `dte/training/trainer.py`, `losses.py`

### What the agent must never touch

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `auto_research.md`
- `program.md`

### Output files

| File | Contents |
|------|----------|
| `agent.log` | Timestamped event log |
| `agent_state.json` | Crash-recovery state (deleted on clean exit) |
| `auto_research.md` | Repo-specific prompt context read by `scripts/agent.py` |
| `outputs/autoresearch/results.tsv` | Per-experiment ledger |
| `outputs/autoresearch/baseline/` | Promoted best model artifacts |
