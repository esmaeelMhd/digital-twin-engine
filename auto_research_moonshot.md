## Moonshot Mode

This mode is for full-idea implementations, not atomic single-file experiments.

The agent may coordinate edits across multiple files inside the editable roots
defined by the active autoresearch config. It should still keep one coherent
idea per experiment and rely on `scripts/autoresearch.py` for measurement.

## Locked Files

The measurement harness and operating rules stay frozen:

- `scripts/autoresearch.py`
- `dte/autoresearch/*`
- `scripts/agent.py`
- `auto_research.md`
- `program.md`

## Repo Constraints

- Preserve the generic `SystemSpec` / `ProcessSimulator` / `PhysicsLoss` flow.
- In `dte/`, do not introduce system-specific branches or hardcoded process names.
- In `dte/models` and `dte/training`, keep config-like bounds, normalization values,
  defaults, and fixed dimensions out of the generic core.
- If a new numeric bound or structural knob matters, thread it through config or
  `SystemSpec` rather than baking it into generic modules.
- Prefer modifying existing files. New files are allowed only within editable roots.

## Search Priorities

Priority order:

1. Lower `best_val_loss`
2. Keep rollouts numerically stable
3. Preserve physical plausibility
4. Keep implementations maintainable enough to survive follow-up experiments

The interesting ideas backlog may require coordinated edits across model, loss,
and config code. In this mode that is acceptable, as long as the experiment
remains one coherent idea and the harness itself is not modified.
