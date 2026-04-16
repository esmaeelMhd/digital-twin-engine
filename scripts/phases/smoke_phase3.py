"""Run a reusable Phase 3 smoke-test matrix for flowsheet graph modeling."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLOWSHEETS = ("exchanger_reactor_tank", "reactor_separator_recycle")


class SmokeError(RuntimeError):
    """Raised when the smoke runner cannot complete successfully."""


def _json_safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not (float("-inf") < numeric < float("inf")):
        return None
    return numeric


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_workspace_dir(raw_workspace: str | None) -> Path:
    if raw_workspace is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / "phase3_smoke" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_phase3_smoke_config(
    *,
    batch_size: int,
    seq_len: int,
    stride: int,
    n_epochs: int,
    max_batches_per_epoch: int,
    eval_batches: int,
) -> dict[str, Any]:
    total_steps = max(10, n_epochs * max_batches_per_epoch + 5)
    return {
        "model": {
            "hidden_dim": 32,
            "message_dim": 12,
            "family_embedding_dim": 8,
            "n_layers": 2,
            "graph_layers": 2,
            "message_passing_steps": 2,
        },
        "optimizer": {
            "peak_lr": 0.001,
            "end_lr": 0.0001,
            "warmup_steps": 1,
            "total_steps": total_steps,
            "gradient_clip": 1.0,
        },
        "training": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "stride": stride,
            "max_batches_per_epoch": max_batches_per_epoch,
        },
        "checkpointing": {
            "max_val_batches": eval_batches,
        },
        "loss_weights": {
            "trajectory": 1.0,
            "stream_consistency": 1.0,
            "unit_consistency": 0.25,
            "plant_balance": 0.1,
            "rollout_stability": 0.01,
        },
        "evaluation": {
            "eval_batches": eval_batches,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3 smoke-test matrix")
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=None,
        help="Workspace directory for generated data, configs, logs, and outputs",
    )
    parser.add_argument(
        "--flowsheets",
        nargs="+",
        default=list(DEFAULT_FLOWSHEETS),
        choices=list(DEFAULT_FLOWSHEETS),
        help="Flowsheet demos to include in the smoke run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=8,
        help="Synthetic trajectories to generate per flowsheet",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=18,
        help="Steps per generated flowsheet trajectory",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step used by the synthetic flowsheet generator",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=8,
        help="Sequence length for training/evaluation windows",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride for training/evaluation windows",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.25,
        help="Validation fraction used when splitting the generated dataset",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1,
        help="Epochs for each flowsheet smoke training run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for flowsheet smoke training/evaluation",
    )
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=2,
        help="Batch cap per epoch for flowsheet smoke training",
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=1,
        help="Validation batches evaluated per flowsheet",
    )
    parser.add_argument(
        "--jax_platforms",
        type=str,
        default="cpu",
        help="Value exported to JAX_PLATFORMS before importing JAX",
    )
    parser.add_argument(
        "--skip_data_generation",
        action="store_true",
        help="Skip synthetic dataset generation and reuse existing workspace datasets",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip flowsheet training and reuse existing workspace checkpoints",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip flowsheet evaluation",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write configs and planned steps without executing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    flowsheets = list(dict.fromkeys(args.flowsheets))
    workspace = resolve_workspace_dir(args.workspace_dir)
    configs_dir = workspace / "configs"
    logs_dir = workspace / "logs"
    data_root = workspace / "data"
    outputs_root = workspace / "outputs"
    summary_path = workspace / "summary.json"

    workspace.mkdir(parents=True, exist_ok=True)

    config = build_phase3_smoke_config(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        n_epochs=args.n_epochs,
        max_batches_per_epoch=args.max_batches_per_epoch,
        eval_batches=args.eval_batches,
    )
    config_path = configs_dir / "training_flowsheet_phase3_smoke.yaml"
    write_yaml(config_path, config)

    summary: dict[str, Any] = {
        "status": "running",
        "workspace_dir": str(workspace),
        "started_at": datetime.now().isoformat(),
        "settings": {
            "flowsheets": flowsheets,
            "n_trajectories": args.n_trajectories,
            "n_steps": args.n_steps,
            "dt": args.dt,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "dry_run": args.dry_run,
            "jax_platforms": args.jax_platforms,
        },
        "configs": {
            "training": str(config_path),
        },
        "steps": [],
        "artifacts": {
            "flowsheets": {
                name: {
                    "data_path": str(data_root / name / "train_data.h5"),
                    "manifest_path": str(data_root / name / "manifest.json"),
                    "output_dir": str(outputs_root / name),
                    "model_path": str(outputs_root / name / "best_model.eqx"),
                    "train_summary_path": str(outputs_root / name / "train_summary.json"),
                    "eval_summary_path": str(outputs_root / name / "eval" / "summary.json"),
                }
                for name in flowsheets
            }
        },
        "failure": None,
    }
    write_json(summary_path, summary)

    if args.jax_platforms:
        os.environ["JAX_PLATFORMS"] = args.jax_platforms

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from dte.data.datasets.flowsheet_dataset import FlowsheetTrajectoryDataset
    from dte.flowsheet.examples import (
        build_exchanger_reactor_tank_flowsheet,
        build_reactor_separator_recycle_flowsheet,
    )
    from dte.flowsheet.synthetic import build_synthetic_flowsheet_dataset
    from dte.models.flowsheet.flowsheet_model import FlowsheetModel
    from dte.training.flowsheet.trainer import FlowsheetTrainer

    flowsheet_builders = {
        "exchanger_reactor_tank": build_exchanger_reactor_tank_flowsheet,
        "reactor_separator_recycle": build_reactor_separator_recycle_flowsheet,
    }

    def write_summary() -> None:
        write_json(summary_path, summary)

    def require_path(path: Path, message: str) -> None:
        if not path.exists():
            raise SmokeError(message)

    def run_internal_step(
        *,
        name: str,
        log_path: Path,
        dry_run: bool,
        fn: Callable[[], dict[str, Any] | None],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        step_summary = {
            "name": name,
            "mode": "internal",
            "log_path": str(log_path),
            "started_at": datetime.now().isoformat(),
            "returncode": None,
            "duration_seconds": None,
            "succeeded": False,
        }

        print("\n" + "=" * 60)
        print(f"STEP: {name}")
        print("=" * 60, flush=True)
        print("Log:", log_path, flush=True)

        if dry_run:
            log_path.write_text("[dry-run] step not executed\n", encoding="utf-8")
            step_summary["returncode"] = 0
            step_summary["duration_seconds"] = 0.0
            step_summary["succeeded"] = True
            return step_summary, None

        start = time.perf_counter()
        result: dict[str, Any] | None = None
        with log_path.open("w", encoding="utf-8") as log_handle:
            try:
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    result = fn()
                step_summary["returncode"] = 0
                step_summary["succeeded"] = True
            except Exception as exc:  # pragma: no cover - exercised in smoke runs
                traceback.print_exc(file=log_handle)
                step_summary["returncode"] = 1
                step_summary["succeeded"] = False
                step_summary["error"] = str(exc)
        step_summary["duration_seconds"] = time.perf_counter() - start
        return step_summary, result

    def generate_dataset(flowsheet_name: str, seed: int) -> dict[str, Any]:
        flowsheet = flowsheet_builders[flowsheet_name]()
        dataset = build_synthetic_flowsheet_dataset(
            flowsheet,
            n_trajectories=args.n_trajectories,
            n_steps=args.n_steps,
            dt=args.dt,
            seed=seed,
            seq_len=args.seq_len,
            stride=args.stride,
        )
        data_dir = data_root / flowsheet_name
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / "train_data.h5"
        manifest_path = data_dir / "manifest.json"
        dataset.save_hdf5(data_path)
        write_json(manifest_path, dataset.manifest())
        return {
            "flowsheet_name": flowsheet_name,
            "dataset_path": str(data_path),
            "manifest_path": str(manifest_path),
            "n_samples": dataset.n_samples,
        }

    def train_flowsheet(flowsheet_name: str, seed: int) -> dict[str, Any]:
        data_path = data_root / flowsheet_name / "train_data.h5"
        require_path(data_path, f"missing flowsheet dataset for training: {data_path}")

        dataset = FlowsheetTrajectoryDataset(str(data_path))
        train_dataset, val_dataset = dataset.split(args.val_fraction)
        model = FlowsheetModel.from_config(config, train_dataset.metadata, jax.random.PRNGKey(seed))
        trainer = FlowsheetTrainer(model, config, train_dataset, val_dataset)
        summary_payload = trainer.fit(jax.random.PRNGKey(seed + 1), n_epochs=args.n_epochs)

        output_dir = outputs_root / flowsheet_name
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "best_model.eqx"
        train_summary_path = output_dir / "train_summary.json"
        config_copy_path = output_dir / "config.yaml"

        eqx.tree_serialise_leaves(model_path, trainer.model)
        write_yaml(config_copy_path, config)
        train_summary = {
            **summary_payload,
            "flowsheet_name": flowsheet_name,
            "train_samples": train_dataset.n_samples,
            "val_samples": val_dataset.n_samples,
            "parameter_counts": trainer.model.get_parameter_count(),
        }
        write_json(train_summary_path, train_summary)
        return {
            "flowsheet_name": flowsheet_name,
            "model_path": str(model_path),
            "train_summary_path": str(train_summary_path),
            "train_summary": train_summary,
        }

    def evaluate_flowsheet(flowsheet_name: str, seed: int) -> dict[str, Any]:
        data_path = data_root / flowsheet_name / "train_data.h5"
        model_path = outputs_root / flowsheet_name / "best_model.eqx"
        require_path(data_path, f"missing flowsheet dataset for evaluation: {data_path}")
        require_path(model_path, f"missing flowsheet checkpoint for evaluation: {model_path}")

        dataset = FlowsheetTrajectoryDataset(str(data_path))
        train_dataset, val_dataset = dataset.split(args.val_fraction)
        model_template = FlowsheetModel.from_config(
            config,
            train_dataset.metadata,
            jax.random.PRNGKey(seed),
        )
        model = eqx.tree_deserialise_leaves(model_path, model_template)
        trainer = FlowsheetTrainer(model, config, train_dataset, val_dataset)
        eval_losses, failure_reason = trainer.validate(
            jax.random.PRNGKey(seed + 2),
            n_batches=args.eval_batches,
        )
        if failure_reason is not None:
            raise SmokeError(f"evaluation failed for {flowsheet_name}: {failure_reason}")
        if eval_losses is None:
            raise SmokeError(f"evaluation returned no metrics for {flowsheet_name}")

        eval_summary = {
            "flowsheet_name": flowsheet_name,
            "losses": {
                name: _json_safe_float(value)
                for name, value in eval_losses.items()
            },
            "best_val_loss": _json_safe_float(eval_losses.get("total")),
            "parameter_counts": model.get_parameter_count(),
            "train_samples": train_dataset.n_samples,
            "val_samples": val_dataset.n_samples,
            "eval_batches": args.eval_batches,
        }
        eval_path = outputs_root / flowsheet_name / "eval" / "summary.json"
        write_json(eval_path, eval_summary)
        return {
            "flowsheet_name": flowsheet_name,
            "eval_summary_path": str(eval_path),
            "eval_summary": eval_summary,
        }

    try:
        for idx, flowsheet_name in enumerate(flowsheets):
            flow_artifacts = summary["artifacts"]["flowsheets"][flowsheet_name]

            if not args.skip_data_generation:
                step, result = run_internal_step(
                    name=f"generate_data:{flowsheet_name}",
                    log_path=logs_dir / f"generate_data_{flowsheet_name}.log",
                    dry_run=args.dry_run,
                    fn=lambda n=flowsheet_name, s=args.seed + idx: generate_dataset(n, s),
                )
                summary["steps"].append(step)
                if result is not None:
                    flow_artifacts["generation"] = result
                write_summary()
                if not step["succeeded"]:
                    raise SmokeError(
                        f"step '{step['name']}' failed with return code {step['returncode']}"
                    )

            if not args.skip_training:
                step, result = run_internal_step(
                    name=f"train:{flowsheet_name}",
                    log_path=logs_dir / f"train_{flowsheet_name}.log",
                    dry_run=args.dry_run,
                    fn=lambda n=flowsheet_name, s=args.seed + 100 + idx: train_flowsheet(n, s),
                )
                summary["steps"].append(step)
                if result is not None:
                    flow_artifacts["train_summary"] = result.get("train_summary")
                write_summary()
                if not step["succeeded"]:
                    raise SmokeError(
                        f"step '{step['name']}' failed with return code {step['returncode']}"
                    )

            if not args.skip_evaluation:
                step, result = run_internal_step(
                    name=f"evaluate:{flowsheet_name}",
                    log_path=logs_dir / f"evaluate_{flowsheet_name}.log",
                    dry_run=args.dry_run,
                    fn=lambda n=flowsheet_name, s=args.seed + 200 + idx: evaluate_flowsheet(n, s),
                )
                summary["steps"].append(step)
                if result is not None:
                    flow_artifacts["eval_summary"] = result.get("eval_summary")
                write_summary()
                if not step["succeeded"]:
                    raise SmokeError(
                        f"step '{step['name']}' failed with return code {step['returncode']}"
                    )

        if not args.dry_run and not args.skip_evaluation:
            eval_losses = []
            for flow_name in flowsheets:
                eval_summary = summary["artifacts"]["flowsheets"][flow_name].get("eval_summary") or {}
                loss_total = ((eval_summary.get("losses") or {}).get("total"))
                if loss_total is not None:
                    eval_losses.append(float(loss_total))
            if eval_losses:
                summary["aggregate"] = {
                    "mean_eval_total_loss": sum(eval_losses) / len(eval_losses),
                    "max_eval_total_loss": max(eval_losses),
                    "min_eval_total_loss": min(eval_losses),
                }

    except SmokeError as exc:
        summary["status"] = "failed"
        summary["failure"] = {"message": str(exc)}
        summary["finished_at"] = datetime.now().isoformat()
        write_summary()
        print(f"\nSmoke run failed. Summary saved to {summary_path}")
        print(str(exc))
        return 1

    summary["status"] = "ok"
    summary["finished_at"] = datetime.now().isoformat()
    write_summary()
    print(f"\nPhase 3 smoke run complete. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
