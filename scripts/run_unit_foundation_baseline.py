"""Run the canonical unit-foundation baseline from corpus generation to control gate."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dte.utils.runtime import runtime_env_defaults


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaselineError(RuntimeError):
    """Raised when the canonical baseline cannot complete successfully."""


def resolve_workspace_dir(raw_workspace: str | None) -> Path:
    if raw_workspace is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / "unit_foundation_baseline" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _resolve_jax_platform_env(platform: str) -> str:
    if platform == "gpu":
        return "cuda,cpu"
    return platform


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _attach_per_system_losses(summary: dict[str, Any], trainer, losses) -> None:
    key = (
        "per_system_val_losses"
        if getattr(trainer, "val_dataset", None) is not None
        else "per_system_train_fallback"
    )
    summary[key] = losses


def _read_per_system_losses(summary: dict[str, Any]) -> dict[str, Any]:
    if "per_system_val_losses" in summary:
        return summary["per_system_val_losses"]
    return summary["per_system_train_fallback"]


def _step_summary(name: str, started_at: float, succeeded: bool, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "started_at": datetime.fromtimestamp(started_at).isoformat(),
        "duration_seconds": time.time() - started_at,
        "succeeded": bool(succeeded),
        **{key: _json_safe(value) for key, value in extra.items()},
    }


def _run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    env_updates: dict[str, str],
    dry_run: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("=" * 80)
    print("Command:", " ".join(command), flush=True)
    print("Log:", log_path, flush=True)

    started_at = time.time()
    if dry_run:
        log_path.write_text("[dry-run] command not executed\n", encoding="utf-8")
        return _step_summary(name, started_at, True, command=command, log_path=log_path)

    env = os.environ.copy()
    env.update(runtime_env_defaults())
    env.update(env_updates)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_handle.write(chunk)
        returncode = process.wait()
    if returncode != 0:
        raise BaselineError(f"step '{name}' failed with exit code {returncode}")
    return _step_summary(name, started_at, True, command=command, log_path=log_path)


def _load_universal_sources(config: dict[str, Any]):
    from dte.data.datasets.universal_unit_dataset import SystemDatasetSource

    systems = config.get("data", {}).get("systems", [])
    if not systems:
        raise BaselineError("Universal training config must define data.systems.")
    return [
        SystemDatasetSource(
            name=str(item["name"]),
            system_config=str(item["system_config"]),
            data_dir=str(item["data_dir"]),
            weight=float(item.get("weight", 1.0)),
        )
        for item in systems
    ]


def _build_target_only_universal_config(
    full_catalog_config: dict[str, Any],
    *,
    target_name: str,
    n_epochs: int,
) -> dict[str, Any]:
    config = copy.deepcopy(full_catalog_config)
    systems = config.get("data", {}).get("systems", [])
    target_systems = [item for item in systems if str(item.get("name")) == target_name]
    if len(target_systems) != 1:
        raise BaselineError(f"Could not resolve unique target system '{target_name}' in universal config.")
    config["data"]["systems"] = target_systems
    config.setdefault("training", {})["n_epochs"] = int(n_epochs)
    config.setdefault("checkpointing", {})["val_every"] = 1
    config["checkpointing"]["save_every"] = int(n_epochs)
    config["checkpointing"].setdefault("max_val_batches", 2)
    config.setdefault("evaluation", {})
    config["evaluation"].setdefault("per_system_batches", 2)
    config["evaluation"].setdefault("forecast_batches", 2)
    config["evaluation"].setdefault("rollout_batches", 2)
    config["evaluation"].setdefault("rollout_samples", 4)
    config["evaluation"].setdefault("uncertainty_batches", 0)
    config["evaluation"].setdefault("uncertainty_samples", 0)
    config["evaluation"].setdefault("sensitivity_batches", 0)

    system_specific_losses = config.get("system_specific_losses", {})
    role_derivative_terms = system_specific_losses.get("role_derivative_terms", [])
    if role_derivative_terms:
        filtered_terms: list[dict[str, Any]] = []
        for term in role_derivative_terms:
            filtered_term = copy.deepcopy(term)
            filtered_term["systems"] = [
                system_name
                for system_name in filtered_term.get("systems", [])
                if system_name == target_name
            ]
            if filtered_term["systems"]:
                filtered_terms.append(filtered_term)
        if filtered_terms:
            config.setdefault("system_specific_losses", {})["role_derivative_terms"] = filtered_terms
        else:
            config.pop("system_specific_losses", None)
    return config


def _batches_per_epoch(training_config: dict[str, Any], *, n_train_samples: int) -> int:
    """Resolve the effective train batches per epoch for a target-only run."""

    batch_size = min(int(training_config["batch_size"]), max(1, int(n_train_samples)))
    full_batches = max(1, int(n_train_samples) // batch_size)
    max_batches_per_epoch = training_config.get("max_batches_per_epoch")
    if max_batches_per_epoch is None:
        return full_batches
    return max(1, min(full_batches, int(max_batches_per_epoch)))


def _build_transfer_warm_start_config(
    target_config: dict[str, Any],
    *,
    n_train_samples: int,
    optimizer_variant: str = "default",
) -> dict[str, Any]:
    """Retune the optimizer for few-shot warm-start calibration.

    The transfer target runs for a small number of effective steps, so reusing the
    long pretraining schedule makes warm starts decay too slowly and overshoot.
    Keep the model/training shape identical, but derive the optimizer schedule from
    the actual target dataset size.
    """

    config = copy.deepcopy(target_config)
    training_cfg = config.setdefault("training", {})
    optimizer_cfg = config.setdefault("optimizer", {})

    if optimizer_variant == "conservative":
        training_cfg["n_epochs"] = min(int(training_cfg["n_epochs"]), 2)
    elif optimizer_variant != "default":
        raise BaselineError(f"Unsupported transfer optimizer variant: {optimizer_variant}")

    steps_per_epoch = _batches_per_epoch(training_cfg, n_train_samples=n_train_samples)
    total_steps = max(1, int(training_cfg["n_epochs"]) * steps_per_epoch)
    if optimizer_variant == "conservative":
        warmup_steps = max(1, min(4, total_steps - 1))
        optimizer_cfg["peak_lr"] = min(float(optimizer_cfg["peak_lr"]), 1e-4)
    else:
        warmup_steps = max(1, min(8, total_steps // 10))
        if warmup_steps >= total_steps:
            warmup_steps = max(1, total_steps - 1)
        optimizer_cfg["peak_lr"] = min(float(optimizer_cfg["peak_lr"]), 2e-4)

    optimizer_cfg["total_steps"] = total_steps
    optimizer_cfg["warmup_steps"] = warmup_steps
    return config


def _resolve_config_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _logical_system_name(system_config: dict[str, Any]) -> str:
    return str(system_config.get("system", {}).get("name", "")).strip()


def _system_family(system_config: dict[str, Any]) -> str:
    return str(system_config.get("system", {}).get("family", "")).strip()


def _parameter_descriptors(system_config: dict[str, Any]) -> list[str]:
    return [
        str(descriptor.get("name"))
        for descriptor in system_config.get("system", {}).get("parameter_descriptors", [])
        if descriptor.get("name")
    ]


def _parameter_value_map(system_config: dict[str, Any]) -> dict[str, float]:
    logical_name = _logical_system_name(system_config)
    values = system_config.get(logical_name, {})
    return {
        str(name): float(value)
        for name, value in values.items()
        if isinstance(value, (int, float))
    }


def _select_transfer_source_system_config(
    target_system_config: dict[str, Any],
    source_sources,
) -> dict[str, Any] | None:
    target_logical_name = _logical_system_name(target_system_config)
    exact_name_match = None
    logical_name_match = None

    for source in source_sources:
        source_system_config = load_yaml(_resolve_config_path(source.system_config))
        if source.name == target_logical_name:
            exact_name_match = source_system_config
            break
        if _logical_system_name(source_system_config) == target_logical_name and logical_name_match is None:
            logical_name_match = source_system_config

    return exact_name_match or logical_name_match


def _shifted_param_indices(
    target_system_config: dict[str, Any],
    source_system_config: dict[str, Any] | None,
) -> tuple[int, ...]:
    if source_system_config is None:
        return ()

    target_descriptors = _parameter_descriptors(target_system_config)
    source_values = _parameter_value_map(source_system_config)
    target_values = _parameter_value_map(target_system_config)
    shifted: list[int] = []
    for idx, name in enumerate(target_descriptors):
        if name not in source_values or name not in target_values:
            continue
        if not np.isclose(source_values[name], target_values[name], rtol=1e-6, atol=1e-9):
            shifted.append(idx)
    return tuple(shifted)


def _build_transfer_calibration_policy(
    target_config: dict[str, Any],
    source_sources,
) -> dict[str, Any]:
    target_sources = _load_universal_sources(target_config)
    if len(target_sources) != 1:
        raise BaselineError("Target-only transfer config must contain exactly one system.")

    target_name = str(target_sources[0].name)
    target_system_config = load_yaml(_resolve_config_path(target_sources[0].system_config))
    target_family = _system_family(target_system_config)
    source_system_config = _select_transfer_source_system_config(target_system_config, source_sources)
    active_param_indices = list(_shifted_param_indices(target_system_config, source_system_config))

    if target_name == "cstr_fast_kinetics" or (target_family == "reactor" and active_param_indices):
        return {
            "name": "reactor_fresh_dynamics_full",
            "optimizer_variant": "default",
            "trainable_mode": "full",
            "tune_normalization": True,
            "tune_physics_params": False,
            "active_param_indices": [],
            "restart_seed_offsets": [0],
            "selection_metric": "rollout_rmse",
            "init_kwargs": {
                "copy_drift_backbone": False,
                "copy_cde_backbone": False,
                "copy_drift_adapter": False,
            },
        }

    if target_name == "two_tank_high_throughput" or (target_family == "hydraulic" and active_param_indices):
        return {
            "name": "hydraulic_policy_set",
            "selection_metric": "rollout_rmse",
            "candidate_policies": [
                {
                    "name": "hydraulic_fresh_cde_adapters_norm",
                    "optimizer_variant": "default",
                    "trainable_mode": "adapters",
                    "tune_normalization": True,
                    "tune_physics_params": False,
                    "active_param_indices": [],
                    "restart_seed_offsets": [0],
                    "selection_metric": "rollout_rmse",
                    "init_kwargs": {
                        "copy_drift_backbone": True,
                        "copy_cde_backbone": False,
                        "copy_drift_adapter": True,
                    },
                },
                {
                    "name": "hydraulic_fresh_cde_full_norm",
                    "optimizer_variant": "default",
                    "trainable_mode": "full",
                    "tune_normalization": True,
                    "tune_physics_params": False,
                    "active_param_indices": [],
                    "restart_seed_offsets": [0],
                    "selection_metric": "rollout_rmse",
                    "init_kwargs": {
                        "copy_drift_backbone": True,
                        "copy_cde_backbone": False,
                        "copy_drift_adapter": True,
                    },
                },
            ],
        }

    return {
        "name": "full_warm_start",
        "optimizer_variant": "default",
        "trainable_mode": "full",
        "tune_normalization": True,
        "tune_physics_params": False,
        "active_param_indices": [],
        "restart_seed_offsets": [0],
        "selection_metric": "rollout_rmse",
        "init_kwargs": {
            "copy_drift_backbone": True,
            "copy_cde_backbone": True,
            "copy_drift_adapter": True,
        },
    }


def _build_transfer_source_config(
    training_config: dict[str, Any],
    *,
    transfer_targets: list[str],
    source_epochs: int,
) -> dict[str, Any]:
    config = copy.deepcopy(training_config)
    original_systems = config.get("data", {}).get("systems", [])
    transfer_target_set = set(transfer_targets)
    filtered_systems = [
        item for item in original_systems if str(item.get("name")) not in transfer_target_set
    ]
    if not filtered_systems:
        raise BaselineError("Transfer source pretraining would have zero source systems.")
    config["data"]["systems"] = filtered_systems
    config.setdefault("training", {})["n_epochs"] = int(source_epochs)
    config.setdefault("checkpointing", {})["val_every"] = 1
    config["checkpointing"]["save_every"] = int(source_epochs)

    source_system_names = {str(item["name"]) for item in filtered_systems}
    system_specific_losses = config.get("system_specific_losses", {})
    role_derivative_terms = system_specific_losses.get("role_derivative_terms", [])
    if role_derivative_terms:
        filtered_terms: list[dict[str, Any]] = []
        for term in role_derivative_terms:
            filtered_term = copy.deepcopy(term)
            filtered_term["systems"] = [
                system_name
                for system_name in filtered_term.get("systems", [])
                if system_name in source_system_names
            ]
            if filtered_term["systems"]:
                filtered_terms.append(filtered_term)
        if filtered_terms:
            config.setdefault("system_specific_losses", {})["role_derivative_terms"] = filtered_terms
        else:
            config.pop("system_specific_losses", None)
    return config


def _resolve_checkpoint(run_dir: Path) -> Path:
    for candidate in (run_dir / "best_model.eqx", run_dir / "final_model.eqx"):
        if candidate.exists():
            return candidate
    raise BaselineError(f"No checkpoint found in {run_dir}")


def _has_completed_universal_run(run_dir: Path) -> bool:
    """Return True when a universal training run has a usable checkpoint and summary."""

    return (run_dir / "summary.json").exists() and any(
        candidate.exists() for candidate in (run_dir / "best_model.eqx", run_dir / "final_model.eqx")
    )


def _same_yaml_payload(path: Path, expected: dict[str, Any]) -> bool:
    """Return True when a YAML file exists and matches the expected payload."""

    if not path.exists():
        return False
    return load_yaml(path) == expected


def _same_json_payload(path: Path, expected: dict[str, Any]) -> bool:
    """Return True when a JSON file exists and matches the expected payload."""

    if not path.exists():
        return False
    return load_json(path) == expected


def _select_best_transfer_restart(
    restart_results: list[dict[str, Any]],
    *,
    selection_metric: str,
) -> dict[str, Any]:
    """Pick the best warm-start restart using a deterministic comparison rule."""

    if not restart_results:
        raise BaselineError("Expected at least one warm-start restart result.")

    if selection_metric == "rollout_rmse":
        return min(
            restart_results,
            key=lambda item: (
                float(item["rollout_metrics"]["rmse"]),
                float(_read_per_system_losses(item["train_summary"])[item["target"]]["total"]),
                int(item["restart_index"]),
            ),
        )
    if selection_metric == "total_loss":
        return min(
            restart_results,
            key=lambda item: (
                float(_read_per_system_losses(item["train_summary"])[item["target"]]["total"]),
                float(item["rollout_metrics"]["rmse"]),
                int(item["restart_index"]),
            ),
        )

    raise BaselineError(f"Unsupported transfer restart selection metric: {selection_metric}")


def _select_best_transfer_candidate(
    candidate_results: list[dict[str, Any]],
    *,
    selection_metric: str,
) -> dict[str, Any]:
    """Pick the best warm-start candidate policy using the selected restart metrics."""

    if not candidate_results:
        raise BaselineError("Expected at least one transfer policy candidate result.")

    if selection_metric == "rollout_rmse":
        return min(
            candidate_results,
            key=lambda item: (
                float(item["selected_restart"]["rollout_metrics"]["rmse"]),
                float(
                    _read_per_system_losses(item["selected_restart"]["train_summary"])[item["target"]]["total"]
                ),
                int(item["candidate_index"]),
            ),
        )
    if selection_metric == "total_loss":
        return min(
            candidate_results,
            key=lambda item: (
                float(
                    _read_per_system_losses(item["selected_restart"]["train_summary"])[item["target"]]["total"]
                ),
                float(item["selected_restart"]["rollout_metrics"]["rmse"]),
                int(item["candidate_index"]),
            ),
        )

    raise BaselineError(f"Unsupported transfer candidate selection metric: {selection_metric}")


def _state_bounds(spec) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(spec.state_dim, -np.inf, dtype=np.float32)
    upper = np.full(spec.state_dim, np.inf, dtype=np.float32)
    for idx, channel in enumerate(getattr(spec, "state_channels", [])):
        if channel.lower_bound is not None:
            lower[idx] = float(channel.lower_bound)
        if channel.upper_bound is not None:
            upper[idx] = float(channel.upper_bound)
    return lower, upper


def _target_state(spec) -> np.ndarray:
    initial = np.asarray(spec.default_initial_state, dtype=np.float32)
    base_scale = np.asarray(
        getattr(spec.normalization, "state_scale", [1.0] * spec.state_dim),
        dtype=np.float32,
    )
    shift = np.maximum(0.05 * base_scale[: spec.state_dim], 1e-3)
    direction = np.where(np.arange(spec.state_dim) % 2 == 0, 1.0, -1.0).astype(np.float32)
    target = initial + direction * shift
    lower, upper = _state_bounds(spec)
    finite_lower = np.where(np.isfinite(lower), lower, -np.inf)
    finite_upper = np.where(np.isfinite(upper), upper, np.inf)
    return np.clip(target, finite_lower, finite_upper).astype(np.float32)


def _run_control_gate(
    *,
    training_config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    control_systems: list[str],
    horizon: int,
    n_candidates: int,
    rollout_samples: int,
    seed: int,
) -> dict[str, Any]:
    from dte.control.mpc_interface import MPCInterfaceConfig, ProcessMPCInterface
    from dte.data.datasets.universal_unit_dataset import MultiSystemTrajectoryDataset
    from dte.evaluation.control_metrics import mismatch_robustness
    from dte.models.universal.digital_twin import UniversalDigitalTwin
    from dte.simulators.registry import get_simulator, get_system_spec

    training_config = load_yaml(training_config_path)
    sources = _load_universal_sources(training_config)
    source_by_name = {source.name: source for source in sources}
    metadata = MultiSystemTrajectoryDataset.metadata_from_sources(sources)
    model = UniversalDigitalTwin.load(str(checkpoint_path), training_config, metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "ok",
        "model_path": str(checkpoint_path.resolve()),
        "training_config_path": str(training_config_path.resolve()),
        "systems": {},
    }
    mismatch_values: list[float] = []

    for idx, system_name in enumerate(control_systems):
        if system_name not in source_by_name:
            raise BaselineError(
                f"Control gate system '{system_name}' is not present in the universal training config."
            )

        source = source_by_name[system_name]
        system_config = load_yaml(PROJECT_ROOT / source.system_config)
        spec = get_system_spec(system_config)
        simulator = get_simulator(spec.name, system_config)
        dt = float(system_config.get("simulation", {}).get("dt", 0.1))
        runtime = ProcessMPCInterface(
            spec,
            simulator,
            model=model,
            config=MPCInterfaceConfig(
                dt=dt,
                horizon=horizon,
                rollout_samples=rollout_samples,
            ),
        )

        initial_state = np.asarray(spec.default_initial_state, dtype=np.float32)
        target_state = _target_state(spec)
        runtime.reset(initial_state=initial_state)
        best = runtime.optimize_random_shooting(
            target_state=target_state,
            horizon=horizon,
            n_candidates=n_candidates,
            use_model=True,
            seed=seed + idx,
        )
        simulator_eval = runtime.evaluate_candidate(
            np.asarray(best["controls"], dtype=np.float32),
            target_state=target_state,
            use_model=False,
            seed=seed + 100 + idx,
        )
        mismatch = mismatch_robustness(simulator_eval["states"], best["states"])
        mismatch_values.append(float(mismatch["normalized_rmse"]))

        system_summary = {
            "target_state": target_state,
            "model_objective": float(best["objective"]),
            "simulator_objective": float(simulator_eval["objective"]),
            "model_metrics": best["metrics"],
            "simulator_metrics": simulator_eval["metrics"],
            "mismatch": mismatch,
            "source": str(best["source"]),
        }
        write_json(output_dir / f"{system_name}.json", _json_safe(system_summary))
        summary["systems"][system_name] = _json_safe(system_summary)

    summary["aggregate"] = {
        "mean_normalized_mismatch_rmse": float(np.mean(mismatch_values)) if mismatch_values else None,
        "max_normalized_mismatch_rmse": float(np.max(mismatch_values)) if mismatch_values else None,
        "n_systems": len(control_systems),
    }
    write_json(output_dir / "summary.json", _json_safe(summary))
    return summary


def _run_transfer_benchmark(
    *,
    full_training_config_path: Path,
    source_config_path: Path,
    source_checkpoint_path: Path,
    output_dir: Path,
    transfer_targets: list[str],
    transfer_epochs: int,
    trainable_mode: str,
    seed: int,
) -> dict[str, Any]:
    import jax

    from dte.calibration.unit_calibration import (
        CalibrationOptions,
        UnitCalibrator,
        initialize_target_model_from_pretrained,
    )
    from dte.data.datasets.universal_unit_dataset import MultiSystemTrajectoryDataset
    from dte.evaluation.universal import compute_forecast_metrics, compute_rollout_metrics
    from dte.models.universal.digital_twin import UniversalDigitalTwin
    from dte.training.universal.trainer import UniversalTrainer

    full_training_config = load_yaml(full_training_config_path)
    source_config = load_yaml(source_config_path)
    source_sources = _load_universal_sources(source_config)
    source_metadata = MultiSystemTrajectoryDataset.metadata_from_sources(source_sources)
    source_model = UniversalDigitalTwin.load(str(source_checkpoint_path), source_config, source_metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "ok",
        "source_config_path": str(source_config_path.resolve()),
        "source_checkpoint_path": str(source_checkpoint_path.resolve()),
        "transfer_targets": transfer_targets,
        "results": {},
    }
    all_improved = True

    for idx, target_name in enumerate(transfer_targets):
        target_config = _build_target_only_universal_config(
            full_training_config,
            target_name=target_name,
            n_epochs=transfer_epochs,
        )
        calibration_policy = _build_transfer_calibration_policy(
            target_config,
            source_sources,
        )
        target_dir = output_dir / target_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_config_path = target_dir / "target_config.json"
        warm_start_config_path = target_dir / "warm_start_config.json"
        calibration_policy_path = target_dir / "calibration_policy.json"
        existing_summary_path = target_dir / "summary.json"

        target_sources = _load_universal_sources(target_config)
        target_dataset = MultiSystemTrajectoryDataset.from_sources(
            target_sources,
            seq_len=int(target_config["training"]["seq_len"]),
            stride=int(target_config["training"]["stride"]),
        )
        train_dataset, val_dataset = target_dataset.split(float(target_config["training"].get("val_split", 0.2)))
        target_metadata = train_dataset.metadata
        warm_start_config = _build_transfer_warm_start_config(
            target_config,
            n_train_samples=train_dataset.n_samples,
            optimizer_variant=str(calibration_policy.get("optimizer_variant", "default")),
        )
        evaluation_key = jax.random.PRNGKey(seed + 1000 + 10 * idx)
        per_system_eval_key, forecast_eval_key, rollout_eval_key = jax.random.split(
            evaluation_key,
            3,
        )

        can_reuse_target = (
            existing_summary_path.exists()
            and _same_json_payload(target_config_path, target_config)
            and _same_json_payload(warm_start_config_path, warm_start_config)
            and _same_json_payload(calibration_policy_path, calibration_policy)
        )
        if can_reuse_target:
            existing_result = load_json(existing_summary_path)
            improved = bool(
                existing_result.get("comparison", {}).get("improved_over_scratch", False)
            )
            all_improved = all_improved and improved
            summary["results"][target_name] = _json_safe(existing_result)
            continue

        write_json(target_config_path, target_config)
        write_json(warm_start_config_path, warm_start_config)
        write_json(calibration_policy_path, calibration_policy)

        selection_metric = str(calibration_policy.get("selection_metric", "rollout_rmse"))
        policy_candidates = calibration_policy.get("candidate_policies", [calibration_policy])
        candidate_results: list[dict[str, Any]] = []

        for candidate_index, candidate_policy in enumerate(policy_candidates):
            restart_seed_offsets = [
                int(offset) for offset in candidate_policy.get("restart_seed_offsets", [0])
            ]
            warm_restart_results: list[dict[str, Any]] = []
            warm_start_root = target_dir / "warm_start" / str(candidate_policy["name"])

            for restart_index, restart_offset in enumerate(restart_seed_offsets):
                restart_dir = (
                    warm_start_root
                    if len(restart_seed_offsets) == 1
                    else warm_start_root / f"restart_{restart_index}"
                )
                warm_model = initialize_target_model_from_pretrained(
                    source_model,
                    source_metadata,
                    target_metadata,
                    warm_start_config,
                    jax.random.PRNGKey(seed + 100 * idx + 10 * candidate_index + restart_offset),
                    **dict(candidate_policy.get("init_kwargs", {})),
                )
                warm_calibrator = UnitCalibrator(
                    warm_model,
                    warm_start_config,
                    train_dataset,
                    val_dataset,
                    options=CalibrationOptions(
                        trainable_mode=str(candidate_policy["trainable_mode"]),
                        tune_normalization=bool(candidate_policy["tune_normalization"]),
                        tune_physics_params=bool(candidate_policy["tune_physics_params"]),
                        active_param_indices=tuple(int(idx) for idx in candidate_policy["active_param_indices"]),
                    ),
                    target_system_id=0,
                )
                warm_summary = warm_calibrator.calibrate(
                    str(restart_dir),
                    key=jax.random.PRNGKey(seed + 100 * idx + 10 * candidate_index + restart_offset + 1),
                )
                warm_best_model = UniversalDigitalTwin.load(
                    str(_resolve_checkpoint(restart_dir)),
                    warm_start_config,
                    target_metadata,
                )
                warm_eval_trainer = UniversalTrainer(
                    warm_best_model,
                    warm_start_config,
                    train_dataset,
                    val_dataset,
                )
                warm_per_system = warm_eval_trainer.evaluate_per_system(
                    per_system_eval_key,
                    n_batches=int(warm_start_config.get("evaluation", {}).get("per_system_batches", 2)),
                )
                _attach_per_system_losses(warm_summary, warm_eval_trainer, warm_per_system)
                warm_forecast = compute_forecast_metrics(
                    warm_eval_trainer.model,
                    warm_eval_trainer,
                    system_idx=0,
                    key=forecast_eval_key,
                    n_batches=int(warm_start_config.get("evaluation", {}).get("forecast_batches", 2)),
                )
                warm_rollout = compute_rollout_metrics(
                    warm_eval_trainer.model,
                    warm_eval_trainer,
                    system_idx=0,
                    key=rollout_eval_key,
                    n_batches=int(warm_start_config.get("evaluation", {}).get("rollout_batches", 2)),
                    n_samples=int(warm_start_config.get("evaluation", {}).get("rollout_samples", 4)),
                )
                restart_result = {
                    "target": target_name,
                    "candidate_index": candidate_index,
                    "policy_name": str(candidate_policy["name"]),
                    "restart_index": restart_index,
                    "restart_seed_offset": restart_offset,
                    "train_summary": warm_summary,
                    "forecast_metrics": warm_forecast,
                    "rollout_metrics": warm_rollout,
                }
                write_json(restart_dir / "summary.json", _json_safe(restart_result))
                warm_restart_results.append(restart_result)

            selected_restart = _select_best_transfer_restart(
                warm_restart_results,
                selection_metric=str(candidate_policy.get("selection_metric", selection_metric)),
            )
            candidate_results.append(
                {
                    "target": target_name,
                    "candidate_index": candidate_index,
                    "policy": copy.deepcopy(candidate_policy),
                    "restart_summaries": warm_restart_results,
                    "selected_restart": selected_restart,
                }
            )

        selected_candidate = _select_best_transfer_candidate(
            candidate_results,
            selection_metric=selection_metric,
        )
        selected_policy = copy.deepcopy(selected_candidate["policy"])
        selected_warm = copy.deepcopy(selected_candidate["selected_restart"])
        warm_summary = copy.deepcopy(selected_warm["train_summary"])
        warm_forecast = copy.deepcopy(selected_warm["forecast_metrics"])
        warm_rollout = copy.deepcopy(selected_warm["rollout_metrics"])

        scratch_model = UniversalDigitalTwin.from_config(
            target_config,
            target_metadata,
            jax.random.PRNGKey(seed + 10 * idx + 4),
        )
        scratch_trainer = UniversalTrainer(
            scratch_model,
            target_config,
            train_dataset,
            val_dataset,
        )
        scratch_train_summary = scratch_trainer.train(
            n_epochs=int(target_config["training"]["n_epochs"]),
            output_dir=str(target_dir / "scratch"),
            key=jax.random.PRNGKey(seed + 10 * idx + 5),
        )
        scratch_best_model = UniversalDigitalTwin.load(
            str(_resolve_checkpoint(target_dir / "scratch")),
            target_config,
            target_metadata,
        )
        scratch_eval_trainer = UniversalTrainer(
            scratch_best_model,
            target_config,
            train_dataset,
            val_dataset,
        )
        scratch_per_system = scratch_eval_trainer.evaluate_per_system(
            per_system_eval_key,
            n_batches=int(target_config.get("evaluation", {}).get("per_system_batches", 2)),
        )
        scratch_forecast = compute_forecast_metrics(
            scratch_eval_trainer.model,
            scratch_eval_trainer,
            system_idx=0,
            key=forecast_eval_key,
            n_batches=int(target_config.get("evaluation", {}).get("forecast_batches", 2)),
        )
        scratch_rollout = compute_rollout_metrics(
            scratch_eval_trainer.model,
            scratch_eval_trainer,
            system_idx=0,
            key=rollout_eval_key,
            n_batches=int(target_config.get("evaluation", {}).get("rollout_batches", 2)),
            n_samples=int(target_config.get("evaluation", {}).get("rollout_samples", 4)),
        )

        warm_total = float(_read_per_system_losses(warm_summary)[target_name]["total"])
        scratch_total = float(scratch_per_system[target_name]["total"])
        improved = (warm_total <= scratch_total) and (warm_rollout["rmse"] <= scratch_rollout["rmse"])
        all_improved = all_improved and improved

        result = {
            "target": target_name,
            "warm_start": {
                "policy": selected_policy,
                "candidate_policies": candidate_results,
                "selection_metric": selection_metric,
                "selected_candidate_index": int(selected_candidate["candidate_index"]),
                "selected_policy_name": str(selected_policy["name"]),
                "selected_restart_index": int(selected_warm["restart_index"]),
                "selected_restart_seed_offset": int(selected_warm["restart_seed_offset"]),
                "effective_optimizer": warm_start_config["optimizer"],
                "train_summary": warm_summary,
                "forecast_metrics": warm_forecast,
                "rollout_metrics": warm_rollout,
            },
            "scratch": {
                "train_summary": {
                    **scratch_train_summary,
                    (
                        "per_system_val_losses"
                        if scratch_eval_trainer.val_dataset is not None
                        else "per_system_train_fallback"
                    ): scratch_per_system,
                },
                "forecast_metrics": scratch_forecast,
                "rollout_metrics": scratch_rollout,
            },
            "comparison": {
                "warm_start_total_loss": warm_total,
                "scratch_total_loss": scratch_total,
                "warm_start_rollout_rmse": float(warm_rollout["rmse"]),
                "scratch_rollout_rmse": float(scratch_rollout["rmse"]),
                "improved_over_scratch": improved,
            },
        }
        write_json(target_dir / "summary.json", _json_safe(result))
        summary["results"][target_name] = _json_safe(result)

    summary["aggregate"] = {
        "all_targets_improved_over_scratch": all_improved,
        "n_targets": len(transfer_targets),
    }
    write_json(output_dir / "summary.json", _json_safe(summary))
    return summary


def _build_acceptance_summary(
    *,
    eval_summary: dict[str, Any] | None,
    transfer_summary: dict[str, Any] | None,
    control_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    rollout_metrics = (eval_summary or {}).get("rollout_metrics", {})
    control_metrics = (eval_summary or {}).get("control_sensitivity_metrics", {})

    rollout_ok = bool(rollout_metrics) and all(
        float(metrics.get("rmse", float("inf"))) < 10.0 for metrics in rollout_metrics.values()
    )
    control_measured = bool(control_metrics)
    transfer_ok = bool(transfer_summary) and bool(
        transfer_summary.get("aggregate", {}).get("all_targets_improved_over_scratch", False)
    )
    aggregate_metric_value = (eval_summary or {}).get("aggregate_metric_value")
    checkpoint_ok = aggregate_metric_value is not None
    control_gate_ok = bool(control_summary) and (
        control_summary.get("aggregate", {}).get("max_normalized_mismatch_rmse") is not None
    )

    gates = {
        "shared_checkpoint_trains_reproducibly": checkpoint_ok,
        "rollout_stability_on_held_out_variants": rollout_ok,
        "transfer_beats_scratch_on_targets": transfer_ok,
        "control_response_fidelity_is_measured": control_measured,
        "control_gate_completed": control_gate_ok,
    }
    return {
        "phase": "phase1_unit_foundation_v1",
        "accepted": all(gates.values()),
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical unit-foundation baseline: regime corpus -> universal training -> evaluation -> control gate."
    )
    parser.add_argument(
        "--generation_config",
        type=str,
        default="configs/generation_phase1_regime.yaml",
        help="Corpus generation manifest for the canonical unit-foundation run.",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training_universal_phase1_regime.yaml",
        help="Universal training config for the canonical unit-foundation run.",
    )
    parser.add_argument("--workspace_dir", type=str, default=None)
    parser.add_argument("--jax_platform", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--control_systems",
        nargs="+",
        default=["cstr", "heat_exchanger", "two_tank"],
        help="Systems used for the control-readiness gate after universal evaluation.",
    )
    parser.add_argument("--control_horizon", type=int, default=12)
    parser.add_argument("--control_candidates", type=int, default=16)
    parser.add_argument("--control_rollout_samples", type=int, default=4)
    parser.add_argument(
        "--transfer_targets",
        nargs="+",
        default=["cstr_fast_kinetics", "heat_exchanger_high_ua", "two_tank_high_throughput"],
        help="Held-out target systems used for the Phase 1 transfer benchmark.",
    )
    parser.add_argument("--transfer_epochs", type=int, default=4)
    parser.add_argument("--transfer_source_epochs", type=int, default=6)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--skip_transfer", action="store_true")
    parser.add_argument("--skip_control", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    workspace_dir = resolve_workspace_dir(args.workspace_dir)
    logs_dir = workspace_dir / "logs"
    outputs_dir = workspace_dir / "outputs"
    run_dir = outputs_dir / "unit_foundation"
    eval_dir = run_dir / "eval"
    transfer_dir = run_dir / "transfer_benchmark"
    control_dir = run_dir / "control_gate"
    summary_path = workspace_dir / "summary.json"

    workspace_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "running",
        "workspace_dir": str(workspace_dir.resolve()),
        "generation_config": str((PROJECT_ROOT / args.generation_config).resolve()),
        "training_config": str((PROJECT_ROOT / args.training_config).resolve()),
        "steps": [],
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "eval_dir": str(eval_dir.resolve()),
            "transfer_dir": str(transfer_dir.resolve()),
            "control_dir": str(control_dir.resolve()),
        },
    }
    write_json(summary_path, summary)

    env_updates = {
        "JAX_PLATFORMS": _resolve_jax_platform_env(args.jax_platform),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    try:
        if not args.skip_generation:
            summary["steps"].append(
                _run_command(
                    name="generate_regime_corpus",
                    command=[
                        sys.executable,
                        "scripts/generate_corpus.py",
                        "--config",
                        args.generation_config,
                    ],
                    log_path=logs_dir / "generate_regime_corpus.log",
                    env_updates=env_updates,
                    dry_run=args.dry_run,
                )
            )
            write_json(summary_path, summary)

        if not args.skip_training:
            summary["steps"].append(
                _run_command(
                    name="train_unit_foundation",
                    command=[
                        sys.executable,
                        "scripts/train_universal.py",
                        "--config",
                        args.training_config,
                        "--output_dir",
                        str(run_dir),
                        "--seed",
                        str(args.seed),
                    ],
                    log_path=logs_dir / "train_unit_foundation.log",
                    env_updates=env_updates,
                    dry_run=args.dry_run,
                )
            )
            write_json(summary_path, summary)

        if not args.skip_evaluation:
            checkpoint_path = run_dir / "best_model.eqx" if args.dry_run else _resolve_checkpoint(run_dir)
            summary["steps"].append(
                _run_command(
                    name="evaluate_unit_foundation",
                    command=[
                        sys.executable,
                        "scripts/evaluate_universal.py",
                        "--model_path",
                        str(checkpoint_path),
                        "--config",
                        str(run_dir / "config.yaml"),
                        "--output_dir",
                        str(eval_dir),
                        "--seed",
                        str(args.seed),
                    ],
                    log_path=logs_dir / "evaluate_unit_foundation.log",
                    env_updates=env_updates,
                    dry_run=args.dry_run,
                )
            )
            write_json(summary_path, summary)

        if not args.skip_transfer:
            started_at = time.time()
            if args.dry_run:
                summary["steps"].append(
                    _step_summary(
                        "transfer_benchmark",
                        started_at,
                        True,
                        transfer_targets=args.transfer_targets,
                        transfer_epochs=args.transfer_epochs,
                        transfer_source_epochs=args.transfer_source_epochs,
                        transfer_trainable_mode="full",
                    )
                )
            else:
                source_config = _build_transfer_source_config(
                    load_yaml(run_dir / "config.yaml"),
                    transfer_targets=list(args.transfer_targets),
                    source_epochs=int(args.transfer_source_epochs),
                )
                transfer_workspace = transfer_dir / "source_pretrain"
                transfer_workspace.mkdir(parents=True, exist_ok=True)
                source_config_path = transfer_workspace / "config.yaml"
                can_reuse_source_pretrain = _has_completed_universal_run(
                    transfer_workspace
                ) and _same_yaml_payload(source_config_path, source_config)
                if can_reuse_source_pretrain:
                    step = _step_summary(
                        "transfer_source_pretrain",
                        started_at,
                        True,
                        reused_existing_artifacts=True,
                        output_dir=str(transfer_workspace.resolve()),
                        transfer_trainable_mode="full",
                    )
                else:
                    with source_config_path.open("w", encoding="utf-8") as handle:
                        yaml.safe_dump(source_config, handle, sort_keys=False)
                    step = _run_command(
                        name="transfer_source_pretrain",
                        command=[
                            sys.executable,
                            "scripts/train_universal.py",
                            "--config",
                            str(source_config_path),
                            "--output_dir",
                            str(transfer_workspace),
                            "--seed",
                            str(args.seed),
                        ],
                        log_path=logs_dir / "transfer_source_pretrain.log",
                        env_updates=env_updates,
                        dry_run=False,
                    )
                summary["steps"].append(step)
                transfer_summary = _run_transfer_benchmark(
                    full_training_config_path=run_dir / "config.yaml",
                    source_config_path=source_config_path,
                    source_checkpoint_path=_resolve_checkpoint(transfer_workspace),
                    output_dir=transfer_dir,
                    transfer_targets=list(args.transfer_targets),
                    transfer_epochs=int(args.transfer_epochs),
                    trainable_mode="full",
                    seed=int(args.seed),
                )
                summary["artifacts"]["transfer_summary"] = str((transfer_dir / "summary.json").resolve())
                summary["steps"].append(
                    _step_summary(
                        "transfer_benchmark",
                        started_at,
                        True,
                        aggregate=transfer_summary.get("aggregate", {}),
                    )
                )
            write_json(summary_path, summary)

        if not args.skip_control:
            started_at = time.time()
            if args.dry_run:
                summary["steps"].append(
                    _step_summary(
                        "control_gate",
                        started_at,
                        True,
                        control_systems=args.control_systems,
                        control_horizon=args.control_horizon,
                        control_candidates=args.control_candidates,
                        control_rollout_samples=args.control_rollout_samples,
                    )
                )
            else:
                checkpoint_path = _resolve_checkpoint(run_dir)
                control_summary = _run_control_gate(
                    training_config_path=run_dir / "config.yaml",
                    checkpoint_path=checkpoint_path,
                    output_dir=control_dir,
                    control_systems=list(args.control_systems),
                    horizon=int(args.control_horizon),
                    n_candidates=int(args.control_candidates),
                    rollout_samples=int(args.control_rollout_samples),
                    seed=int(args.seed),
                )
                summary["artifacts"]["control_summary"] = str((control_dir / "summary.json").resolve())
                summary["steps"].append(
                    _step_summary(
                        "control_gate",
                        started_at,
                        True,
                        aggregate=control_summary.get("aggregate", {}),
                    )
                )
            write_json(summary_path, summary)

        if args.dry_run:
            summary["acceptance"] = _build_acceptance_summary(
                eval_summary=None,
                transfer_summary=None,
                control_summary=None,
            )
        else:
            with (eval_dir / "summary.json").open("r", encoding="utf-8") as handle:
                eval_summary = json.load(handle)
            transfer_summary = None
            if not args.skip_transfer and (transfer_dir / "summary.json").exists():
                with (transfer_dir / "summary.json").open("r", encoding="utf-8") as handle:
                    transfer_summary = json.load(handle)
            control_summary = None
            if not args.skip_control and (control_dir / "summary.json").exists():
                with (control_dir / "summary.json").open("r", encoding="utf-8") as handle:
                    control_summary = json.load(handle)
            summary["acceptance"] = _build_acceptance_summary(
                eval_summary=eval_summary,
                transfer_summary=transfer_summary,
                control_summary=control_summary,
            )

        summary["status"] = "dry_run" if args.dry_run else "ok"
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
