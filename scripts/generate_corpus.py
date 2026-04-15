"""Generate a multi-system synthetic corpus from a manifest config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import jax
import yaml

from dte.data.generation_generic import GenericDataGenerator
from dte.simulators.registry import get_simulator, get_system_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def load_corpus_manifest(path: str) -> dict[str, Any]:
    return _load_yaml(_resolve_path(path))


def build_generation_jobs(
    manifest: dict[str, Any],
    *,
    output_root_override: str | None = None,
    n_trajectories_override: int | None = None,
    n_steps_override: int | None = None,
    batch_size_override: int | None = None,
    simulation_mode_override: str | None = None,
    selected_systems: set[str] | None = None,
) -> list[dict[str, Any]]:
    defaults = manifest.get("defaults", {})
    output_root = _resolve_path(output_root_override) or _resolve_path(defaults.get("output_root", "data"))
    systems = manifest.get("systems", [])
    if not systems:
        raise ValueError("Corpus manifest must define at least one system.")

    jobs: list[dict[str, Any]] = []
    for index, item in enumerate(systems):
        name = str(item["name"])
        if selected_systems is not None and name not in selected_systems:
            continue
        system_config = _resolve_path(item["system_config"])
        if output_root_override is not None:
            output_dir = output_root / name
        else:
            raw_output_dir = item.get("output_dir")
            if raw_output_dir is None:
                output_dir = output_root / name
            else:
                output_path = Path(str(raw_output_dir))
                output_dir = output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
        jobs.append(
            {
                "name": name,
                "system_config": system_config,
                "output_dir": output_dir,
                "n_trajectories": int(item.get("n_trajectories", n_trajectories_override or defaults.get("n_trajectories", 1000))),
                "n_steps": int(item.get("n_steps", n_steps_override or defaults.get("n_steps", 500))),
                "batch_size": (
                    batch_size_override
                    if batch_size_override is not None
                    else item.get("batch_size")
                ),
                "simulation_mode": str(
                    simulation_mode_override
                    or item.get("simulation_mode", defaults.get("simulation_mode", "dataset"))
                ),
                "seed_offset": int(item.get("seed_offset", index)),
            }
        )
    if not jobs:
        raise ValueError("Corpus manifest selection produced no systems.")
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Generate a multi-system synthetic corpus")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generation_phase1_corpus.yaml",
        help="Path to corpus generation manifest",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional output-root override for systems without explicit output_dir",
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=None,
        help="Optional override applied to every selected system",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="Optional override applied to every selected system",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override applied to every selected system",
    )
    parser.add_argument(
        "--simulation_mode",
        type=str,
        default=None,
        choices=["dataset", "reference"],
        help="Optional override applied to every selected system",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Optional subset of system names to generate",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional global seed override")
    args = parser.parse_args()

    manifest = load_corpus_manifest(args.config)
    base_seed = int(args.seed if args.seed is not None else manifest.get("seed", 42))
    selected_systems = None if args.systems is None else set(args.systems)
    jobs = build_generation_jobs(
        manifest,
        output_root_override=args.output_root,
        n_trajectories_override=args.n_trajectories,
        n_steps_override=args.n_steps,
        batch_size_override=args.batch_size,
        simulation_mode_override=args.simulation_mode,
        selected_systems=selected_systems,
    )

    summaries: list[dict[str, Any]] = []
    for job in jobs:
        with job["system_config"].open("r", encoding="utf-8") as handle:
            system_config = yaml.safe_load(handle) or {}
        spec = get_system_spec(system_config)
        simulator = get_simulator(spec.name, system_config)
        generator = GenericDataGenerator(simulator, system_config, spec)
        batch_size = (
            int(job["batch_size"])
            if job["batch_size"] is not None
            else generator.recommend_batch_size(jax.default_backend())
        )

        job["output_dir"].mkdir(parents=True, exist_ok=True)
        output_path = job["output_dir"] / "train_data.h5"
        seed = base_seed + int(job["seed_offset"])
        print(
            f"[corpus] generating {job['name']} -> {output_path} "
            f"({job['n_trajectories']} trajectories, {job['n_steps']} steps, seed={seed})"
        )
        summary = generator.generate_dataset_to_hdf5(
            jax.random.PRNGKey(seed),
            str(output_path),
            n_trajectories=int(job["n_trajectories"]),
            n_steps=int(job["n_steps"]),
            simulation_mode=str(job["simulation_mode"]),
            batch_size=batch_size,
        )
        summaries.append(
            {
                "name": job["name"],
                "system_config": str(job["system_config"]),
                "output_dir": str(job["output_dir"]),
                "seed": seed,
                "batch_size": batch_size,
                "simulation_mode": job["simulation_mode"],
                "summary": summary,
            }
        )

    manifest_path = _resolve_path(args.output_root) or _resolve_path(manifest.get("defaults", {}).get("output_root", "data"))
    output_manifest = manifest_path / "phase1_corpus_manifest.json"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(
            _json_safe(
                {
                "config": args.config,
                "seed": base_seed,
                "systems": summaries,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[corpus] wrote manifest to {output_manifest}")


if __name__ == "__main__":
    main()
