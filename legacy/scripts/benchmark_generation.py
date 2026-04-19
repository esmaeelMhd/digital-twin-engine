"""Benchmark shared data-generation throughput across one or more systems."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import jax
import yaml

from dte.data.generators.generic import GenericDataGenerator
from dte.simulators.registry import get_simulator, get_system_spec, list_systems


def _default_config_paths() -> list[str]:
    """Return the default benchmark configs for all registered systems."""
    config_paths = []
    for system_name in list_systems():
        config_path = Path("configs") / f"{system_name}_default.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Expected default config for system '{system_name}' at {config_path}"
            )
        config_paths.append(str(config_path))
    return config_paths


def _load_target(config_path: str) -> dict[str, Any]:
    """Load one benchmark target from a system config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    system_spec = get_system_spec(config)
    simulator = get_simulator(system_spec.name, config)
    generator = GenericDataGenerator(simulator, config, system_spec)
    return {
        "config": config,
        "config_path": config_path,
        "system_name": system_spec.name,
        "generator": generator,
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    """Render a short markdown report for a multi-system benchmark run."""
    lines = [
        "# Generation Benchmark",
        "",
        f"- Backend: `{payload['backend']}`",
        f"- Trajectories per run: `{payload['n_trajectories']}`",
        f"- Steps per trajectory: `{payload['n_steps']}`",
        f"- Candidate batch sizes: `{payload['batch_sizes']}`",
        "",
    ]

    for system_payload in payload["systems"]:
        lines.extend(
            [
                f"## {system_payload['system_name']}",
                "",
                f"- Config: `{system_payload['config_path']}`",
                f"- Best observed batch size: `{system_payload['best_batch_size']}`",
                f"- Conservative default: `{system_payload['recommended_default']}`",
                "",
                "| Batch Size | Total (s) | Signal (s) | Steady State (s) | Rollout (s) | Invalid | Exceptions |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        for result in system_payload["results"]:
            lines.append(
                "| "
                f"{result['batch_size']} | "
                f"{result['total_generation_seconds']:.2f} | "
                f"{result['signal_generation_seconds']:.2f} | "
                f"{result['steady_state_seconds']:.2f} | "
                f"{result['rollout_seconds']:.2f} | "
                f"{result['invalid_trajectories']} | "
                f"{result['exceptions']} |"
            )

        lines.extend(
            [
                "",
                f"Recommendation: use `batch_size={system_payload['best_batch_size']}` for `{system_payload['system_name']}` on this machine.",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def run_one_benchmark(
    generator: GenericDataGenerator,
    seed: int,
    n_trajectories: int,
    n_steps: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run one benchmark configuration and return timing/profile data."""
    key = jax.random.PRNGKey(seed)
    with tempfile.TemporaryDirectory(prefix="dte-bench-") as tmpdir:
        output_path = os.path.join(tmpdir, "train_data.h5")
        generator.generate_dataset_to_hdf5(
            key,
            output_path,
            n_trajectories=n_trajectories,
            n_steps=n_steps,
            simulation_mode="dataset",
            batch_size=batch_size,
        )
        profile = dict(generator.last_profile)

    return {
        "batch_size": batch_size,
        "total_generation_seconds": float(profile["total_generation_seconds"]),
        "signal_generation_seconds": float(profile["signal_generation_seconds"]),
        "steady_state_seconds": float(profile["steady_state_seconds"]),
        "rollout_seconds": float(profile["rollout_seconds"]),
        "measurement_noise_seconds": float(profile.get("measurement_noise_seconds", 0.0)),
        "validation_seconds": float(profile.get("validation_seconds", 0.0)),
        "invalid_trajectories": int(profile.get("invalid_trajectories", 0)),
        "exceptions": int(profile.get("exceptions", 0)),
        "attempts": int(profile.get("attempts", 0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark shared generation batch sizes across one or more systems"
    )
    parser.add_argument(
        "--config",
        dest="config_paths",
        action="append",
        default=None,
        help=(
            "Path to a system config file. Repeat to benchmark multiple systems. "
            "Defaults to all registered *_default.yaml configs."
        ),
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=8,
        help="Number of trajectories to generate in each benchmark run",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=32,
        help="Number of steps per trajectory in each benchmark run",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON (defaults under outputs/benchmarks/)",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=None,
        help="Path to save a markdown benchmark report (defaults under outputs/benchmarks/)",
    )
    args = parser.parse_args()

    config_paths = args.config_paths or _default_config_paths()
    targets = [_load_target(config_path) for config_path in config_paths]
    backend = jax.default_backend()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print("Generation benchmark")
    print(f"Backend: {backend}")
    print(f"Trajectories per run: {args.n_trajectories}")
    print(f"Steps per trajectory: {args.n_steps}")
    print(f"Candidate batch sizes: {args.batch_sizes}")
    print(f"Systems: {[target['system_name'] for target in targets]}")

    system_payloads = []
    for target in targets:
        generator = target["generator"]
        print(f"\nSystem: {target['system_name']} ({target['config_path']})")

        results = []
        for batch_size in args.batch_sizes:
            print(f"  Benchmarking batch_size={batch_size} ...")
            result = run_one_benchmark(
                generator=generator,
                seed=args.seed,
                n_trajectories=args.n_trajectories,
                n_steps=args.n_steps,
                batch_size=batch_size,
            )
            results.append(result)
            print(
                f"    Total {result['total_generation_seconds']:.2f}s | "
                f"signal {result['signal_generation_seconds']:.2f}s | "
                f"steady_state {result['steady_state_seconds']:.2f}s | "
                f"rollout {result['rollout_seconds']:.2f}s | "
                f"invalid {result['invalid_trajectories']} | "
                f"exceptions {result['exceptions']}"
            )

        best = min(results, key=lambda result: result["total_generation_seconds"])
        recommended_default = generator.recommend_batch_size(backend)
        print(
            f"  Recommendation: best observed batch size {best['batch_size']} "
            f"(conservative default {recommended_default})"
        )
        system_payloads.append(
            {
                "system_name": target["system_name"],
                "config_path": target["config_path"],
                "results": results,
                "best_batch_size": best["batch_size"],
                "recommended_default": recommended_default,
            }
        )

    if args.output_json is None or args.output_md is None:
        output_dir = os.path.join("outputs", "benchmarks")
        os.makedirs(output_dir, exist_ok=True)
        if args.output_json is None:
            args.output_json = os.path.join(output_dir, f"generation-benchmark-{timestamp}.json")
        if args.output_md is None:
            args.output_md = os.path.join(output_dir, f"generation-benchmark-{timestamp}.md")

    payload = {
        "backend": backend,
        "n_trajectories": args.n_trajectories,
        "n_steps": args.n_steps,
        "batch_sizes": args.batch_sizes,
        "systems": system_payloads,
        "timestamp": timestamp,
    }

    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved benchmark results to {args.output_json}")

    if args.output_md is not None:
        with open(args.output_md, "w") as f:
            f.write(build_markdown_report(payload))
        print(f"Saved benchmark report to {args.output_md}")


if __name__ == "__main__":
    main()
