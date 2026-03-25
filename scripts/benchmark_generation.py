"""Benchmark data-generation batch sizes and print a recommendation."""

import argparse
from datetime import datetime
import json
import os
import tempfile

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import jax
import yaml

from dte.data.generation import DataGenerator
from dte.simulators.cstr import CSTRParams, CSTRSimulator


def build_markdown_report(payload: dict) -> str:
    """Render a short markdown report for a benchmark run."""
    lines = [
        "# Generation Benchmark",
        "",
        f"- Backend: `{payload['backend']}`",
        f"- Trajectories per run: `{payload['n_trajectories']}`",
        f"- Steps per trajectory: `{payload['n_steps']}`",
        f"- Best observed batch size: `{payload['best_batch_size']}`",
        f"- Conservative default: `{payload['recommended_default']}`",
        "",
        "| Batch Size | Total (s) | Signal (s) | Steady State (s) | Rollout (s) | Fast SS | Fallbacks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in payload["results"]:
        lines.append(
            "| "
            f"{result['batch_size']} | "
            f"{result['total_generation_seconds']:.2f} | "
            f"{result['signal_generation_seconds']:.2f} | "
            f"{result['steady_state_seconds']:.2f} | "
            f"{result['rollout_seconds']:.2f} | "
            f"{result['fast_steady_states']} | "
            f"{result['steady_state_fallbacks']} |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Use `batch_size={payload['best_batch_size']}` on this machine for similar workloads.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_one_benchmark(
    generator: DataGenerator,
    seed: int,
    n_trajectories: int,
    n_steps: int,
    batch_size: int,
) -> dict:
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
        "total_generation_seconds": profile["total_generation_seconds"],
        "signal_generation_seconds": profile["signal_generation_seconds"],
        "steady_state_seconds": profile["steady_state_seconds"],
        "rollout_seconds": profile["rollout_seconds"],
        "fast_steady_states": int(profile.get("steady_state_fast_successes", 0)),
        "steady_state_fallbacks": int(profile.get("steady_state_fallbacks", 0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark generation batch sizes")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cstr_default.yaml",
        help="Path to config file",
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

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    params = CSTRParams(**config["cstr"])
    simulator = CSTRSimulator(params)
    generator = DataGenerator(simulator, config)
    backend = jax.default_backend()

    print("Generation benchmark")
    print(f"Backend: {backend}")
    print(f"Trajectories per run: {args.n_trajectories}")
    print(f"Steps per trajectory: {args.n_steps}")
    print(f"Candidate batch sizes: {args.batch_sizes}")

    results = []
    for batch_size in args.batch_sizes:
        print(f"\nBenchmarking batch_size={batch_size} ...")
        result = run_one_benchmark(
            generator=generator,
            seed=args.seed,
            n_trajectories=args.n_trajectories,
            n_steps=args.n_steps,
            batch_size=batch_size,
        )
        results.append(result)
        print(
            f"  Total {result['total_generation_seconds']:.2f}s | "
            f"signal {result['signal_generation_seconds']:.2f}s | "
            f"steady_state {result['steady_state_seconds']:.2f}s | "
            f"rollout {result['rollout_seconds']:.2f}s | "
            f"fast_ss {result['fast_steady_states']} | "
            f"fallbacks {result['steady_state_fallbacks']}"
        )

    best = min(results, key=lambda result: result["total_generation_seconds"])
    recommended_default = DataGenerator.recommend_batch_size(backend)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.output_json is None or args.output_md is None:
        output_dir = os.path.join("outputs", "benchmarks")
        os.makedirs(output_dir, exist_ok=True)
        if args.output_json is None:
            args.output_json = os.path.join(output_dir, f"generation-benchmark-{timestamp}.json")
        if args.output_md is None:
            args.output_md = os.path.join(output_dir, f"generation-benchmark-{timestamp}.md")

    print("\nRecommendation")
    print(f"  Best observed batch size: {best['batch_size']}")
    print(f"  Observed total time: {best['total_generation_seconds']:.2f}s")
    print(f"  Conservative default for backend '{backend}': {recommended_default}")

    payload = {
        "backend": backend,
        "n_trajectories": args.n_trajectories,
        "n_steps": args.n_steps,
        "results": results,
        "best_batch_size": best["batch_size"],
        "recommended_default": recommended_default,
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
