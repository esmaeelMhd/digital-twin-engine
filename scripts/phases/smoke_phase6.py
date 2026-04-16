"""Run a reusable Phase 6 smoke-test matrix for the demo app and API."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_CONFIGS = (
    PROJECT_ROOT / "configs" / "cstr_default.yaml",
    PROJECT_ROOT / "configs" / "heat_exchanger_default.yaml",
    PROJECT_ROOT / "configs" / "two_tank_default.yaml",
)
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training_default.yaml"
DEFAULT_DEMO_CONFIG = PROJECT_ROOT / "configs" / "demo_app.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "definitely_missing.eqx"


class SmokeError(RuntimeError):
    """Raised when the smoke runner cannot complete successfully."""


def resolve_workspace_dir(raw_workspace: str | None) -> Path:
    if raw_workspace is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / "phase6_smoke" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _step_summary(name: str, started_at: float, succeeded: bool, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "started_at": datetime.fromtimestamp(started_at).isoformat(),
        "duration_seconds": time.time() - started_at,
        "succeeded": bool(succeeded),
        **{key: _json_safe(value) for key, value in extra.items()},
    }


def build_env(
    *,
    system_configs: list[Path],
    training_config: Path,
    demo_config: Path,
    model_path: Path,
    jax_platform: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["DTE_SYSTEM_CONFIG"] = ",".join(str(path.resolve()) for path in system_configs)
    env["DTE_TRAINING_CONFIG"] = str(training_config.resolve())
    env["DTE_DEMO_CONFIG"] = str(demo_config.resolve())
    env["DTE_MODEL_PATH"] = str(model_path.resolve())
    env["JAX_PLATFORMS"] = jax_platform
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def start_process(
    *,
    name: str,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    started_at = time.time()
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, _step_summary(
        name,
        started_at,
        succeeded=True,
        command=command,
        log_path=str(log_path),
        pid=process.pid,
    )


def stop_process(process: subprocess.Popen[str] | None, timeout: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def wait_for_http(
    session: requests.Session,
    *,
    url: str,
    timeout_seconds: float,
    expect_json: bool = False,
    expect_substring: str | None = None,
) -> requests.Response:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = session.get(url, timeout=2.0)
            if response.status_code >= 500:
                raise SmokeError(f"{url} returned status {response.status_code}")
            if expect_json:
                response.json()
            if expect_substring is not None and expect_substring not in response.text:
                raise SmokeError(f"{url} missing expected marker '{expect_substring}'")
            return response
        except Exception as exc:  # pragma: no cover - exercised in runtime smoke only
            last_error = exc
            time.sleep(0.5)
    raise SmokeError(f"Timed out waiting for {url}: {last_error}")


def request_json(
    session: requests.Session,
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = session.request(method, url, json=payload, timeout=15.0)
    if response.status_code != 200:
        raise SmokeError(
            f"{method} {url} failed with status {response.status_code}: {response.text[:500]}"
        )
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 6 demo smoke matrix.")
    parser.add_argument("--workspace_dir", type=str, default=None)
    parser.add_argument("--api_port", type=int, default=8016)
    parser.add_argument("--streamlit_port", type=int, default=8516)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--startup_timeout", type=float, default=45.0)
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--training_config", type=str, default=str(DEFAULT_TRAINING_CONFIG))
    parser.add_argument("--demo_config", type=str, default=str(DEFAULT_DEMO_CONFIG))
    parser.add_argument(
        "--system_config",
        action="append",
        dest="system_configs",
        default=None,
        help="May be passed multiple times. Defaults to the three Phase 6 demo systems.",
    )
    parser.add_argument("--jax_platform", type=str, default="cpu")
    parser.add_argument("--skip_streamlit", action="store_true")
    parser.add_argument("--skip_api", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    workspace_dir = resolve_workspace_dir(args.workspace_dir)
    logs_dir = workspace_dir / "logs"
    outputs_dir = workspace_dir / "outputs"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    system_configs = (
        [Path(item) for item in args.system_configs]
        if args.system_configs
        else list(DEFAULT_SYSTEM_CONFIGS)
    )
    training_config = Path(args.training_config)
    demo_config = Path(args.demo_config)
    model_path = Path(args.model_path)

    env = build_env(
        system_configs=system_configs,
        training_config=training_config,
        demo_config=demo_config,
        model_path=model_path,
        jax_platform=args.jax_platform,
    )

    api_url = f"http://{args.host}:{args.api_port}"
    streamlit_url = f"http://{args.host}:{args.streamlit_port}"

    api_command = [
        sys.executable,
        "-m",
        "dte.api.service",
        "--host",
        args.host,
        "--port",
        str(args.api_port),
    ]
    streamlit_command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/demo_app.py",
        "--server.headless",
        "true",
        "--server.port",
        str(args.streamlit_port),
        "--browser.gatherUsageStats",
        "false",
    ]

    summary: dict[str, Any] = {
        "status": "pending",
        "workspace_dir": str(workspace_dir),
        "api_url": api_url,
        "streamlit_url": streamlit_url,
        "system_configs": [str(path) for path in system_configs],
        "training_config": str(training_config),
        "demo_config": str(demo_config),
        "model_path": str(model_path),
        "jax_platform": args.jax_platform,
        "steps": [],
        "artifacts": {},
    }

    if args.dry_run:
        summary["status"] = "dry_run"
        summary["steps"] = [
            {
                "name": "api_process",
                "command": api_command,
                "skipped": bool(args.skip_api),
            },
            {
                "name": "streamlit_process",
                "command": streamlit_command,
                "skipped": bool(args.skip_streamlit),
            },
        ]
        write_json(workspace_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    session = requests.Session()
    api_process: subprocess.Popen[str] | None = None
    streamlit_process: subprocess.Popen[str] | None = None
    try:
        if not args.skip_api:
            api_process, step = start_process(
                name="start_api",
                command=api_command,
                env=env,
                log_path=logs_dir / "api.log",
            )
            summary["steps"].append(step)

            started_at = time.time()
            health_response = wait_for_http(
                session,
                url=f"{api_url}/health",
                timeout_seconds=float(args.startup_timeout),
                expect_json=True,
            )
            health_json = health_response.json()
            write_json(outputs_dir / "api_health.json", health_json)
            summary["steps"].append(
                _step_summary(
                    "wait_for_api_health",
                    started_at,
                    succeeded=True,
                    response_path=str(outputs_dir / "api_health.json"),
                    loaded_systems=health_json.get("loaded_systems", []),
                )
            )
            if len(health_json.get("loaded_systems", [])) < 3:
                raise SmokeError("API did not load the expected demo systems.")

            catalog_started = time.time()
            catalog_json = request_json(session, method="GET", url=f"{api_url}/demo/catalog")
            write_json(outputs_dir / "demo_catalog.json", catalog_json)
            summary["steps"].append(
                _step_summary(
                    "demo_catalog",
                    catalog_started,
                    succeeded=True,
                    response_path=str(outputs_dir / "demo_catalog.json"),
                    n_demos=len(catalog_json.get("demos", [])),
                    n_flowsheets=len(catalog_json.get("flowsheets", [])),
                )
            )
            if len(catalog_json.get("demos", [])) < 3:
                raise SmokeError("Demo catalog returned fewer than three demos.")

            simulate_started = time.time()
            simulate_json = request_json(
                session,
                method="POST",
                url=f"{api_url}/demo/simulate",
                payload={
                    "system": "two_tank",
                    "initial_state": [1.0, 0.8],
                    "controls": [[0.7, 0.82]] * 10,
                    "disturbances": [[0.6, 0.0]] * 10,
                    "dt": 0.1,
                },
            )
            write_json(outputs_dir / "demo_simulate_two_tank.json", simulate_json)
            summary["steps"].append(
                _step_summary(
                    "demo_simulate",
                    simulate_started,
                    succeeded=True,
                    response_path=str(outputs_dir / "demo_simulate_two_tank.json"),
                    n_steps=len(simulate_json.get("states", [])),
                )
            )

            rollout_started = time.time()
            rollout_json = request_json(
                session,
                method="POST",
                url=f"{api_url}/demo/rollout",
                payload={
                    "system": "heat_exchanger",
                    "initial_state": [360.0, 310.0],
                    "controls": [[4.5, 4.0]] * 12,
                    "disturbances": [[365.0, 295.0]] * 12,
                    "dt": 0.1,
                    "n_samples": 4,
                },
            )
            write_json(outputs_dir / "demo_rollout_heat_exchanger.json", rollout_json)
            summary["steps"].append(
                _step_summary(
                    "demo_rollout",
                    rollout_started,
                    succeeded=True,
                    response_path=str(outputs_dir / "demo_rollout_heat_exchanger.json"),
                    source=rollout_json.get("source"),
                )
            )

            optimize_started = time.time()
            optimize_json = request_json(
                session,
                method="POST",
                url=f"{api_url}/demo/optimize_control",
                payload={
                    "system": "cstr",
                    "initial_state": [0.5, 0.5, 350.0, 300.0],
                    "disturbances": [[1.0, 320.0]] * 12,
                    "target_state": [0.3, 0.8, 338.0, 304.0],
                    "tracked_state_names": ["Cb", "T"],
                    "dt": 0.1,
                    "n_candidates": 10,
                    "seed": 7,
                },
            )
            write_json(outputs_dir / "demo_optimize_cstr.json", optimize_json)
            summary["steps"].append(
                _step_summary(
                    "demo_optimize_control",
                    optimize_started,
                    succeeded=True,
                    response_path=str(outputs_dir / "demo_optimize_cstr.json"),
                    objective=optimize_json.get("objective"),
                )
            )

            compare_started = time.time()
            compare_json = request_json(
                session,
                method="POST",
                url=f"{api_url}/demo/compare_scenarios",
                payload={
                    "system": "cstr",
                    "initial_state": [0.5, 0.5, 350.0, 300.0],
                    "baseline_controls": [[50.0, 300.0]] * 12,
                    "candidate_controls": [[60.0, 295.0]] * 12,
                    "disturbances": [[1.0, 320.0]] * 12,
                    "dt": 0.1,
                    "n_samples": 4,
                    "seed": 9,
                },
            )
            write_json(outputs_dir / "demo_compare_cstr.json", compare_json)
            summary["steps"].append(
                _step_summary(
                    "demo_compare_scenarios",
                    compare_started,
                    succeeded=True,
                    response_path=str(outputs_dir / "demo_compare_cstr.json"),
                    final_state_delta_norm=compare_json.get("summary", {}).get("final_state_delta_norm"),
                )
            )

            summary["artifacts"].update(
                {
                    "api_health": str(outputs_dir / "api_health.json"),
                    "demo_catalog": str(outputs_dir / "demo_catalog.json"),
                    "demo_simulate": str(outputs_dir / "demo_simulate_two_tank.json"),
                    "demo_rollout": str(outputs_dir / "demo_rollout_heat_exchanger.json"),
                    "demo_optimize": str(outputs_dir / "demo_optimize_cstr.json"),
                    "demo_compare": str(outputs_dir / "demo_compare_cstr.json"),
                }
            )

        if not args.skip_streamlit:
            streamlit_process, step = start_process(
                name="start_streamlit",
                command=streamlit_command,
                env=env,
                log_path=logs_dir / "streamlit.log",
            )
            summary["steps"].append(step)

            health_started = time.time()
            st_health = wait_for_http(
                session,
                url=f"{streamlit_url}/_stcore/health",
                timeout_seconds=float(args.startup_timeout),
                expect_substring="ok",
            )
            write_text(outputs_dir / "streamlit_health.txt", st_health.text)
            summary["steps"].append(
                _step_summary(
                    "streamlit_health",
                    health_started,
                    succeeded=True,
                    response_path=str(outputs_dir / "streamlit_health.txt"),
                )
            )

            root_started = time.time()
            root_response = wait_for_http(
                session,
                url=streamlit_url,
                timeout_seconds=15.0,
                expect_substring="<!DOCTYPE html>",
            )
            write_text(outputs_dir / "streamlit_root.html", root_response.text)
            summary["steps"].append(
                _step_summary(
                    "streamlit_root",
                    root_started,
                    succeeded=True,
                    response_path=str(outputs_dir / "streamlit_root.html"),
                    status_code=root_response.status_code,
                )
            )
            summary["artifacts"].update(
                {
                    "streamlit_health": str(outputs_dir / "streamlit_health.txt"),
                    "streamlit_root": str(outputs_dir / "streamlit_root.html"),
                }
            )

        summary["status"] = "ok"
        write_json(workspace_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_json(workspace_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        stop_process(streamlit_process)
        stop_process(api_process)
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
