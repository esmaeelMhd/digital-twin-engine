"""Coded phase registry and acceptance extraction for convergence work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PhaseRunStatus:
    """Machine-readable status for one canonical phase workspace."""

    phase_id: str
    summary_path: Path
    status: str
    accepted: bool
    error: str | None
    gates: dict[str, bool]


@dataclass(frozen=True)
class PhaseSpec:
    """Definition of one convergence-program phase."""

    phase_id: str
    title: str
    default_workspace: str
    summary_relpath: str
    editable_targets: tuple[str, ...]
    forbidden_paths: tuple[str, ...]

    def resolve_workspace(self, raw_workspace: str | None = None) -> Path:
        if raw_workspace:
            path = Path(raw_workspace)
            return path if path.is_absolute() else PROJECT_ROOT / path
        return PROJECT_ROOT / self.default_workspace

    def summary_path_for_workspace(self, workspace_dir: Path) -> Path:
        return workspace_dir / self.summary_relpath

    def build_command(
        self,
        *,
        workspace_dir: Path,
        jax_platform: str,
        skip_generation: bool,
        skip_training: bool = False,
        skip_evaluation: bool = False,
        skip_transfer: bool = False,
        skip_control: bool = False,
        transfer_targets: list[str] | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        if self.phase_id != "phase1_unit_foundation_v1":
            raise ValueError(f"Unsupported phase command: {self.phase_id}")

        command = [
            "python",
            "scripts/run_unit_foundation_baseline.py",
            "--workspace_dir",
            str(workspace_dir),
            "--jax_platform",
            jax_platform,
        ]
        if skip_generation:
            command.append("--skip_generation")
        if skip_training:
            command.append("--skip_training")
        if skip_evaluation:
            command.append("--skip_evaluation")
        if skip_transfer:
            command.append("--skip_transfer")
        if skip_control:
            command.append("--skip_control")
        if transfer_targets:
            command.extend(["--transfer_targets", *transfer_targets])
        if dry_run:
            command.append("--dry_run")
        return command


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _phase1_status(spec: PhaseSpec, workspace_dir: Path) -> PhaseRunStatus:
    summary_path = spec.summary_path_for_workspace(workspace_dir)
    if not summary_path.exists():
        return PhaseRunStatus(
            phase_id=spec.phase_id,
            summary_path=summary_path,
            status="missing",
            accepted=False,
            error=None,
            gates={},
        )

    payload = _read_json(summary_path)
    raw_status = str(payload.get("status", "unknown")).strip().lower()
    acceptance = payload.get("acceptance", {}) if isinstance(payload.get("acceptance"), dict) else {}
    gates = {
        str(name): bool(value)
        for name, value in (acceptance.get("gates") or {}).items()
    }
    accepted = bool(acceptance.get("accepted", False))
    error = payload.get("error")

    if raw_status == "ok" and accepted:
        status = "accepted"
    elif raw_status == "ok":
        status = "not_accepted"
    elif raw_status in {"running", "failed", "dry_run"}:
        status = raw_status
    else:
        status = "unknown"

    return PhaseRunStatus(
        phase_id=spec.phase_id,
        summary_path=summary_path,
        status=status,
        accepted=accepted,
        error=str(error) if error is not None else None,
        gates=gates,
    )


_PHASES: dict[str, PhaseSpec] = {
    "phase1_unit_foundation_v1": PhaseSpec(
        phase_id="phase1_unit_foundation_v1",
        title="Phase 1: Unit Foundation V1",
        default_workspace="outputs/unit_foundation_phase1",
        summary_relpath="summary.json",
        editable_targets=(
            "configs/generation_phase1_regime.yaml",
            "configs/training_universal_phase1_regime.yaml",
            "scripts/run_unit_foundation_baseline.py",
            "scripts/train_universal.py",
            "scripts/evaluate_universal.py",
            "dte/models/universal/",
            "dte/training/universal/",
            "dte/calibration/",
            "tests/test_universal_digital_twin.py",
            "tests/test_unit_foundation_baseline.py",
            "tests/test_unit_calibration.py",
            "tests/test_universal_regime_config.py",
        ),
        forbidden_paths=(
            "scripts/autoresearch.py",
            "dte/autoresearch/",
            "scripts/agent.py",
            "auto_research.md",
            "program.md",
            "legacy/",
        ),
    ),
}


def list_phase_ids() -> list[str]:
    return sorted(_PHASES)


def get_phase_spec(phase_id: str) -> PhaseSpec:
    try:
        return _PHASES[phase_id]
    except KeyError as exc:
        raise KeyError(f"Unknown convergence phase '{phase_id}'.") from exc


def read_phase_status(phase_id: str, workspace_dir: str | Path | None = None) -> PhaseRunStatus:
    spec = get_phase_spec(phase_id)
    resolved_workspace = (
        spec.resolve_workspace(str(workspace_dir))
        if workspace_dir is not None
        else spec.resolve_workspace(None)
    )
    if spec.phase_id == "phase1_unit_foundation_v1":
        return _phase1_status(spec, resolved_workspace)
    raise ValueError(f"Unsupported phase status reader: {spec.phase_id}")
