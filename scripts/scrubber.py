#!/usr/bin/env python3
"""Safely apply Gemini SEARCH/REPLACE blocks to local files.

Expected input format:

ANALYSIS: optional free-form text

FILE: path/to/file.py
<<<<<<< SEARCH
exact old text
=======
replacement text
>>>>>>> REPLACE

Additional SEARCH/REPLACE blocks may follow, optionally reusing the same FILE.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScrubberError(Exception):
    """Raised when SEARCH/REPLACE output cannot be applied safely."""


@dataclass(frozen=True)
class SearchReplaceBlock:
    file_path: str
    search: str
    replace: str
    index: int


@dataclass(frozen=True)
class FilePatchPlan:
    relative_path: str
    absolute_path: Path
    original: str
    modified: str
    block_count: int


def parse_search_replace_output(text: str) -> list[SearchReplaceBlock]:
    """Parse Gemini-style FILE + SEARCH/REPLACE blocks from raw text."""

    lines = text.splitlines()
    current_file: str | None = None
    blocks: list[SearchReplaceBlock] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]

        if line.startswith("FILE:"):
            current_file = line.split(":", 1)[1].strip()
            if not current_file:
                raise ScrubberError("Encountered an empty FILE: header.")
            idx += 1
            continue

        if line.strip() != "<<<<<<< SEARCH":
            idx += 1
            continue

        if not current_file:
            raise ScrubberError("Found SEARCH/REPLACE block before any FILE: header.")

        idx += 1
        search_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip() != "=======":
            search_lines.append(lines[idx])
            idx += 1
        if idx >= len(lines):
            raise ScrubberError(f"Block {len(blocks) + 1} is missing the ======= separator.")

        idx += 1
        replace_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip() != ">>>>>>> REPLACE":
            replace_lines.append(lines[idx])
            idx += 1
        if idx >= len(lines):
            raise ScrubberError(f"Block {len(blocks) + 1} is missing the >>>>>>> REPLACE footer.")

        idx += 1
        search = "\n".join(search_lines)
        replace = "\n".join(replace_lines)
        if not search:
            raise ScrubberError(f"Block {len(blocks) + 1} has an empty SEARCH section.")

        blocks.append(
            SearchReplaceBlock(
                file_path=current_file,
                search=search,
                replace=replace,
                index=len(blocks) + 1,
            )
        )

    if not blocks:
        raise ScrubberError("No SEARCH/REPLACE blocks were found in the input.")

    return blocks


def resolve_target_path(repo_root: Path, raw_path: str) -> tuple[str, Path]:
    """Resolve a repo-relative file path and reject path escapes."""

    candidate = Path(raw_path.strip())
    if not raw_path.strip():
        raise ScrubberError("Encountered an empty target path.")
    if candidate.is_absolute():
        raise ScrubberError(f"Absolute paths are not allowed: {raw_path}")

    repo_root_resolved = repo_root.resolve()
    target_path = (repo_root_resolved / candidate).resolve()
    try:
        relative = target_path.relative_to(repo_root_resolved)
    except ValueError as exc:
        raise ScrubberError(f"Path escapes repo root: {raw_path}") from exc

    return relative.as_posix(), target_path


def _read_text(path: Path, encoding: str) -> str:
    try:
        return path.read_text(encoding=encoding)
    except FileNotFoundError as exc:
        raise ScrubberError(f"Target file does not exist: {path}") from exc


def build_patch_plan(
    blocks: list[SearchReplaceBlock],
    *,
    repo_root: Path = PROJECT_ROOT,
    encoding: str = "utf-8",
    allowed_files: set[str] | None = None,
) -> list[FilePatchPlan]:
    """Validate all blocks and build an all-or-nothing application plan."""

    working_texts: dict[Path, str] = {}
    originals: dict[Path, str] = {}
    relative_paths: dict[Path, str] = {}
    block_counts: dict[Path, int] = {}
    ordered_paths: list[Path] = []

    normalized_allowed = None
    if allowed_files is not None:
        normalized_allowed = {Path(path).as_posix() for path in allowed_files}

    for block in blocks:
        relative_path, absolute_path = resolve_target_path(repo_root, block.file_path)
        if normalized_allowed is not None and relative_path not in normalized_allowed:
            raise ScrubberError(
                f"Block {block.index} targets disallowed file: {relative_path}"
            )

        if absolute_path not in working_texts:
            original = _read_text(absolute_path, encoding)
            working_texts[absolute_path] = original
            originals[absolute_path] = original
            relative_paths[absolute_path] = relative_path
            block_counts[absolute_path] = 0
            ordered_paths.append(absolute_path)

        current = working_texts[absolute_path]
        count = current.count(block.search)
        if count == 0:
            raise ScrubberError(
                f"Block {block.index} SEARCH did not match in {relative_path}."
            )
        if count > 1:
            raise ScrubberError(
                f"Block {block.index} SEARCH matched {count} times in {relative_path}; "
                "refine the SEARCH block so it is unique."
            )

        updated = current.replace(block.search, block.replace, 1)
        if updated == current:
            raise ScrubberError(
                f"Block {block.index} is a no-op in {relative_path}; SEARCH and REPLACE are identical."
            )

        working_texts[absolute_path] = updated
        block_counts[absolute_path] += 1

    plans: list[FilePatchPlan] = []
    for absolute_path in ordered_paths:
        plans.append(
            FilePatchPlan(
                relative_path=relative_paths[absolute_path],
                absolute_path=absolute_path,
                original=originals[absolute_path],
                modified=working_texts[absolute_path],
                block_count=block_counts[absolute_path],
            )
        )
    return plans


def render_diff(plan: FilePatchPlan) -> str:
    """Render a unified diff for a planned file modification."""

    diff_lines = difflib.unified_diff(
        plan.original.splitlines(),
        plan.modified.splitlines(),
        fromfile=plan.relative_path,
        tofile=plan.relative_path,
        lineterm="",
    )
    return "\n".join(diff_lines)


def _atomic_write_text(path: Path, text: str, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding=encoding,
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def apply_patch_plan(plans: list[FilePatchPlan], *, encoding: str = "utf-8") -> None:
    """Write all file modifications, rolling back if any write fails."""

    written: list[FilePatchPlan] = []
    try:
        for plan in plans:
            _atomic_write_text(plan.absolute_path, plan.modified, encoding)
            written.append(plan)
    except Exception:
        for plan in reversed(written):
            try:
                _atomic_write_text(plan.absolute_path, plan.original, encoding)
            except Exception:
                pass
        raise


def load_input_text(input_path: str | None, encoding: str) -> str:
    """Load SEARCH/REPLACE text from a file or stdin."""

    if input_path:
        return Path(input_path).read_text(encoding=encoding)
    if sys.stdin.isatty():
        raise ScrubberError("No input provided. Pass --input or pipe Gemini output on stdin.")
    return sys.stdin.read()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely apply Gemini SEARCH/REPLACE output to local files."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a file containing Gemini SEARCH/REPLACE output. Defaults to stdin.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Repository root used to resolve FILE: paths.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text encoding for reading the patch payload and target files.",
    )
    parser.add_argument(
        "--allow-file",
        action="append",
        default=[],
        help="Restrict modifications to this repo-relative file. Repeat to allow multiple files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print diffs without writing files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    try:
        text = load_input_text(args.input, args.encoding)
        blocks = parse_search_replace_output(text)
        plans = build_patch_plan(
            blocks,
            repo_root=repo_root,
            encoding=args.encoding,
            allowed_files=set(args.allow_file) if args.allow_file else None,
        )
    except ScrubberError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for plan in plans:
        diff_text = render_diff(plan)
        if diff_text:
            print(diff_text)
            print()

    if args.dry_run:
        print(f"Dry run OK: {len(plans)} file(s), {sum(plan.block_count for plan in plans)} block(s).")
        return 0

    try:
        apply_patch_plan(plans, encoding=args.encoding)
    except Exception as exc:
        print(f"ERROR: failed to write files safely: {exc}", file=sys.stderr)
        return 1

    print(f"Applied {sum(plan.block_count for plan in plans)} block(s) across {len(plans)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
