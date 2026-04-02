from pathlib import Path

from scripts.scrubber import (
    ScrubberError,
    apply_patch_plan,
    build_patch_plan,
    main,
    parse_search_replace_output,
    resolve_target_path,
)


def test_parse_search_replace_output_parses_multiple_blocks() -> None:
    payload = """ANALYSIS: tighten loss schedule

FILE: configs/training_default.yaml
<<<<<<< SEARCH
peak_lr: 5.0e-4
=======
peak_lr: 3.0e-4
>>>>>>> REPLACE
<<<<<<< SEARCH
gradient_clip: 0.5
=======
gradient_clip: 0.3
>>>>>>> REPLACE
"""

    blocks = parse_search_replace_output(payload)

    assert len(blocks) == 2
    assert blocks[0].file_path == "configs/training_default.yaml"
    assert blocks[0].search == "peak_lr: 5.0e-4"
    assert blocks[1].replace == "gradient_clip: 0.3"


def test_resolve_target_path_rejects_repo_escape(tmp_path: Path) -> None:
    try:
        resolve_target_path(tmp_path, "../outside.py")
    except ScrubberError as exc:
        assert "escapes repo root" in str(exc)
    else:
        raise AssertionError("Expected resolve_target_path to reject repo escape.")


def test_build_patch_plan_rejects_ambiguous_match(tmp_path: Path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    payload = """FILE: demo.py
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
"""

    blocks = parse_search_replace_output(payload)

    try:
        build_patch_plan(blocks, repo_root=tmp_path)
    except ScrubberError as exc:
        assert "matched 2 times" in str(exc)
    else:
        raise AssertionError("Expected ambiguous SEARCH block to fail.")


def test_apply_patch_plan_updates_file(tmp_path: Path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("value = 1\nprint(value)\n", encoding="utf-8")
    payload = """FILE: demo.py
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
"""

    plans = build_patch_plan(parse_search_replace_output(payload), repo_root=tmp_path)
    apply_patch_plan(plans)

    assert target.read_text(encoding="utf-8") == "value = 2\nprint(value)\n"


def test_main_dry_run_respects_allow_file(tmp_path: Path, capsys) -> None:
    target = tmp_path / "demo.py"
    target.write_text("value = 1\n", encoding="utf-8")
    payload_path = tmp_path / "gemini.txt"
    payload_path.write_text(
        """FILE: demo.py
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
""",
        encoding="utf-8",
    )

    code = main(
        [
            "--input",
            str(payload_path),
            "--repo-root",
            str(tmp_path),
            "--allow-file",
            "demo.py",
            "--dry-run",
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    assert "--- demo.py" in captured.out
    assert "Dry run OK" in captured.out
    assert target.read_text(encoding="utf-8") == "value = 1\n"
