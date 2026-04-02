import scripts.agent as agent_module

from scripts.agent import (
    _choose_file,
    _normalize_gemini_thinking_level,
    build_dynamic_prompt,
    build_prompt,
    build_static_prompt_context,
    rollback_experiment_change,
)


def test_normalize_gemini_thinking_level_keeps_medium_on_pro():
    assert _normalize_gemini_thinking_level("gemini-3.1-pro-preview", "medium") == "medium"


def test_normalize_gemini_thinking_level_keeps_medium_on_flash():
    assert _normalize_gemini_thinking_level("gemini-3.1-flash-lite-preview", "medium") == "medium"


def test_normalize_gemini_thinking_level_downgrades_minimal_on_pro():
    assert _normalize_gemini_thinking_level("gemini-3.1-pro-preview", "minimal") == "low"


def test_build_static_prompt_context_mentions_search_replace_contract():
    prompt = build_static_prompt_context(["foo.py"], agent_context="repo ctx")

    assert "foo.py" in prompt
    assert "repo ctx" in prompt
    assert "SEARCH/REPLACE" in prompt


def test_build_prompt_is_static_plus_dynamic_sections():
    static_prompt = build_static_prompt_context(["foo.py"], agent_context="repo ctx")
    dynamic_prompt = build_dynamic_prompt(
        "foo.py",
        "print('hello')\n",
        [],
        1.2345,
        recent_run_context="recent runs",
    )

    full_prompt = build_prompt(
        "foo.py",
        "print('hello')\n",
        [],
        1.2345,
        ["foo.py"],
        agent_context="repo ctx",
        recent_run_context="recent runs",
    )

    assert full_prompt == static_prompt + dynamic_prompt


def test_choose_file_prefers_clean_candidates(monkeypatch):
    monkeypatch.setattr(
        agent_module,
        "git_file_is_dirty",
        lambda path: path == "dirty.py",
    )

    chosen = _choose_file([], ["dirty.py", "clean.py"], None)

    assert chosen == "clean.py"


def test_rollback_experiment_change_restores_original_content(tmp_path, monkeypatch):
    target = tmp_path / "demo.py"
    target.write_text("modified\n", encoding="utf-8")
    git_calls = []

    monkeypatch.setattr(agent_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(agent_module, "git", lambda *args: git_calls.append(args) or ("", 0))

    rollback_experiment_change("demo.py", "original\n", committed=False)

    assert target.read_text(encoding="utf-8") == "original\n"
    assert ("restore", "--staged", "--", "demo.py") in git_calls


def test_rollback_experiment_change_resets_commit_before_restore(tmp_path, monkeypatch):
    target = tmp_path / "demo.py"
    target.write_text("modified\n", encoding="utf-8")
    sequence = []

    monkeypatch.setattr(agent_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        agent_module,
        "git_reset_last_commit",
        lambda: sequence.append("reset"),
    )
    monkeypatch.setattr(
        agent_module,
        "git",
        lambda *args: sequence.append(("git", args)) or ("", 0),
    )

    rollback_experiment_change("demo.py", "original\n", committed=True)

    assert sequence[0] == "reset"
    assert target.read_text(encoding="utf-8") == "original\n"
