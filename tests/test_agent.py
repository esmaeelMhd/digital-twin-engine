import scripts.agent as agent_module

from scripts.agent import (
    _call_gemini_with_cache_retry,
    _choose_file,
    _is_gemini_cache_miss_error,
    _normalize_gemini_thinking_level,
    annotate_description_with_idea,
    FatalAPIError,
    build_dynamic_prompt,
    build_prompt,
    build_static_prompt_context,
    load_structured_ideas,
    rollback_experiment_change,
    select_next_structured_idea,
)


def test_normalize_gemini_thinking_level_keeps_medium_on_pro():
    assert _normalize_gemini_thinking_level("gemini-3.1-pro-preview", "medium") == "medium"


def test_normalize_gemini_thinking_level_keeps_medium_on_flash():
    assert _normalize_gemini_thinking_level("gemini-3.1-flash-lite-preview", "medium") == "medium"


def test_normalize_gemini_thinking_level_downgrades_minimal_on_pro():
    assert _normalize_gemini_thinking_level("gemini-3.1-pro-preview", "minimal") == "low"


def test_is_gemini_cache_miss_error_detects_missing_cached_content():
    assert _is_gemini_cache_miss_error(
        "403 PERMISSION_DENIED. CachedContent not found (or permission denied)"
    )


def test_call_gemini_with_cache_retry_refreshes_and_retries(monkeypatch):
    calls = []

    def fake_call(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        if len(calls) == 1:
            raise FatalAPIError("403 PERMISSION_DENIED. CachedContent not found (or permission denied)")
        return '{"ok": true}'

    refresh_calls = []

    monkeypatch.setattr(agent_module, "log_to_file", lambda msg: None)

    result = _call_gemini_with_cache_retry(
        fake_call,
        "hello",
        kwargs={"cached_content": "cachedContents/old", "temperature": 0.1},
        refresh_cache=lambda: refresh_calls.append("refresh") or "cachedContents/new",
    )

    assert result == '{"ok": true}'
    assert refresh_calls == ["refresh"]
    assert calls[0]["cached_content"] == "cachedContents/old"
    assert calls[1]["cached_content"] == "cachedContents/new"


def test_call_gemini_with_cache_retry_falls_back_to_uncached_prompt(monkeypatch):
    calls = []

    def fake_call(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        if len(calls) == 1:
            raise FatalAPIError("403 PERMISSION_DENIED. CachedContent not found (or permission denied)")
        return '{"ok": true}'

    monkeypatch.setattr(agent_module, "log_to_file", lambda msg: None)

    result = _call_gemini_with_cache_retry(
        fake_call,
        "hello",
        kwargs={"cached_content": "cachedContents/old", "temperature": 0.1},
        refresh_cache=lambda: None,
    )

    assert result == '{"ok": true}'
    assert calls[0]["cached_content"] == "cachedContents/old"
    assert "cached_content" not in calls[1]


def test_build_static_prompt_context_mentions_search_replace_contract():
    prompt = build_static_prompt_context(["foo.py"], agent_context="repo ctx")

    assert "foo.py" in prompt
    assert "repo ctx" in prompt
    assert "SEARCH/REPLACE" in prompt
    assert "Do NOT focus on routine hyperparameter optimization" in prompt
    assert "Small hyperparameter changes are allowed only when they support a larger" in prompt


def test_build_prompt_is_static_plus_dynamic_sections():
    static_prompt = build_static_prompt_context(["foo.py"], agent_context="repo ctx")
    dynamic_prompt = build_dynamic_prompt(
        "foo.py",
        "print('hello')\n",
        [],
        1.2345,
        recent_run_context="recent runs",
        ideas=[],
        selected_idea=None,
    )

    full_prompt = build_prompt(
        "foo.py",
        "print('hello')\n",
        [],
        1.2345,
        ["foo.py"],
        agent_context="repo ctx",
        recent_run_context="recent runs",
        ideas=[],
        selected_idea=None,
    )

    assert full_prompt == static_prompt + dynamic_prompt


def test_build_dynamic_prompt_includes_selected_structured_idea():
    prompt = build_dynamic_prompt(
        "foo.py",
        "print('hello')\n",
        [],
        1.2345,
        recent_run_context="recent runs",
        ideas=[
            {
                "id": "idea-1",
                "title": "Try a better residual path",
                "target_file": "foo.py",
                "priority": 1,
                "rationale": "Test rationale",
                "instructions": ["Keep it small"],
                "tags": ["residual"],
            }
        ],
        selected_idea={
            "id": "idea-1",
            "title": "Try a better residual path",
            "target_file": "foo.py",
            "priority": 1,
            "rationale": "Test rationale",
            "instructions": ["Keep it small"],
            "tags": ["residual"],
        },
    )

    assert "Structured idea backlog" in prompt
    assert "Required next idea" in prompt
    assert '"idea_id": "optional structured idea identifier' in prompt


def test_choose_file_prefers_clean_candidates(monkeypatch):
    monkeypatch.setattr(
        agent_module,
        "git_file_is_dirty",
        lambda path: path == "dirty.py",
    )

    chosen = _choose_file([], ["dirty.py", "clean.py"], None)

    assert chosen == "clean.py"


def test_choose_file_prefers_code_over_config_when_counts_tie(monkeypatch):
    monkeypatch.setattr(agent_module, "git_file_is_dirty", lambda path: False)

    chosen = _choose_file([], ["configs/training_default.yaml", "dte/training/trainer.py"], None)

    assert chosen == "dte/training/trainer.py"


def test_choose_file_prefers_structured_idea_target(monkeypatch):
    monkeypatch.setattr(agent_module, "git_file_is_dirty", lambda path: False)

    chosen = _choose_file(
        [],
        ["dte/models/encoder.py", "dte/training/trainer.py"],
        None,
        preferred_file="dte/models/encoder.py",
    )

    assert chosen == "dte/models/encoder.py"


def test_select_next_structured_idea_skips_history_attempts():
    ideas = [
        {"id": "idea-1", "title": "First idea", "target_file": "foo.py", "priority": 1},
        {"id": "idea-2", "title": "Second idea", "target_file": "foo.py", "priority": 2},
    ]
    history = [
        {"description": "[idea:idea-1] First idea", "file": "foo.py", "status": "discard", "val_loss": 1.0},
    ]

    selected = select_next_structured_idea(ideas, history, ["foo.py"], None)

    assert selected["id"] == "idea-2"


def test_load_structured_ideas_reads_yaml(tmp_path):
    ideas_path = tmp_path / "ideas.yaml"
    ideas_path.write_text(
        "ideas:\n"
        "  - id: custom_one\n"
        "    title: First custom idea\n"
        "    target_file: foo.py\n"
        "    priority: 2\n"
        "    instructions:\n"
        "      - Keep it surgical\n",
        encoding="utf-8",
    )

    ideas = load_structured_ideas(override_path=str(ideas_path))

    assert ideas[0]["id"] == "custom_one"
    assert ideas[0]["target_file"] == "foo.py"
    assert ideas[0]["instructions"] == ["Keep it surgical"]


def test_annotate_description_with_idea_is_idempotent():
    description = annotate_description_with_idea("Add diffusion floor", "latent_diffusion_floor")

    assert description == "[idea:latent_diffusion_floor] Add diffusion floor"
    assert (
        annotate_description_with_idea(description, "latent_diffusion_floor")
        == description
    )


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


def test_rollback_experiment_change_reverts_non_head_experiment_commit(tmp_path, monkeypatch):
    target = tmp_path / "demo.py"
    target.write_text("modified\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(agent_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(agent_module, "git_head_sha", lambda: "head999")
    monkeypatch.setattr(agent_module, "git_is_ancestor", lambda ancestor, descendant="HEAD": True)
    monkeypatch.setattr(agent_module, "git_revert_commit", lambda sha: calls.append(("revert", sha)))
    monkeypatch.setattr(agent_module, "git", lambda *args: calls.append(("git", args)) or ("", 0))

    rollback_experiment_change("demo.py", "original\n", committed=True, commit_sha="abc1234")

    assert ("revert", "abc1234") in calls
    assert target.read_text(encoding="utf-8") == "original\n"
