"""Tests for the multi-system corpus generation manifest helpers."""

from scripts.generate_corpus import build_generation_jobs, load_corpus_manifest


def test_phase1_regime_manifest_resolves_all_expected_systems():
    manifest = load_corpus_manifest("configs/generation_phase1_regime.yaml")

    jobs = build_generation_jobs(manifest)

    assert len(jobs) == 14
    assert jobs[0]["name"] == "cstr"
    assert jobs[-1]["name"] == "bioreactor_compartment_high_transfer"
    assert all("data/phase1_regime/" in str(job["output_dir"]) for job in jobs)
    assert {job["n_trajectories"] for job in jobs} == {256}
    assert {job["n_steps"] for job in jobs} == {96}


def test_build_generation_jobs_output_root_override_replaces_explicit_output_dirs():
    manifest = {
        "defaults": {
            "n_trajectories": 10,
            "n_steps": 16,
            "simulation_mode": "dataset",
            "output_root": "data/default_root",
        },
        "systems": [
            {
                "name": "cstr_fast_kinetics",
                "system_config": "configs/cstr_fast_kinetics_hot_feed.yaml",
                "output_dir": "data/should_be_overridden",
            }
        ],
    }

    jobs = build_generation_jobs(
        manifest,
        output_root_override="/tmp/corpus_override",
    )

    assert len(jobs) == 1
    assert jobs[0]["name"] == "cstr_fast_kinetics"
    assert str(jobs[0]["output_dir"]) == "/tmp/corpus_override/cstr_fast_kinetics"
