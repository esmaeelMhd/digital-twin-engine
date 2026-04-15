"""Tests for the non-smoke Phase 1 regime universal config."""

import yaml

from scripts.train_universal import _load_sources


def test_phase1_regime_training_config_resolves_all_sources():
    with open("configs/training_universal_phase1_regime.yaml", "r") as handle:
        config = yaml.safe_load(handle)

    sources = _load_sources(config)

    assert len(sources) == 14
    assert sources[0].name == "cstr"
    assert sources[-1].name == "bioreactor_compartment_high_transfer"
    assert all(source.data_dir.startswith("data/phase1_regime/") for source in sources)
