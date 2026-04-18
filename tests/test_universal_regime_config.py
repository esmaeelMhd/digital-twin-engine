"""Tests for the non-smoke Phase 1 regime universal configs."""

import yaml

from scripts.train_universal import _load_sources


def _load_config(path: str) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def test_phase1_regime_training_config_resolves_all_sources():
    config = _load_config("configs/training_universal_phase1_regime.yaml")
    sources = _load_sources(config)

    assert len(sources) == 14
    assert sources[0].name == "cstr"
    assert sources[-1].name == "bioreactor_compartment_high_transfer"
    assert all(source.data_dir.startswith("data/phase1_regime/") for source in sources)


def test_phase1_regime_training_config_targets_current_rollout_outliers():
    config = _load_config("configs/training_universal_phase1_regime.yaml")
    weights = {item["name"]: float(item["weight"]) for item in config["data"]["systems"]}
    terms = {item["name"]: item for item in config["system_specific_losses"]["role_derivative_terms"]}

    assert float(config["loss_weights"]["k_step"]) == 0.25
    assert weights["cstr"] == 1.25
    assert weights["cstr_fast_kinetics"] == 1.75
    assert weights["separator"] == 1.5
    assert weights["separator_sharp_split"] == 2.0

    assert terms["reactor_species_dynamics"]["systems"] == ["cstr", "cstr_fast_kinetics"]
    assert terms["reactor_species_dynamics"]["state_role"] == "concentration"
    assert float(terms["reactor_species_dynamics"]["weight"]) == 0.35

    assert terms["separator_cut_dynamics"]["systems"] == ["separator", "separator_sharp_split"]
    assert terms["separator_cut_dynamics"]["state_role"] == "concentration"
    assert float(terms["separator_cut_dynamics"]["weight"]) == 0.35

    assert terms["separator_thermal_dynamics"]["systems"] == ["separator", "separator_sharp_split"]
    assert terms["separator_thermal_dynamics"]["state_role"] == "temperature"
    assert float(terms["separator_thermal_dynamics"]["weight"]) == 0.15


def test_phase1_regime_rebalanced_training_config_resolves_all_sources():
    config = _load_config("configs/training_universal_phase1_regime_rebalanced.yaml")
    sources = _load_sources(config)

    assert len(sources) == 14
    assert sources[0].name == "cstr"
    assert sources[-1].name == "bioreactor_compartment_high_transfer"
    assert all(source.data_dir.startswith("data/phase1_regime/") for source in sources)


def test_phase1_regime_rebalanced_training_config_upweights_target_systems():
    config = _load_config("configs/training_universal_phase1_regime_rebalanced.yaml")
    weights = {item["name"]: float(item["weight"]) for item in config["data"]["systems"]}

    assert weights["storage_tank"] == 2.5
    assert weights["storage_tank_high_holdup"] == 2.5
    assert weights["separator"] == 2.0
    assert weights["separator_sharp_split"] == 2.0
    assert weights["isothermal_cstr"] == 1.5
    assert weights["isothermal_cstr_slow_kinetics"] == 1.5
    assert weights["heat_exchanger_high_ua"] == 1.5
    assert weights["cstr"] == 1.0
    assert weights["two_tank"] == 1.0


def test_phase1_regime_rebalanced_gpu_training_config_resolves_all_sources():
    config = _load_config("configs/training_universal_phase1_regime_rebalanced_gpu.yaml")
    sources = _load_sources(config)

    assert len(sources) == 14
    assert sources[0].name == "cstr"
    assert sources[-1].name == "bioreactor_compartment_high_transfer"
    assert all(source.data_dir.startswith("data/phase1_regime/") for source in sources)


def test_phase1_regime_rebalanced_gpu_training_budget_is_larger_than_cpu_variant():
    cpu_cfg = _load_config("configs/training_universal_phase1_regime_rebalanced.yaml")
    gpu_cfg = _load_config("configs/training_universal_phase1_regime_rebalanced_gpu.yaml")

    assert gpu_cfg["training"]["batch_size"] == cpu_cfg["training"]["batch_size"]
    assert gpu_cfg["training"]["n_epochs"] > cpu_cfg["training"]["n_epochs"]
    assert gpu_cfg["training"]["max_batches_per_epoch"] > cpu_cfg["training"]["max_batches_per_epoch"]
    assert gpu_cfg["optimizer"]["total_steps"] > cpu_cfg["optimizer"]["total_steps"]
    assert gpu_cfg["checkpointing"]["max_val_batches"] > cpu_cfg["checkpointing"]["max_val_batches"]
    assert gpu_cfg["evaluation"]["per_system_batches"] > cpu_cfg["evaluation"]["per_system_batches"]


def test_phase1_regime_storage_focus_training_config_resolves_all_sources():
    config = _load_config("configs/training_universal_phase1_regime_storage_focus.yaml")
    sources = _load_sources(config)

    assert len(sources) == 14
    assert sources[0].name == "cstr"
    assert sources[-1].name == "bioreactor_compartment_high_transfer"
    assert all(source.data_dir.startswith("data/phase1_regime/") for source in sources)


def test_phase1_regime_storage_focus_training_config_only_upweights_storage_pair():
    config = _load_config("configs/training_universal_phase1_regime_storage_focus.yaml")
    weights = {item["name"]: float(item["weight"]) for item in config["data"]["systems"]}

    assert weights["storage_tank"] == 2.0
    assert weights["storage_tank_high_holdup"] == 2.0
    assert weights["cstr"] == 1.0
    assert weights["cstr_fast_kinetics"] == 1.0
    assert weights["separator"] == 1.0
    assert weights["separator_sharp_split"] == 1.0
    assert weights["heat_exchanger"] == 1.0
    assert weights["heat_exchanger_high_ua"] == 1.0


def test_phase1_regime_storage_dynamics_training_config_resolves_all_sources():
    config = _load_config("configs/training_universal_phase1_regime_storage_dynamics.yaml")
    sources = _load_sources(config)

    assert len(sources) == 14
    assert sources[0].name == "cstr"
    assert sources[-1].name == "bioreactor_compartment_high_transfer"
    assert all(source.data_dir.startswith("data/phase1_regime/") for source in sources)


def test_phase1_regime_storage_dynamics_training_config_targets_storage_inventory():
    config = _load_config("configs/training_universal_phase1_regime_storage_dynamics.yaml")
    terms = config["system_specific_losses"]["role_derivative_terms"]

    assert len(terms) == 1
    term = terms[0]
    assert term["name"] == "storage_inventory_dynamics"
    assert term["systems"] == ["storage_tank", "storage_tank_high_holdup"]
    assert term["state_role"] == "inventory"
    assert float(term["weight"]) == 0.5
