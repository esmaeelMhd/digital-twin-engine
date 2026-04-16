"""Tests for Phase 2 adapter conditioning and calibration utilities."""

import equinox as eqx
import jax
import jax.numpy as jnp
import yaml

from dte.calibration.unit_calibration import (
    CalibrationOptions,
    UnitCalibrator,
    initialize_target_model_from_pretrained,
)
from dte.data.datasets.universal_unit_dataset import (
    MultiSystemTrajectoryDataset,
    PreparedSystemDataset,
    SystemDatasetSource,
)
from dte.models.universal_digital_twin import UniversalDigitalTwin
from dte.simulators.registry import get_system_spec


class _DummyDataset:
    def __init__(self, n_samples: int = 8):
        self.n_samples = n_samples

    def split(self, val_fraction):
        return self, self


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_dataset(sources: list[SystemDatasetSource]) -> MultiSystemTrajectoryDataset:
    entries = []
    for source in sources:
        system_config = _load_yaml(source.system_config)
        spec = get_system_spec(system_config)
        entries.append(
            PreparedSystemDataset(
                source=source,
                spec=spec,
                dataset=_DummyDataset(),
                system_config=system_config,
            )
        )
    return MultiSystemTrajectoryDataset(entries, seq_len=5, stride=1)


def _build_config() -> dict:
    return {
        "model": {
            "latent_dim": 16,
            "shared_hidden_dim": 32,
            "system_embedding_dim": 8,
            "state_group_token_dim": 12,
            "state_group_kind_dim": 6,
            "state_group_encoder_layers": 2,
            "state_group_coupling_layers": 2,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "drift_layers": 2,
            "use_system_spec_embedding": True,
            "use_variational_encoder": True,
            "adapters": {
                "enabled": True,
                "bottleneck_dim": 8,
                "residual_scale": 0.1,
                "encoder": True,
                "drift": True,
                "decoder": True,
            },
            "neural_cde": {"enabled": True, "hidden_dim": 16, "n_layers": 2},
        },
        "training": {
            "batch_size": 2,
            "seq_len": 5,
            "stride": 1,
            "n_epochs": 1,
        },
        "optimizer": {
            "peak_lr": 1e-3,
            "end_lr": 1e-4,
            "warmup_steps": 1,
            "total_steps": 4,
            "gradient_clip": 1.0,
        },
        "loss_weights": {
            "reconstruction": 1.0,
            "trajectory": 1.0,
            "one_step": 0.5,
            "k_step": 0.0,
            "kl": 1e-4,
            "state_bounds": 0.0,
            "positivity": 0.0,
        },
        "evaluation": {
            "per_system_batches": 1,
        },
    }


def _count_trainable(model, filter_spec) -> int:
    trainable, _ = eqx.partition(model, filter_spec)
    return sum(
        leaf.size for leaf in jax.tree.leaves(eqx.filter(trainable, eqx.is_inexact_array))
    )


def test_multisystem_metadata_collects_phase2_conditioning_fields():
    dataset = _build_dataset(
        [
            SystemDatasetSource(
                name="cstr",
                system_config="configs/cstr_default.yaml",
                data_dir="data/cstr",
            ),
            SystemDatasetSource(
                name="heat_exchanger",
                system_config="configs/heat_exchanger_default.yaml",
                data_dir="data/heat_exchanger",
            ),
            SystemDatasetSource(
                name="two_tank",
                system_config="configs/two_tank_default.yaml",
                data_dir="data/two_tank",
            ),
        ]
    )

    metadata = dataset.metadata

    assert "reactor" in metadata.family_names
    assert "hydraulic" in metadata.family_names
    assert "reaction_class" in metadata.conditioning_category_names
    assert "liquid_well_mixed" in metadata.conditioning_value_names
    assert "gravity_flow" in metadata.law_tag_names
    assert metadata.parameter_law_tag_id.shape[-1] == dataset.max_param_dim


def test_initialize_target_model_from_pretrained_copies_shared_conditioning_weights():
    config = _build_config()
    source_dataset = _build_dataset(
        [
            SystemDatasetSource(
                name="cstr",
                system_config="configs/cstr_default.yaml",
                data_dir="data/cstr",
            ),
            SystemDatasetSource(
                name="heat_exchanger",
                system_config="configs/heat_exchanger_default.yaml",
                data_dir="data/heat_exchanger",
            ),
        ]
    )
    target_dataset = _build_dataset(
        [
            SystemDatasetSource(
                name="cstr_variant",
                system_config="configs/cstr_default.yaml",
                data_dir="data/cstr_variant",
            )
        ]
    )

    pretrained_model = UniversalDigitalTwin.from_config(
        config,
        source_dataset.metadata,
        jax.random.PRNGKey(0),
    )
    target_model = initialize_target_model_from_pretrained(
        pretrained_model,
        source_dataset.metadata,
        target_dataset.metadata,
        config,
        jax.random.PRNGKey(1),
    )

    reactor_idx_source = source_dataset.metadata.family_names.index("reactor")
    reactor_idx_target = target_dataset.metadata.family_names.index("reactor")

    assert jnp.allclose(
        target_model.family_embedding_table[reactor_idx_target],
        pretrained_model.family_embedding_table[reactor_idx_source],
    )


def test_unit_calibrator_uses_adapter_only_filter_and_selected_param_mask():
    config = _build_config()
    source_dataset = _build_dataset(
        [
            SystemDatasetSource(
                name="cstr",
                system_config="configs/cstr_default.yaml",
                data_dir="data/cstr",
            )
        ]
    )
    model = UniversalDigitalTwin.from_config(
        config,
        source_dataset.metadata,
        jax.random.PRNGKey(3),
    )

    options = CalibrationOptions(
        trainable_mode="adapters",
        tune_normalization=True,
        tune_physics_params=True,
        active_param_indices=(1, 3),
    )
    calibrator = UnitCalibrator(
        model,
        config,
        source_dataset,
        source_dataset,
        options=options,
        target_system_id=0,
    )

    adapter_count = calibrator.trainable_parameter_count
    full_count = _count_trainable(model, model.trainable_filter_spec(mode="full"))

    assert adapter_count > 0
    assert adapter_count < full_count
    assert jnp.allclose(
        calibrator.model.param_bias_mask_table[0],
        jnp.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
    )
