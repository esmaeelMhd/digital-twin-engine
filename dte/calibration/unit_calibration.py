"""Few-shot calibration utilities for the universal digital twin."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax

from dte.data.multi_system_dataset import MultiSystemTrajectoryDataset, UniversalSystemMetadata
from dte.models.universal_digital_twin import UniversalDigitalTwin
from dte.training.universal_trainer import UniversalTrainer


def _trainable_parameter_count(model, filter_spec) -> int:
    trainable, _ = eqx.partition(model, filter_spec)
    return sum(
        leaf.size for leaf in jax.tree.leaves(eqx.filter(trainable, eqx.is_inexact_array))
    )


def _copy_rows_by_name(
    target_table,
    source_table,
    source_names: tuple[str, ...],
    target_names: tuple[str, ...],
):
    """Copy embedding rows for shared symbolic names."""

    result = target_table
    source_index = {name: idx for idx, name in enumerate(source_names)}
    for target_idx, name in enumerate(target_names):
        source_idx = source_index.get(name)
        if source_idx is not None and source_table.shape[1:] == target_table.shape[1:]:
            result = result.at[target_idx].set(source_table[source_idx])
    return result


def initialize_target_model_from_pretrained(
    pretrained_model: UniversalDigitalTwin,
    source_metadata: UniversalSystemMetadata,
    target_metadata: UniversalSystemMetadata,
    config: dict,
    key,
) -> UniversalDigitalTwin:
    """Initialize a target-only calibration model from a pretrained checkpoint.

    Shared backbone weights are copied exactly. Named embedding rows are copied
    wherever the target metadata shares the same symbolic names; new target-only
    symbols keep their fresh initialization.
    """

    model = UniversalDigitalTwin.from_config(config, target_metadata, key)

    model = eqx.tree_at(
        lambda m: (
            m.state_group_encoder,
            m.state_group_mixer,
            m.group_token_film_scale,
            m.group_token_film_shift,
            m.group_mixer_film_scale,
            m.group_mixer_film_shift,
            m.decoder_film_scale,
            m.decoder_film_shift,
            m.encoder_mean,
            m.encoder_logvar,
            m.decoder,
            m.drift,
        ),
        model,
        replace=(
            pretrained_model.state_group_encoder,
            pretrained_model.state_group_mixer,
            pretrained_model.group_token_film_scale,
            pretrained_model.group_token_film_shift,
            pretrained_model.group_mixer_film_scale,
            pretrained_model.group_mixer_film_shift,
            pretrained_model.decoder_film_scale,
            pretrained_model.decoder_film_shift,
            pretrained_model.encoder_mean,
            pretrained_model.encoder_logvar,
            pretrained_model.decoder,
            pretrained_model.drift,
        ),
    )

    if pretrained_model.cde_matrix is not None and model.cde_matrix is not None:
        model = eqx.tree_at(lambda m: m.cde_matrix, model, pretrained_model.cde_matrix)
    if pretrained_model.descriptor_proj is not None and model.descriptor_proj is not None:
        model = eqx.tree_at(lambda m: m.descriptor_proj, model, pretrained_model.descriptor_proj)
    if pretrained_model.encoder_adapter is not None and model.encoder_adapter is not None:
        model = eqx.tree_at(lambda m: m.encoder_adapter, model, pretrained_model.encoder_adapter)
    if pretrained_model.drift_adapter is not None and model.drift_adapter is not None:
        model = eqx.tree_at(lambda m: m.drift_adapter, model, pretrained_model.drift_adapter)
    if pretrained_model.decoder_adapter is not None and model.decoder_adapter is not None:
        model = eqx.tree_at(lambda m: m.decoder_adapter, model, pretrained_model.decoder_adapter)

    model = eqx.tree_at(
        lambda m: m.system_embedding_table,
        model,
        _copy_rows_by_name(
            model.system_embedding_table,
            pretrained_model.system_embedding_table,
            source_metadata.system_names,
            target_metadata.system_names,
        ),
    )
    model = eqx.tree_at(
        lambda m: m.family_embedding_table,
        model,
        _copy_rows_by_name(
            model.family_embedding_table,
            pretrained_model.family_embedding_table,
            source_metadata.family_names,
            target_metadata.family_names,
        ),
    )
    model = eqx.tree_at(
        lambda m: m.subtype_embedding_table,
        model,
        _copy_rows_by_name(
            model.subtype_embedding_table,
            pretrained_model.subtype_embedding_table,
            source_metadata.subtype_names,
            target_metadata.subtype_names,
        ),
    )
    model = eqx.tree_at(
        lambda m: m.law_embedding_table,
        model,
        _copy_rows_by_name(
            model.law_embedding_table,
            pretrained_model.law_embedding_table,
            source_metadata.law_tag_names,
            target_metadata.law_tag_names,
        ),
    )
    model = eqx.tree_at(
        lambda m: m.conditioning_category_embedding_table,
        model,
        _copy_rows_by_name(
            model.conditioning_category_embedding_table,
            pretrained_model.conditioning_category_embedding_table,
            source_metadata.conditioning_category_names,
            target_metadata.conditioning_category_names,
        ),
    )
    model = eqx.tree_at(
        lambda m: m.conditioning_value_embedding_table,
        model,
        _copy_rows_by_name(
            model.conditioning_value_embedding_table,
            pretrained_model.conditioning_value_embedding_table,
            source_metadata.conditioning_value_names,
            target_metadata.conditioning_value_names,
        ),
    )
    model = eqx.tree_at(
        lambda m: m.state_group_kind_embedding_table,
        model,
        _copy_rows_by_name(
            model.state_group_kind_embedding_table,
            pretrained_model.state_group_kind_embedding_table,
            source_metadata.state_group_kind_names,
            target_metadata.state_group_kind_names,
        ),
    )
    return model


@dataclass(frozen=True)
class CalibrationOptions:
    """Configuration for adapter-based customer calibration."""

    trainable_mode: Literal["adapters", "full"] = "adapters"
    tune_normalization: bool = True
    tune_physics_params: bool = False
    active_param_indices: tuple[int, ...] | None = None


class UnitCalibrator:
    """Run few-shot calibration on a target-only universal dataset."""

    def __init__(
        self,
        model: UniversalDigitalTwin,
        config: dict,
        train_dataset: MultiSystemTrajectoryDataset,
        val_dataset: MultiSystemTrajectoryDataset | None = None,
        *,
        options: CalibrationOptions | None = None,
        target_system_id: int = 0,
    ):
        self.model = model
        self.config = copy.deepcopy(config)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.options = options or CalibrationOptions()
        self.target_system_id = int(target_system_id)
        self.target_system_name = train_dataset.system_names[self.target_system_id]

        if self.options.tune_physics_params:
            self.model = self.model.configure_param_calibration_mask(
                self.target_system_id,
                list(self.options.active_param_indices)
                if self.options.active_param_indices is not None
                else None,
            )
        else:
            self.model = self.model.configure_param_calibration_mask(
                self.target_system_id,
                [],
            )

        self.filter_spec = self._build_filter_spec()
        self.trainer = UniversalTrainer(
            self.model,
            self.config,
            self.train_dataset,
            self.val_dataset,
            filter_spec=self.filter_spec,
        )

    def _build_filter_spec(self):
        spec = self.model.trainable_filter_spec(
            mode=self.options.trainable_mode,
            include_calibration=(
                self.options.tune_normalization or self.options.tune_physics_params
            ),
        )
        if not self.options.tune_normalization:
            spec = eqx.tree_at(
                lambda m: (
                    m.state_center_delta_table,
                    m.state_scale_log_delta_table,
                    m.control_center_delta_table,
                    m.control_scale_log_delta_table,
                    m.disturbance_center_delta_table,
                    m.disturbance_scale_log_delta_table,
                ),
                spec,
                replace=(False,) * 6,
            )
        if not self.options.tune_physics_params:
            spec = eqx.tree_at(
                lambda m: m.param_bias_table,
                spec,
                replace=False,
            )
        return spec

    @property
    def trainable_parameter_count(self) -> int:
        return _trainable_parameter_count(self.model, self.filter_spec)

    def calibrate(
        self,
        output_dir: str,
        *,
        key,
        time_budget_seconds: float | None = None,
    ) -> dict:
        """Run calibration and return a machine-readable summary."""

        n_epochs = int(self.config["training"]["n_epochs"])
        train_summary = self.trainer.train(
            n_epochs=n_epochs,
            output_dir=output_dir,
            key=key,
            time_budget_seconds=time_budget_seconds,
        )
        self.model = self.trainer.model
        eval_batches = int(self.config.get("evaluation", {}).get("per_system_batches", 4))
        key, eval_key = jax.random.split(key)
        per_system = self.trainer.evaluate_per_system(eval_key, n_batches=eval_batches)
        return {
            **train_summary,
            "target_system": self.target_system_name,
            "target_system_id": self.target_system_id,
            "trainable_mode": self.options.trainable_mode,
            "tune_normalization": self.options.tune_normalization,
            "tune_physics_params": self.options.tune_physics_params,
            "active_param_indices": list(self.options.active_param_indices or ()),
            "trainable_parameter_count": self.trainable_parameter_count,
            "parameter_counts": self.model.get_parameter_count(),
            "per_system_val_losses": per_system,
        }
