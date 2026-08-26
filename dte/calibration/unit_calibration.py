"""Few-shot calibration utilities for the universal digital twin."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax

from dte.data.datasets.universal_unit_dataset import (
    MultiSystemTrajectoryDataset,
    UniversalSystemMetadata,
)
from dte.models.universal.digital_twin import UniversalDigitalTwin
from dte.evaluation.universal import per_system_metrics_key
from dte.training.universal.trainer import UniversalTrainer


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


def _system_family_name(metadata: UniversalSystemMetadata, system_idx: int) -> str | None:
    """Return the family name for one system index when available."""

    if not metadata.family_names or metadata.family_id.size == 0:
        return None
    family_idx = int(metadata.family_id[system_idx])
    if family_idx < 0 or family_idx >= len(metadata.family_names):
        return None
    return metadata.family_names[family_idx]


def _system_subtype_name(metadata: UniversalSystemMetadata, system_idx: int) -> str | None:
    """Return the subtype name for one system index when available."""

    if not metadata.subtype_names or metadata.subtype_id.size == 0:
        return None
    subtype_idx = int(metadata.subtype_id[system_idx])
    if subtype_idx < 0 or subtype_idx >= len(metadata.subtype_names):
        return None
    return metadata.subtype_names[subtype_idx]


def _name_similarity_score(source_name: str, target_name: str) -> int:
    """Score symbolic similarity between source and target system names."""

    source_tokens = source_name.split("_")
    target_tokens = target_name.split("_")
    prefix = 0
    while (
        prefix < len(source_tokens)
        and prefix < len(target_tokens)
        and source_tokens[prefix] == target_tokens[prefix]
    ):
        prefix += 1
    overlap = len(set(source_tokens) & set(target_tokens))
    return prefix * 10 + overlap


def _closest_source_system_index(
    source_metadata: UniversalSystemMetadata,
    target_metadata: UniversalSystemMetadata,
    target_idx: int,
) -> int | None:
    """Select the closest source system when the target name is new."""

    target_name = target_metadata.system_names[target_idx]
    source_name_to_idx = {name: idx for idx, name in enumerate(source_metadata.system_names)}
    exact_idx = source_name_to_idx.get(target_name)
    if exact_idx is not None:
        return exact_idx

    target_family = _system_family_name(target_metadata, target_idx)
    target_subtype = _system_subtype_name(target_metadata, target_idx)
    best_idx: int | None = None
    best_score = -1

    for source_idx, source_name in enumerate(source_metadata.system_names):
        score = _name_similarity_score(source_name, target_name)
        if target_family is not None and _system_family_name(source_metadata, source_idx) == target_family:
            score += 1000
        if target_subtype is not None and _system_subtype_name(source_metadata, source_idx) == target_subtype:
            score += 100
        if score > best_score:
            best_score = score
            best_idx = source_idx

    return best_idx if best_score > 0 else None


def _copy_system_rows_with_fallback(
    target_table,
    source_table,
    source_metadata: UniversalSystemMetadata,
    target_metadata: UniversalSystemMetadata,
):
    """Copy exact-name rows, then fall back to the closest source system row."""

    if source_table.shape[1:] != target_table.shape[1:]:
        return target_table

    result = target_table
    source_index = {name: idx for idx, name in enumerate(source_metadata.system_names)}
    for target_idx, target_name in enumerate(target_metadata.system_names):
        source_idx = source_index.get(target_name)
        if source_idx is None:
            source_idx = _closest_source_system_index(source_metadata, target_metadata, target_idx)
        if source_idx is not None:
            result = result.at[target_idx].set(source_table[source_idx])
    return result


def _linear_shapes_match(source_layer, target_layer) -> bool:
    """Return True when two Equinox linear layers are shape-compatible."""

    if source_layer is None or target_layer is None:
        return False
    return (
        getattr(source_layer, "weight", None) is not None
        and getattr(target_layer, "weight", None) is not None
        and source_layer.weight.shape == target_layer.weight.shape
        and (
            (source_layer.bias is None and target_layer.bias is None)
            or (
                source_layer.bias is not None
                and target_layer.bias is not None
                and source_layer.bias.shape == target_layer.bias.shape
            )
        )
    )


def _adapter_shapes_match(source_adapter, target_adapter) -> bool:
    """Return True when two bottleneck adapters are shape-compatible."""

    if source_adapter is None or target_adapter is None:
        return False
    return _linear_shapes_match(source_adapter.down, target_adapter.down) and _linear_shapes_match(
        source_adapter.up, target_adapter.up
    )


def _residual_mlp_shapes_match(source_mlp, target_mlp) -> bool:
    """Return True when two ResidualMLP modules are layer-shape compatible."""

    if source_mlp is None or target_mlp is None:
        return False
    if len(source_mlp.layers) != len(target_mlp.layers):
        return False
    return all(
        _linear_shapes_match(source_layer, target_layer)
        for source_layer, target_layer in zip(source_mlp.layers, target_mlp.layers)
    ) and _linear_shapes_match(source_mlp.output_layer, target_mlp.output_layer)


def initialize_target_model_from_pretrained(
    pretrained_model: UniversalDigitalTwin,
    source_metadata: UniversalSystemMetadata,
    target_metadata: UniversalSystemMetadata,
    config: dict,
    key,
    *,
    copy_system_embedding_rows: bool = True,
    copy_calibration_rows: bool = True,
    copy_param_bias_rows: bool = True,
    copy_dynamics_backbone: bool = True,
    copy_drift_backbone: bool | None = None,
    copy_cde_backbone: bool | None = None,
    copy_drift_adapter: bool = True,
) -> UniversalDigitalTwin:
    """Initialize a target-only calibration model from a pretrained checkpoint.

    Shared backbone weights are copied exactly. Named embedding rows are copied
    wherever the target metadata shares the same symbolic names; new target-only
    symbols keep their fresh initialization.
    """

    model = UniversalDigitalTwin.from_config(config, target_metadata, key)

    if _residual_mlp_shapes_match(pretrained_model.state_group_encoder, model.state_group_encoder):
        model = eqx.tree_at(
            lambda m: m.state_group_encoder,
            model,
            pretrained_model.state_group_encoder,
        )
    if _residual_mlp_shapes_match(pretrained_model.state_group_mixer, model.state_group_mixer):
        model = eqx.tree_at(
            lambda m: m.state_group_mixer,
            model,
            pretrained_model.state_group_mixer,
        )
    if _residual_mlp_shapes_match(pretrained_model.encoder_mean, model.encoder_mean):
        model = eqx.tree_at(
            lambda m: m.encoder_mean,
            model,
            pretrained_model.encoder_mean,
        )
    if _residual_mlp_shapes_match(pretrained_model.encoder_logvar, model.encoder_logvar):
        model = eqx.tree_at(
            lambda m: m.encoder_logvar,
            model,
            pretrained_model.encoder_logvar,
        )
    if _residual_mlp_shapes_match(pretrained_model.decoder, model.decoder):
        model = eqx.tree_at(
            lambda m: m.decoder,
            model,
            pretrained_model.decoder,
        )
    use_drift_backbone = copy_dynamics_backbone if copy_drift_backbone is None else copy_drift_backbone
    use_cde_backbone = copy_dynamics_backbone if copy_cde_backbone is None else copy_cde_backbone

    if use_drift_backbone and _residual_mlp_shapes_match(pretrained_model.drift, model.drift):
        model = eqx.tree_at(
            lambda m: m.drift,
            model,
            pretrained_model.drift,
        )

    if use_cde_backbone and _residual_mlp_shapes_match(pretrained_model.cde_matrix, model.cde_matrix):
        model = eqx.tree_at(lambda m: m.cde_matrix, model, pretrained_model.cde_matrix)
    if _linear_shapes_match(pretrained_model.descriptor_proj, model.descriptor_proj):
        model = eqx.tree_at(lambda m: m.descriptor_proj, model, pretrained_model.descriptor_proj)
    if _linear_shapes_match(pretrained_model.group_token_film_scale, model.group_token_film_scale):
        model = eqx.tree_at(
            lambda m: m.group_token_film_scale,
            model,
            pretrained_model.group_token_film_scale,
        )
    if _linear_shapes_match(pretrained_model.group_token_film_shift, model.group_token_film_shift):
        model = eqx.tree_at(
            lambda m: m.group_token_film_shift,
            model,
            pretrained_model.group_token_film_shift,
        )
    if _linear_shapes_match(pretrained_model.group_mixer_film_scale, model.group_mixer_film_scale):
        model = eqx.tree_at(
            lambda m: m.group_mixer_film_scale,
            model,
            pretrained_model.group_mixer_film_scale,
        )
    if _linear_shapes_match(pretrained_model.group_mixer_film_shift, model.group_mixer_film_shift):
        model = eqx.tree_at(
            lambda m: m.group_mixer_film_shift,
            model,
            pretrained_model.group_mixer_film_shift,
        )
    if _linear_shapes_match(pretrained_model.decoder_film_scale, model.decoder_film_scale):
        model = eqx.tree_at(
            lambda m: m.decoder_film_scale,
            model,
            pretrained_model.decoder_film_scale,
        )
    if _linear_shapes_match(pretrained_model.decoder_film_shift, model.decoder_film_shift):
        model = eqx.tree_at(
            lambda m: m.decoder_film_shift,
            model,
            pretrained_model.decoder_film_shift,
        )
    if _adapter_shapes_match(pretrained_model.encoder_adapter, model.encoder_adapter):
        model = eqx.tree_at(lambda m: m.encoder_adapter, model, pretrained_model.encoder_adapter)
    if copy_drift_adapter and _adapter_shapes_match(pretrained_model.drift_adapter, model.drift_adapter):
        model = eqx.tree_at(lambda m: m.drift_adapter, model, pretrained_model.drift_adapter)
    if _adapter_shapes_match(pretrained_model.decoder_adapter, model.decoder_adapter):
        model = eqx.tree_at(lambda m: m.decoder_adapter, model, pretrained_model.decoder_adapter)

    if copy_system_embedding_rows:
        model = eqx.tree_at(
            lambda m: m.system_embedding_table,
            model,
            _copy_system_rows_with_fallback(
                model.system_embedding_table,
                pretrained_model.system_embedding_table,
                source_metadata,
                target_metadata,
            ),
        )
    if copy_calibration_rows:
        model = eqx.tree_at(
            lambda m: (
                m.state_center_delta_table,
                m.state_scale_log_delta_table,
                m.control_center_delta_table,
                m.control_scale_log_delta_table,
                m.disturbance_center_delta_table,
                m.disturbance_scale_log_delta_table,
            ),
            model,
            (
                _copy_system_rows_with_fallback(
                    model.state_center_delta_table,
                    pretrained_model.state_center_delta_table,
                    source_metadata,
                    target_metadata,
                ),
                _copy_system_rows_with_fallback(
                    model.state_scale_log_delta_table,
                    pretrained_model.state_scale_log_delta_table,
                    source_metadata,
                    target_metadata,
                ),
                _copy_system_rows_with_fallback(
                    model.control_center_delta_table,
                    pretrained_model.control_center_delta_table,
                    source_metadata,
                    target_metadata,
                ),
                _copy_system_rows_with_fallback(
                    model.control_scale_log_delta_table,
                    pretrained_model.control_scale_log_delta_table,
                    source_metadata,
                    target_metadata,
                ),
                _copy_system_rows_with_fallback(
                    model.disturbance_center_delta_table,
                    pretrained_model.disturbance_center_delta_table,
                    source_metadata,
                    target_metadata,
                ),
                _copy_system_rows_with_fallback(
                    model.disturbance_scale_log_delta_table,
                    pretrained_model.disturbance_scale_log_delta_table,
                    source_metadata,
                    target_metadata,
                ),
            ),
        )
    if copy_param_bias_rows:
        model = eqx.tree_at(
            lambda m: m.param_bias_table,
            model,
            _copy_system_rows_with_fallback(
                model.param_bias_table,
                pretrained_model.param_bias_table,
                source_metadata,
                target_metadata,
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
            per_system_metrics_key(self.trainer): per_system,
        }
