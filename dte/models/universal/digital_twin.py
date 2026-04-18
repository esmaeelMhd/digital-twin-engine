"""Shared universal digital twin for mixed-system training.

The universal path intentionally starts simpler than the single-system
`DigitalTwin`: it operates in padded normalized space, uses explicit masks, and
conditions on structured unit metadata. Phase 2 extends that conditioning with
family/law/tag embeddings plus lightweight residual adapters for low-parameter
customer calibration.
"""

from __future__ import annotations

from typing import Dict, Literal

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from dte.data.datasets.universal_unit_dataset import UniversalSystemMetadata


def _count_params(module) -> int:
    return sum(x.size for x in jax.tree.leaves(eqx.filter(module, eqx.is_inexact_array)))


def _masked_embedding_mean(embeddings: Array, weights: Array) -> Array:
    weights = weights.astype(embeddings.dtype)
    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=embeddings.dtype))
    return jnp.sum(embeddings * weights[:, None], axis=0) / denom


class ResidualMLP(eqx.Module):
    """Small residual MLP used by the universal encoder/decoder/drift."""

    layers: list
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, n_layers + 1)
        self.layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[i]))
        final_in_dim = hidden_dim if n_layers > 0 else input_dim
        self.output_layer = eqx.nn.Linear(final_in_dim, output_dim, key=keys[-1])

    def __call__(self, x: Array) -> Array:
        h = x
        for i, layer in enumerate(self.layers):
            out = jax.nn.gelu(layer(h if i > 0 else x))
            h = h + out if i > 0 and h.shape == out.shape else out
        base = h if self.layers else x
        return self.output_layer(base)


class BottleneckAdapter(eqx.Module):
    """Low-parameter residual adapter used during customer calibration."""

    down: eqx.nn.Linear
    up: eqx.nn.Linear
    residual_scale: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        bottleneck_dim: int,
        *,
        residual_scale: float,
        key: PRNGKeyArray,
    ):
        key_down, key_up = jax.random.split(key)
        self.down = eqx.nn.Linear(input_dim + context_dim, bottleneck_dim, key=key_down)
        self.up = eqx.nn.Linear(bottleneck_dim, input_dim, key=key_up)
        self.residual_scale = residual_scale

    def __call__(self, x: Array, context: Array) -> Array:
        h = jax.nn.gelu(self.down(jnp.concatenate([x, context], axis=-1)))
        return x + self.residual_scale * self.up(h)


class UniversalDigitalTwin(eqx.Module):
    """Shared padded digital twin trained across multiple systems."""

    state_group_encoder: ResidualMLP
    state_group_mixer: ResidualMLP
    group_token_film_scale: eqx.nn.Linear
    group_token_film_shift: eqx.nn.Linear
    group_mixer_film_scale: eqx.nn.Linear
    group_mixer_film_shift: eqx.nn.Linear
    decoder_film_scale: eqx.nn.Linear
    decoder_film_shift: eqx.nn.Linear
    encoder_mean: ResidualMLP
    encoder_logvar: ResidualMLP
    decoder: ResidualMLP
    drift: ResidualMLP
    cde_matrix: ResidualMLP | None
    descriptor_proj: eqx.nn.Linear | None
    channel_context_proj: eqx.nn.Linear | None
    law_feature_proj: eqx.nn.Linear | None
    encoder_adapter: BottleneckAdapter | None
    drift_adapter: BottleneckAdapter | None
    decoder_adapter: BottleneckAdapter | None

    system_embedding_table: Float[Array, "n_systems embedding_dim"]
    family_embedding_table: Float[Array, "n_families embedding_dim"]
    subtype_embedding_table: Float[Array, "n_subtypes embedding_dim"]
    law_embedding_table: Float[Array, "n_law_tags embedding_dim"]
    conditioning_category_embedding_table: Float[Array, "n_conditioning_categories embedding_dim"]
    conditioning_value_embedding_table: Float[Array, "n_conditioning_values embedding_dim"]
    state_role_embedding_table: Float[Array, "n_state_roles embedding_dim"]
    control_role_embedding_table: Float[Array, "n_control_roles embedding_dim"]
    disturbance_role_embedding_table: Float[Array, "n_disturbance_roles embedding_dim"]
    channel_name_embedding_table: Float[Array, "n_channel_names embedding_dim"]

    state_center_table: Float[Array, "n_systems max_state_dim"]
    state_scale_table: Float[Array, "n_systems max_state_dim"]
    control_center_table: Float[Array, "n_systems max_control_dim"]
    control_scale_table: Float[Array, "n_systems max_control_dim"]
    disturbance_center_table: Float[Array, "n_systems max_disturbance_dim"]
    disturbance_scale_table: Float[Array, "n_systems max_disturbance_dim"]
    param_scale_table: Float[Array, "n_systems max_param_dim"]
    state_mask_table: Float[Array, "n_systems max_state_dim"]
    control_mask_table: Float[Array, "n_systems max_control_dim"]
    disturbance_mask_table: Float[Array, "n_systems max_disturbance_dim"]
    param_mask_table: Float[Array, "n_systems max_param_dim"]
    descriptor_table: Float[Array, "n_systems descriptor_dim"]
    state_group_mask_table: Float[Array, "n_systems max_state_groups max_state_dim"]
    state_group_active_table: Float[Array, "n_systems max_state_groups"]
    state_group_kind_id_table: Array
    state_group_kind_embedding_table: Float[Array, "n_group_kinds group_kind_dim"]
    family_id_table: Array
    subtype_id_table: Array
    law_tag_mask_table: Float[Array, "n_systems n_law_tags"]
    conditioning_value_id_table: Array
    parameter_law_tag_id_table: Array
    state_role_id_table: Array
    control_role_id_table: Array
    disturbance_role_id_table: Array
    state_name_id_table: Array
    control_name_id_table: Array
    disturbance_name_id_table: Array
    law_feature_table: Float[Array, "n_systems n_law_features"]

    state_center_delta_table: Float[Array, "n_systems max_state_dim"]
    state_scale_log_delta_table: Float[Array, "n_systems max_state_dim"]
    control_center_delta_table: Float[Array, "n_systems max_control_dim"]
    control_scale_log_delta_table: Float[Array, "n_systems max_control_dim"]
    disturbance_center_delta_table: Float[Array, "n_systems max_disturbance_dim"]
    disturbance_scale_log_delta_table: Float[Array, "n_systems max_disturbance_dim"]
    param_bias_table: Float[Array, "n_systems max_param_dim"]
    param_bias_mask_table: Float[Array, "n_systems max_param_dim"]

    latent_dim: int = eqx.field(static=True)
    max_state_dim: int = eqx.field(static=True)
    max_control_dim: int = eqx.field(static=True)
    max_disturbance_dim: int = eqx.field(static=True)
    max_param_dim: int = eqx.field(static=True)
    max_state_groups: int = eqx.field(static=True)
    system_embedding_dim: int = eqx.field(static=True)
    descriptor_dim: int = eqx.field(static=True)
    group_kind_dim: int = eqx.field(static=True)
    state_group_token_dim: int = eqx.field(static=True)
    path_dim: int = eqx.field(static=True)
    use_system_spec_embedding: bool = eqx.field(static=True)
    neural_cde_enabled: bool = eqx.field(static=True)
    latent_solver_method: str = eqx.field(static=True)
    latent_solver_rtol: float = eqx.field(static=True)
    latent_solver_atol: float = eqx.field(static=True)
    latent_solver_dt0_factor: float = eqx.field(static=True)
    latent_solver_max_steps: int = eqx.field(static=True)
    use_variational_encoder: bool = eqx.field(static=True)
    adapters_enabled: bool = eqx.field(static=True)
    adapter_bottleneck_dim: int = eqx.field(static=True)
    channel_conditioning_enabled: bool = eqx.field(static=True)
    law_conditioning_enabled: bool = eqx.field(static=True)
    system_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        metadata: UniversalSystemMetadata,
        *,
        latent_dim: int,
        shared_hidden_dim: int,
        system_embedding_dim: int,
        state_group_token_dim: int,
        group_kind_dim: int,
        state_group_encoder_layers: int,
        state_group_coupling_layers: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        n_drift_layers: int,
        use_system_spec_embedding: bool,
        neural_cde_enabled: bool,
        neural_cde_hidden_dim: int,
        neural_cde_layers: int,
        latent_solver_method: str,
        latent_solver_rtol: float,
        latent_solver_atol: float,
        latent_solver_dt0_factor: float,
        latent_solver_max_steps: int,
        use_variational_encoder: bool,
        channel_conditioning_enabled: bool,
        law_conditioning_enabled: bool,
        adapters_enabled: bool,
        adapter_bottleneck_dim: int,
        adapter_residual_scale: float,
        enable_encoder_adapter: bool,
        enable_drift_adapter: bool,
        enable_decoder_adapter: bool,
        key: PRNGKeyArray,
    ):
        self.latent_dim = latent_dim
        self.max_state_dim = int(metadata.state_center.shape[1])
        self.max_control_dim = int(metadata.control_center.shape[1])
        self.max_disturbance_dim = int(metadata.disturbance_center.shape[1])
        self.max_param_dim = int(metadata.param_scale.shape[1])
        self.max_state_groups = int(metadata.state_group_mask.shape[1])
        self.system_embedding_dim = system_embedding_dim
        self.descriptor_dim = int(metadata.system_descriptor.shape[1])
        self.group_kind_dim = group_kind_dim
        self.state_group_token_dim = state_group_token_dim
        self.path_dim = 1 + self.max_control_dim + self.max_disturbance_dim
        self.use_system_spec_embedding = use_system_spec_embedding
        self.neural_cde_enabled = neural_cde_enabled
        self.latent_solver_method = latent_solver_method
        self.latent_solver_rtol = latent_solver_rtol
        self.latent_solver_atol = latent_solver_atol
        self.latent_solver_dt0_factor = latent_solver_dt0_factor
        self.latent_solver_max_steps = latent_solver_max_steps
        self.use_variational_encoder = use_variational_encoder
        self.channel_conditioning_enabled = channel_conditioning_enabled
        self.law_conditioning_enabled = law_conditioning_enabled
        self.adapters_enabled = adapters_enabled
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.system_names = tuple(metadata.system_names)

        self.state_center_table = metadata.state_center
        self.state_scale_table = metadata.state_scale
        self.control_center_table = metadata.control_center
        self.control_scale_table = metadata.control_scale
        self.disturbance_center_table = metadata.disturbance_center
        self.disturbance_scale_table = metadata.disturbance_scale
        self.param_scale_table = metadata.param_scale
        self.state_mask_table = metadata.state_mask
        self.control_mask_table = metadata.control_mask
        self.disturbance_mask_table = metadata.disturbance_mask
        self.param_mask_table = metadata.param_mask
        self.descriptor_table = metadata.system_descriptor
        self.state_group_mask_table = metadata.state_group_mask
        self.state_group_active_table = metadata.state_group_active
        self.state_group_kind_id_table = metadata.state_group_kind_id

        n_systems = len(metadata.system_names)
        n_group_kinds = max(len(metadata.state_group_kind_names), 1)
        n_families = max(len(metadata.family_names), 1)
        n_subtypes = max(len(metadata.subtype_names), 1)
        n_law_tags = max(len(metadata.law_tag_names), 1)
        n_conditioning_categories = max(len(metadata.conditioning_category_names), 1)
        n_conditioning_values = max(len(metadata.conditioning_value_names), 1)

        self.family_id_table = (
            metadata.family_id
            if int(getattr(metadata.family_id, "size", 0)) > 0
            else jnp.zeros((n_systems,), dtype=jnp.int32)
        )
        self.subtype_id_table = (
            metadata.subtype_id
            if int(getattr(metadata.subtype_id, "size", 0)) > 0
            else jnp.zeros((n_systems,), dtype=jnp.int32)
        )
        self.law_tag_mask_table = (
            metadata.law_tag_mask
            if int(getattr(metadata.law_tag_mask, "size", 0)) > 0
            else jnp.ones((n_systems, 1), dtype=jnp.float32)
        )
        self.conditioning_value_id_table = (
            metadata.conditioning_value_id
            if int(getattr(metadata.conditioning_value_id, "size", 0)) > 0
            else jnp.zeros((n_systems, n_conditioning_categories), dtype=jnp.int32)
        )
        self.parameter_law_tag_id_table = (
            metadata.parameter_law_tag_id
            if int(getattr(metadata.parameter_law_tag_id, "size", 0)) > 0
            else jnp.zeros((n_systems, self.max_param_dim), dtype=jnp.int32)
        )
        self.state_role_id_table = (
            metadata.state_role_id
            if int(getattr(metadata.state_role_id, "size", 0)) > 0
            else jnp.zeros((n_systems, self.max_state_dim), dtype=jnp.int32)
        )
        self.control_role_id_table = (
            metadata.control_role_id
            if int(getattr(metadata.control_role_id, "size", 0)) > 0
            else jnp.zeros((n_systems, self.max_control_dim), dtype=jnp.int32)
        )
        self.disturbance_role_id_table = (
            metadata.disturbance_role_id
            if int(getattr(metadata.disturbance_role_id, "size", 0)) > 0
            else jnp.zeros((n_systems, self.max_disturbance_dim), dtype=jnp.int32)
        )
        self.state_name_id_table = (
            metadata.state_name_id
            if int(getattr(metadata.state_name_id, "size", 0)) > 0
            else jnp.zeros((n_systems, self.max_state_dim), dtype=jnp.int32)
        )
        self.control_name_id_table = (
            metadata.control_name_id
            if int(getattr(metadata.control_name_id, "size", 0)) > 0
            else jnp.zeros((n_systems, self.max_control_dim), dtype=jnp.int32)
        )
        self.disturbance_name_id_table = (
            metadata.disturbance_name_id
            if int(getattr(metadata.disturbance_name_id, "size", 0)) > 0
            else jnp.zeros((n_systems, self.max_disturbance_dim), dtype=jnp.int32)
        )
        self.law_feature_table = (
            metadata.law_feature_defaults
            if int(getattr(metadata.law_feature_defaults, "size", 0)) > 0
            else jnp.zeros((n_systems, 0), dtype=jnp.float32)
        )

        self.state_center_delta_table = jnp.zeros_like(self.state_center_table)
        self.state_scale_log_delta_table = jnp.zeros_like(self.state_scale_table)
        self.control_center_delta_table = jnp.zeros_like(self.control_center_table)
        self.control_scale_log_delta_table = jnp.zeros_like(self.control_scale_table)
        self.disturbance_center_delta_table = jnp.zeros_like(self.disturbance_center_table)
        self.disturbance_scale_log_delta_table = jnp.zeros_like(self.disturbance_scale_table)
        self.param_bias_table = jnp.zeros_like(self.param_scale_table)
        self.param_bias_mask_table = self.param_mask_table.astype(jnp.float32)

        (
            key_embed,
            key_desc,
            key_family,
            key_subtype,
            key_law,
            key_conditioning_category,
            key_conditioning_value,
            key_state_role,
            key_control_role,
            key_disturbance_role,
            key_channel_name,
            key_group_kind,
            key_group_enc,
            key_group_mix,
            key_group_token_film_scale,
            key_group_token_film_shift,
            key_group_mixer_film_scale,
            key_group_mixer_film_shift,
            key_decoder_film_scale,
            key_decoder_film_shift,
            key_channel_context_proj,
            key_law_feature_proj,
            key_enc_mean,
            key_enc_logvar,
            key_dec,
            key_drift,
            key_cde,
            key_encoder_adapter,
            key_drift_adapter,
            key_decoder_adapter,
        ) = jax.random.split(key, 30)

        self.system_embedding_table = (
            0.02 * jax.random.normal(key_embed, (n_systems, system_embedding_dim))
        )
        self.family_embedding_table = (
            0.02 * jax.random.normal(key_family, (n_families, system_embedding_dim))
        )
        self.subtype_embedding_table = (
            0.02 * jax.random.normal(key_subtype, (n_subtypes, system_embedding_dim))
        )
        self.law_embedding_table = (
            0.02 * jax.random.normal(key_law, (n_law_tags, system_embedding_dim))
        )
        self.conditioning_category_embedding_table = (
            0.02
            * jax.random.normal(
                key_conditioning_category,
                (n_conditioning_categories, system_embedding_dim),
            )
        )
        self.conditioning_value_embedding_table = (
            0.02
            * jax.random.normal(
                key_conditioning_value,
                (n_conditioning_values, system_embedding_dim),
            )
        )
        self.state_role_embedding_table = 0.02 * jax.random.normal(
            key_state_role,
            (max(len(metadata.state_role_names), 1), system_embedding_dim),
        )
        self.control_role_embedding_table = 0.02 * jax.random.normal(
            key_control_role,
            (max(len(metadata.control_role_names), 1), system_embedding_dim),
        )
        self.disturbance_role_embedding_table = 0.02 * jax.random.normal(
            key_disturbance_role,
            (max(len(metadata.disturbance_role_names), 1), system_embedding_dim),
        )
        self.channel_name_embedding_table = 0.02 * jax.random.normal(
            key_channel_name,
            (max(len(metadata.channel_name_names), 1), system_embedding_dim),
        )
        self.state_group_kind_embedding_table = (
            0.02 * jax.random.normal(key_group_kind, (n_group_kinds, group_kind_dim))
        )

        if self.use_system_spec_embedding:
            self.descriptor_proj = eqx.nn.Linear(
                self.descriptor_dim,
                system_embedding_dim,
                key=key_desc,
            )
        else:
            self.descriptor_proj = None

        if self.channel_conditioning_enabled:
            self.channel_context_proj = eqx.nn.Linear(
                system_embedding_dim,
                system_embedding_dim,
                key=key_channel_context_proj,
            )
        else:
            self.channel_context_proj = None
        if self.law_conditioning_enabled and int(self.law_feature_table.shape[1]) > 0:
            self.law_feature_proj = eqx.nn.Linear(
                int(self.law_feature_table.shape[1]),
                system_embedding_dim,
                key=key_law_feature_proj,
            )
        else:
            self.law_feature_proj = None

        context_dim = system_embedding_dim
        group_encoder_input_dim = 2 * self.max_state_dim + context_dim + group_kind_dim
        group_coupling_input_dim = 2 * state_group_token_dim + context_dim + group_kind_dim
        encoder_input_dim = (
            state_group_token_dim
            + 2 * self.max_control_dim
            + 2 * self.max_param_dim
            + context_dim
        )
        decoder_input_dim = (
            latent_dim
            + 2 * self.max_control_dim
            + 2 * self.max_param_dim
            + self.max_state_dim
            + group_kind_dim
            + context_dim
        )
        drift_input_dim = (
            latent_dim
            + 2 * self.max_control_dim
            + 2 * self.max_disturbance_dim
            + 2 * self.max_param_dim
            + context_dim
        )

        self.state_group_encoder = ResidualMLP(
            group_encoder_input_dim,
            shared_hidden_dim,
            state_group_token_dim,
            state_group_encoder_layers,
            key=key_group_enc,
        )
        self.state_group_mixer = ResidualMLP(
            group_coupling_input_dim,
            shared_hidden_dim,
            state_group_token_dim,
            state_group_coupling_layers,
            key=key_group_mix,
        )
        self.group_token_film_scale = eqx.nn.Linear(
            self.descriptor_dim,
            state_group_token_dim,
            key=key_group_token_film_scale,
        )
        self.group_token_film_shift = eqx.nn.Linear(
            self.descriptor_dim,
            state_group_token_dim,
            key=key_group_token_film_shift,
        )
        self.group_mixer_film_scale = eqx.nn.Linear(
            self.descriptor_dim,
            state_group_token_dim,
            key=key_group_mixer_film_scale,
        )
        self.group_mixer_film_shift = eqx.nn.Linear(
            self.descriptor_dim,
            state_group_token_dim,
            key=key_group_mixer_film_shift,
        )
        self.decoder_film_scale = eqx.nn.Linear(
            self.descriptor_dim,
            self.max_state_dim,
            key=key_decoder_film_scale,
        )
        self.decoder_film_shift = eqx.nn.Linear(
            self.descriptor_dim,
            self.max_state_dim,
            key=key_decoder_film_shift,
        )
        self.encoder_mean = ResidualMLP(
            encoder_input_dim,
            shared_hidden_dim,
            latent_dim,
            n_encoder_layers,
            key=key_enc_mean,
        )
        self.encoder_logvar = ResidualMLP(
            encoder_input_dim,
            shared_hidden_dim,
            latent_dim,
            n_encoder_layers,
            key=key_enc_logvar,
        )
        self.decoder = ResidualMLP(
            decoder_input_dim,
            shared_hidden_dim,
            self.max_state_dim,
            n_decoder_layers,
            key=key_dec,
        )
        self.drift = ResidualMLP(
            drift_input_dim,
            shared_hidden_dim,
            latent_dim,
            n_drift_layers,
            key=key_drift,
        )
        if self.neural_cde_enabled:
            self.cde_matrix = ResidualMLP(
                drift_input_dim,
                neural_cde_hidden_dim,
                latent_dim * self.path_dim,
                neural_cde_layers,
                key=key_cde,
            )
        else:
            self.cde_matrix = None

        if self.adapters_enabled and enable_encoder_adapter:
            self.encoder_adapter = BottleneckAdapter(
                encoder_input_dim,
                context_dim,
                adapter_bottleneck_dim,
                residual_scale=adapter_residual_scale,
                key=key_encoder_adapter,
            )
        else:
            self.encoder_adapter = None
        if self.adapters_enabled and enable_drift_adapter:
            self.drift_adapter = BottleneckAdapter(
                drift_input_dim,
                context_dim,
                adapter_bottleneck_dim,
                residual_scale=adapter_residual_scale,
                key=key_drift_adapter,
            )
        else:
            self.drift_adapter = None
        if self.adapters_enabled and enable_decoder_adapter:
            self.decoder_adapter = BottleneckAdapter(
                decoder_input_dim,
                context_dim,
                adapter_bottleneck_dim,
                residual_scale=adapter_residual_scale,
                key=key_decoder_adapter,
            )
        else:
            self.decoder_adapter = None

    @classmethod
    def from_config(
        cls,
        config: dict,
        metadata: UniversalSystemMetadata,
        key: PRNGKeyArray,
    ) -> "UniversalDigitalTwin":
        model_cfg = config.get("model", {})
        neural_cde_cfg = model_cfg.get("neural_cde", {})
        latent_solver_cfg = model_cfg.get("latent_solver", {})
        channel_conditioning_cfg = model_cfg.get("channel_conditioning", {})
        law_conditioning_cfg = model_cfg.get("law_conditioning", {})
        adapter_cfg = model_cfg.get("adapters", {})
        return cls(
            metadata,
            latent_dim=int(model_cfg.get("latent_dim", 32)),
            shared_hidden_dim=int(model_cfg.get("shared_hidden_dim", 128)),
            system_embedding_dim=int(model_cfg.get("system_embedding_dim", 32)),
            state_group_token_dim=int(
                model_cfg.get(
                    "state_group_token_dim",
                    model_cfg.get("shared_hidden_dim", 128),
                )
            ),
            group_kind_dim=int(model_cfg.get("state_group_kind_dim", 16)),
            state_group_encoder_layers=int(model_cfg.get("state_group_encoder_layers", 2)),
            state_group_coupling_layers=int(model_cfg.get("state_group_coupling_layers", 2)),
            n_encoder_layers=int(model_cfg.get("encoder_layers", 3)),
            n_decoder_layers=int(model_cfg.get("decoder_layers", 3)),
            n_drift_layers=int(model_cfg.get("drift_layers", 3)),
            use_system_spec_embedding=bool(model_cfg.get("use_system_spec_embedding", True)),
            neural_cde_enabled=bool(neural_cde_cfg.get("enabled", False)),
            neural_cde_hidden_dim=int(
                neural_cde_cfg.get("hidden_dim", model_cfg.get("shared_hidden_dim", 128))
            ),
            neural_cde_layers=int(neural_cde_cfg.get("n_layers", 2)),
            latent_solver_method=str(latent_solver_cfg.get("method", "heun")),
            latent_solver_rtol=float(latent_solver_cfg.get("rtol", 1e-3)),
            latent_solver_atol=float(latent_solver_cfg.get("atol", 1e-4)),
            latent_solver_dt0_factor=float(latent_solver_cfg.get("dt0_factor", 0.5)),
            latent_solver_max_steps=int(latent_solver_cfg.get("max_steps", 512)),
            use_variational_encoder=bool(model_cfg.get("use_variational_encoder", True)),
            channel_conditioning_enabled=bool(channel_conditioning_cfg.get("enabled", False)),
            law_conditioning_enabled=bool(law_conditioning_cfg.get("enabled", False)),
            adapters_enabled=bool(adapter_cfg.get("enabled", False)),
            adapter_bottleneck_dim=int(adapter_cfg.get("bottleneck_dim", 16)),
            adapter_residual_scale=float(adapter_cfg.get("residual_scale", 0.1)),
            enable_encoder_adapter=bool(adapter_cfg.get("encoder", True)),
            enable_drift_adapter=bool(adapter_cfg.get("drift", True)),
            enable_decoder_adapter=bool(adapter_cfg.get("decoder", True)),
            key=key,
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            eqx.tree_serialise_leaves(f, self)
        print(f"Universal model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        config: dict,
        metadata: UniversalSystemMetadata,
    ) -> "UniversalDigitalTwin":
        template = cls.from_config(config, metadata, jax.random.PRNGKey(0))
        with open(path, "rb") as f:
            model = eqx.tree_deserialise_leaves(f, template)
        print(f"Universal model loaded from {path}")
        return model

    def trainable_filter_spec(
        self,
        mode: Literal["full", "adapters"] = "full",
        *,
        include_calibration: bool = False,
    ):
        """Build an Equinox filter spec for pretraining or calibration."""

        fixed_replace = (
            False,
        ) * 28
        calibration_replace = (False,) * 7
        fixed_selector = lambda m: (  # noqa: E731
            m.state_center_table,
            m.state_scale_table,
            m.control_center_table,
            m.control_scale_table,
            m.disturbance_center_table,
            m.disturbance_scale_table,
            m.param_scale_table,
            m.state_mask_table,
            m.control_mask_table,
            m.disturbance_mask_table,
            m.param_mask_table,
            m.descriptor_table,
            m.state_group_mask_table,
            m.state_group_active_table,
            m.state_group_kind_id_table,
            m.family_id_table,
            m.subtype_id_table,
            m.law_tag_mask_table,
            m.conditioning_value_id_table,
            m.parameter_law_tag_id_table,
            m.state_role_id_table,
            m.control_role_id_table,
            m.disturbance_role_id_table,
            m.state_name_id_table,
            m.control_name_id_table,
            m.disturbance_name_id_table,
            m.law_feature_table,
            m.param_bias_mask_table,
        )
        calibration_selector = lambda m: (  # noqa: E731
            m.state_center_delta_table,
            m.state_scale_log_delta_table,
            m.control_center_delta_table,
            m.control_scale_log_delta_table,
            m.disturbance_center_delta_table,
            m.disturbance_scale_log_delta_table,
            m.param_bias_table,
        )

        if mode == "full":
            spec = jax.tree.map(eqx.is_inexact_array, self)
        elif mode == "adapters":
            spec = jax.tree.map(lambda _: False, self)
            if self.encoder_adapter is not None:
                spec = eqx.tree_at(
                    lambda m: m.encoder_adapter,
                    spec,
                    replace=jax.tree.map(eqx.is_inexact_array, self.encoder_adapter),
                )
            if self.drift_adapter is not None:
                spec = eqx.tree_at(
                    lambda m: m.drift_adapter,
                    spec,
                    replace=jax.tree.map(eqx.is_inexact_array, self.drift_adapter),
                )
            if self.decoder_adapter is not None:
                spec = eqx.tree_at(
                    lambda m: m.decoder_adapter,
                    spec,
                    replace=jax.tree.map(eqx.is_inexact_array, self.decoder_adapter),
                )
        else:
            raise ValueError(f"Unsupported universal trainable mode: {mode}")

        spec = eqx.tree_at(fixed_selector, spec, replace=fixed_replace)
        if not include_calibration:
            spec = eqx.tree_at(
                calibration_selector,
                spec,
                replace=calibration_replace,
            )
        return spec

    def configure_param_calibration_mask(
        self,
        system_id: int,
        active_param_indices: list[int] | None = None,
    ) -> "UniversalDigitalTwin":
        """Return a model with a per-system physics-parameter calibration mask."""

        base_mask = jnp.zeros((self.max_param_dim,), dtype=jnp.float32)
        if active_param_indices is None:
            row = self.param_mask_table[system_id].astype(jnp.float32)
        else:
            valid = [idx for idx in active_param_indices if 0 <= int(idx) < self.max_param_dim]
            if valid:
                row = base_mask.at[jnp.asarray(valid, dtype=jnp.int32)].set(1.0)
            else:
                row = base_mask
            row = row * self.param_mask_table[system_id].astype(jnp.float32)
        new_mask_table = self.param_bias_mask_table.at[system_id].set(row)
        return eqx.tree_at(lambda m: m.param_bias_mask_table, self, new_mask_table)

    def _static_conditioning_context(self, system_id: Array) -> Array:
        context = self.system_embedding_table[system_id]
        context = context + 0.5 * self.family_embedding_table[self.family_id_table[system_id]]
        context = context + 0.5 * self.subtype_embedding_table[self.subtype_id_table[system_id]]
        context = context + 0.5 * _masked_embedding_mean(
            self.law_embedding_table,
            self.law_tag_mask_table[system_id],
        )
        conditioning_embeddings = (
            self.conditioning_category_embedding_table
            + self.conditioning_value_embedding_table[self.conditioning_value_id_table[system_id]]
        )
        context = context + 0.5 * jnp.mean(conditioning_embeddings, axis=0)
        if self.descriptor_proj is not None:
            context = context + 0.5 * jax.nn.gelu(
                self.descriptor_proj(self.descriptor_table[system_id])
            )
        if self.law_feature_proj is not None:
            context = context + 0.5 * jax.nn.gelu(
                self.law_feature_proj(self.law_feature_table[system_id])
            )
        return context

    def _parameter_conditioning_summary(
        self,
        system_id: Array,
        params_scaled: Array,
        param_mask: Array,
    ) -> Array:
        law_embeddings = self.law_embedding_table[self.parameter_law_tag_id_table[system_id]]
        weights = param_mask.astype(law_embeddings.dtype) * (
            1.0 + 0.1 * jnp.tanh(params_scaled)
        )
        return _masked_embedding_mean(law_embeddings, weights)

    def _system_context(
        self,
        system_id: Array,
        params_scaled: Array | None = None,
        param_mask: Array | None = None,
        control_mask: Array | None = None,
        disturbance_mask: Array | None = None,
    ) -> Array:
        context = self._static_conditioning_context(system_id)
        if params_scaled is not None and param_mask is not None:
            context = context + 0.5 * self._parameter_conditioning_summary(
                system_id,
                params_scaled,
                param_mask,
            )
        if self.channel_context_proj is not None:
            context = context + 0.5 * self._channel_conditioning_summary(
                system_id,
                control_mask=control_mask,
                disturbance_mask=disturbance_mask,
            )
        return context

    def _channel_semantic_mean(
        self,
        role_embedding_table: Array,
        name_id_table: Array,
        role_id_table: Array,
        system_id: Array,
        weights: Array,
    ) -> Array:
        semantics = (
            role_embedding_table[role_id_table[system_id]]
            + self.channel_name_embedding_table[name_id_table[system_id]]
        )
        return _masked_embedding_mean(semantics, weights)

    def _channel_conditioning_summary(
        self,
        system_id: Array,
        *,
        control_mask: Array | None = None,
        disturbance_mask: Array | None = None,
    ) -> Array:
        state_summary = self._channel_semantic_mean(
            self.state_role_embedding_table,
            self.state_name_id_table,
            self.state_role_id_table,
            system_id,
            self.state_mask_table[system_id],
        )
        control_weights = (
            self.control_mask_table[system_id]
            if control_mask is None
            else self.control_mask_table[system_id] * control_mask
        )
        control_summary = self._channel_semantic_mean(
            self.control_role_embedding_table,
            self.control_name_id_table,
            self.control_role_id_table,
            system_id,
            control_weights,
        )
        disturbance_weights = (
            self.disturbance_mask_table[system_id]
            if disturbance_mask is None
            else self.disturbance_mask_table[system_id] * disturbance_mask
        )
        disturbance_summary = self._channel_semantic_mean(
            self.disturbance_role_embedding_table,
            self.disturbance_name_id_table,
            self.disturbance_role_id_table,
            system_id,
            disturbance_weights,
        )
        return jax.nn.gelu(
            self.channel_context_proj(
                (state_summary + control_summary + disturbance_summary) / 3.0
            )
        )

    def _apply_descriptor_film(
        self,
        values: Array,
        descriptor: Array,
        scale_layer: eqx.nn.Linear,
        shift_layer: eqx.nn.Linear,
    ) -> Array:
        scale = 1.0 + 0.1 * jax.nn.tanh(scale_layer(descriptor))
        shift = 0.1 * jax.nn.tanh(shift_layer(descriptor))
        if values.ndim == 2:
            scale = scale[None, :]
            shift = shift[None, :]
        return values * scale + shift

    def _apply_adapter(
        self,
        adapter: BottleneckAdapter | None,
        values: Array,
        context: Array,
    ) -> Array:
        if adapter is None:
            return values
        return adapter(values, context)

    def _state_group_tables(self, system_id: Array) -> tuple[Array, Array, Array, Array]:
        group_masks = self.state_group_mask_table[system_id]
        group_active = self.state_group_active_table[system_id]
        group_kind_ids = self.state_group_kind_id_table[system_id]
        group_kind_embeddings = self.state_group_kind_embedding_table[group_kind_ids]
        return group_masks, group_active, group_kind_ids, group_kind_embeddings

    def _encode_state_groups(
        self,
        state_norm: Array,
        state_mask: Array,
        system_id: Array,
        params_scaled: Array | None = None,
        param_mask: Array | None = None,
    ) -> Array:
        context = self._system_context(system_id, params_scaled, param_mask)
        descriptor = self.descriptor_table[system_id]
        group_masks, group_active, _, group_kind_embeddings = self._state_group_tables(system_id)
        effective_masks = group_masks * state_mask[None, :]
        context_tokens = jnp.broadcast_to(context, (self.max_state_groups, context.shape[0]))
        group_features = jnp.concatenate(
            [
                effective_masks * state_norm[None, :],
                effective_masks,
                group_kind_embeddings,
                context_tokens,
            ],
            axis=-1,
        )
        group_tokens = jax.vmap(self.state_group_encoder)(group_features)
        group_tokens = self._apply_descriptor_film(
            group_tokens,
            descriptor,
            self.group_token_film_scale,
            self.group_token_film_shift,
        )
        group_tokens = group_tokens * group_active[:, None]

        denom = jnp.maximum(jnp.sum(group_active), jnp.asarray(1.0, dtype=group_tokens.dtype))
        pooled = jnp.sum(group_tokens, axis=0) / denom
        pooled_tokens = jnp.broadcast_to(pooled, group_tokens.shape)
        coupling_features = jnp.concatenate(
            [
                group_tokens,
                pooled_tokens,
                group_kind_embeddings,
                context_tokens,
            ],
            axis=-1,
        )
        mixed_updates = jax.vmap(self.state_group_mixer)(coupling_features)
        mixed_tokens = group_tokens + mixed_updates * group_active[:, None]
        mixed_tokens = self._apply_descriptor_film(
            mixed_tokens,
            descriptor,
            self.group_mixer_film_scale,
            self.group_mixer_film_shift,
        )
        mixed_tokens = mixed_tokens * group_active[:, None]
        return jnp.sum(mixed_tokens, axis=0) / denom

    def _effective_state_center(self, system_ids: Array) -> Array:
        return self.state_center_table[system_ids] + self.state_center_delta_table[system_ids]

    def _effective_state_scale(self, system_ids: Array) -> Array:
        log_delta = jnp.clip(self.state_scale_log_delta_table[system_ids], -2.0, 2.0)
        return self.state_scale_table[system_ids] * jnp.exp(log_delta)

    def _effective_control_center(self, system_ids: Array) -> Array:
        return self.control_center_table[system_ids] + self.control_center_delta_table[system_ids]

    def _effective_control_scale(self, system_ids: Array) -> Array:
        log_delta = jnp.clip(self.control_scale_log_delta_table[system_ids], -2.0, 2.0)
        return self.control_scale_table[system_ids] * jnp.exp(log_delta)

    def _effective_disturbance_center(self, system_ids: Array) -> Array:
        return (
            self.disturbance_center_table[system_ids]
            + self.disturbance_center_delta_table[system_ids]
        )

    def _effective_disturbance_scale(self, system_ids: Array) -> Array:
        log_delta = jnp.clip(
            self.disturbance_scale_log_delta_table[system_ids],
            -2.0,
            2.0,
        )
        return self.disturbance_scale_table[system_ids] * jnp.exp(log_delta)

    def normalize_states(self, states: Array, system_ids: Array) -> Array:
        center = self._effective_state_center(system_ids)
        scale = self._effective_state_scale(system_ids)
        if states.ndim == 3:
            center = center[:, None, :]
            scale = scale[:, None, :]
        return (states - center) * scale

    def denormalize_states(self, states: Array, system_ids: Array) -> Array:
        center = self._effective_state_center(system_ids)
        scale = self._effective_state_scale(system_ids)
        if states.ndim == 3:
            center = center[:, None, :]
            scale = scale[:, None, :]
        return states / jnp.maximum(scale, 1e-6) + center

    def normalize_controls(self, controls: Array, system_ids: Array) -> Array:
        center = self._effective_control_center(system_ids)
        scale = self._effective_control_scale(system_ids)
        if controls.ndim == 3:
            center = center[:, None, :]
            scale = scale[:, None, :]
        return (controls - center) * scale

    def normalize_disturbances(self, disturbances: Array, system_ids: Array) -> Array:
        center = self._effective_disturbance_center(system_ids)
        scale = self._effective_disturbance_scale(system_ids)
        if disturbances.ndim == 3:
            center = center[:, None, :]
            scale = scale[:, None, :]
        return (disturbances - center) * scale

    def scale_params(self, params: Array, system_ids: Array) -> Array:
        calibrated_params = params + self.param_bias_table[system_ids] * self.param_bias_mask_table[system_ids]
        scale = self.param_scale_table[system_ids]
        return jnp.sign(calibrated_params) * jnp.log1p(jnp.abs(calibrated_params)) * scale

    def encode(
        self,
        state_norm: Array,
        params_scaled: Array,
        control_norm: Array,
        state_mask: Array,
        control_mask: Array,
        param_mask: Array,
        system_id: Array,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, Array, Array]:
        state_summary = self._encode_state_groups(
            state_norm,
            state_mask,
            system_id,
            params_scaled,
            param_mask,
        )
        context = self._system_context(
            system_id,
            params_scaled,
            param_mask,
            control_mask,
        )
        features = jnp.concatenate(
            [
                state_summary,
                control_norm * control_mask,
                control_mask,
                params_scaled * param_mask,
                param_mask,
                context,
            ]
        )
        features = self._apply_adapter(self.encoder_adapter, features, context)
        z_mean = self.encoder_mean(features)
        z_logvar = jnp.clip(self.encoder_logvar(features), -8.0, 5.0)
        if key is None or not self.use_variational_encoder:
            z = z_mean
        else:
            std = jnp.exp(0.5 * z_logvar)
            eps = jax.random.normal(key, z_mean.shape)
            z_sample = z_mean + eps * std
            z = 0.5 * z_mean + 0.5 * z_sample
        return z, z_mean, z_logvar

    def decode(
        self,
        z: Array,
        params_scaled: Array,
        control_norm: Array,
        state_mask: Array,
        control_mask: Array,
        param_mask: Array,
        system_id: Array,
    ) -> Array:
        context = self._system_context(
            system_id,
            params_scaled,
            param_mask,
            control_mask,
        )
        descriptor = self.descriptor_table[system_id]
        group_masks, group_active, _, group_kind_embeddings = self._state_group_tables(system_id)
        effective_masks = group_masks * state_mask[None, :]
        shared_features = jnp.concatenate(
            [
                z,
                control_norm * control_mask,
                control_mask,
                params_scaled * param_mask,
                param_mask,
                context,
            ]
        )
        shared_features = jnp.broadcast_to(
            shared_features,
            (self.max_state_groups, shared_features.shape[0]),
        )
        group_features = jnp.concatenate(
            [
                shared_features,
                effective_masks,
                group_kind_embeddings,
            ],
            axis=-1,
        )
        if self.decoder_adapter is not None:
            group_features = jax.vmap(
                lambda features: self.decoder_adapter(features, context)
            )(group_features)
        group_outputs = jax.vmap(self.decoder)(group_features)
        group_outputs = self._apply_descriptor_film(
            group_outputs,
            descriptor,
            self.decoder_film_scale,
            self.decoder_film_shift,
        )
        decoded = jnp.sum(group_outputs * effective_masks * group_active[:, None], axis=0)
        return decoded * state_mask

    def control_path_term(
        self,
        z: Array,
        control_norm: Array,
        disturbance_norm: Array,
        params_scaled: Array,
        control_mask: Array,
        disturbance_mask: Array,
        param_mask: Array,
        system_id: Array,
        path_derivative: Array,
    ) -> Array:
        if not self.neural_cde_enabled or self.cde_matrix is None:
            return jnp.zeros_like(z)
        context = self._system_context(
            system_id,
            params_scaled,
            param_mask,
            control_mask,
            disturbance_mask,
        )
        features = jnp.concatenate(
            [
                z,
                control_norm * control_mask,
                control_mask,
                disturbance_norm * disturbance_mask,
                disturbance_mask,
                params_scaled * param_mask,
                param_mask,
                context,
            ]
        )
        features = self._apply_adapter(self.drift_adapter, features, context)
        matrix = self.cde_matrix(features).reshape(self.latent_dim, self.path_dim)
        return matrix @ path_derivative

    def _latent_solver(self):
        method = self.latent_solver_method
        if method == "heun":
            return None
        if method == "tsit5":
            return diffrax.Tsit5()
        if method == "kvaerno5":
            return diffrax.Kvaerno5()
        if method == "implicit_euler":
            return diffrax.ImplicitEuler()
        raise ValueError(f"Unsupported universal latent solver '{method}'.")

    def _piecewise_linear_value_and_derivative(
        self,
        ts: Array,
        values: Array,
        t: Array,
    ) -> tuple[Array, Array]:
        idx = jnp.clip(
            jnp.searchsorted(ts[1:], t, side="right"),
            0,
            ts.shape[0] - 2,
        )
        t0 = ts[idx]
        t1 = ts[idx + 1]
        v0 = values[idx]
        v1 = values[idx + 1]
        safe_dt = jnp.maximum(t1 - t0, jnp.asarray(1e-6, dtype=ts.dtype))
        alpha = jnp.clip((t - t0) / safe_dt, 0.0, 1.0)
        value = v0 + alpha * (v1 - v0)
        derivative = (v1 - v0) / safe_dt
        return value, derivative

    def _latent_drift_at_time(
        self,
        t: Array,
        z: Array,
        ts: Array,
        controls_norm: Array,
        disturbances_norm: Array,
        params_scaled: Array,
        control_mask: Array,
        disturbance_mask: Array,
        param_mask: Array,
        system_id: Array,
    ) -> Array:
        control_t, control_derivative = self._piecewise_linear_value_and_derivative(
            ts,
            controls_norm,
            t,
        )
        disturbance_t, disturbance_derivative = self._piecewise_linear_value_and_derivative(
            ts,
            disturbances_norm,
            t,
        )
        path_derivative = None
        if self.neural_cde_enabled:
            path_derivative = jnp.concatenate(
                [
                    jnp.asarray([1.0], dtype=z.dtype),
                    control_derivative,
                    disturbance_derivative,
                ]
            )
        return self.latent_drift(
            z,
            control_t,
            disturbance_t,
            params_scaled,
            control_mask,
            disturbance_mask,
            param_mask,
            system_id,
            path_derivative,
        )

    def _integrate_latent_interval(
        self,
        z_prev: Array,
        ts: Array,
        controls_norm: Array,
        disturbances_norm: Array,
        params_scaled: Array,
        control_mask: Array,
        disturbance_mask: Array,
        param_mask: Array,
        system_id: Array,
    ) -> Array:
        solver = self._latent_solver()
        if solver is None:
            dt = ts[1] - ts[0]
            safe_dt = jnp.maximum(dt, jnp.asarray(1e-6, dtype=dt.dtype))
            control_mid = 0.5 * (controls_norm[0] + controls_norm[1])
            disturbance_mid = 0.5 * (disturbances_norm[0] + disturbances_norm[1])
            path_derivative = jnp.concatenate(
                [
                    jnp.asarray([1.0], dtype=z_prev.dtype),
                    (controls_norm[1] - controls_norm[0]) / safe_dt,
                    (disturbances_norm[1] - disturbances_norm[0]) / safe_dt,
                ]
            )

            def total_drift(z_curr: Array) -> Array:
                return self.latent_drift(
                    z_curr,
                    control_mid,
                    disturbance_mid,
                    params_scaled,
                    control_mask,
                    disturbance_mask,
                    param_mask,
                    system_id,
                    path_derivative if self.neural_cde_enabled else None,
                )

            k1 = total_drift(z_prev)
            z_euler = z_prev + dt * k1
            k2 = total_drift(z_euler)
            return z_prev + 0.5 * dt * (k1 + k2)

        t0 = ts[0]
        t1 = ts[-1]
        safe_dt0 = jnp.maximum(
            jnp.abs(ts[1] - ts[0]) * self.latent_solver_dt0_factor,
            jnp.asarray(1e-4, dtype=ts.dtype),
        )

        def drift_fn(t, z, args):
            del args
            return self._latent_drift_at_time(
                t,
                z,
                ts,
                controls_norm,
                disturbances_norm,
                params_scaled,
                control_mask,
                disturbance_mask,
                param_mask,
                system_id,
            )

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(drift_fn),
            solver,
            t0=t0,
            t1=t1,
            dt0=safe_dt0,
            y0=z_prev,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(
                rtol=self.latent_solver_rtol,
                atol=self.latent_solver_atol,
            ),
            max_steps=self.latent_solver_max_steps,
        )
        return solution.ys[0]

    def latent_drift(
        self,
        z: Array,
        control_norm: Array,
        disturbance_norm: Array,
        params_scaled: Array,
        control_mask: Array,
        disturbance_mask: Array,
        param_mask: Array,
        system_id: Array,
        path_derivative: Array | None = None,
    ) -> Array:
        context = self._system_context(
            system_id,
            params_scaled,
            param_mask,
            control_mask,
            disturbance_mask,
        )
        features = jnp.concatenate(
            [
                z,
                control_norm * control_mask,
                control_mask,
                disturbance_norm * disturbance_mask,
                disturbance_mask,
                params_scaled * param_mask,
                param_mask,
                context,
            ]
        )
        features = self._apply_adapter(self.drift_adapter, features, context)
        drift = self.drift(features)
        if path_derivative is not None:
            drift = drift + self.control_path_term(
                z,
                control_norm,
                disturbance_norm,
                params_scaled,
                control_mask,
                disturbance_mask,
                param_mask,
                system_id,
                path_derivative,
            )
        return drift

    def latent_step(
        self,
        z_prev: Array,
        control_t: Array,
        control_tp1: Array,
        disturbance_t: Array,
        disturbance_tp1: Array,
        params_scaled: Array,
        control_mask: Array,
        disturbance_mask: Array,
        param_mask: Array,
        system_id: Array,
        dt: Array,
    ) -> Array:
        safe_dt = jnp.maximum(dt, jnp.asarray(1e-6, dtype=dt.dtype))
        local_ts = jnp.asarray([0.0, safe_dt], dtype=dt.dtype)
        local_controls = jnp.stack([control_t, control_tp1], axis=0)
        local_disturbances = jnp.stack([disturbance_t, disturbance_tp1], axis=0)
        return jax.lax.cond(
            dt <= 0,
            lambda _: z_prev,
            lambda _: self._integrate_latent_interval(
                z_prev,
                local_ts,
                local_controls,
                local_disturbances,
                params_scaled,
                control_mask,
                disturbance_mask,
                param_mask,
                system_id,
            ),
            operand=None,
        )

    def rollout_latent(
        self,
        ts: Array,
        z0: Array,
        controls_norm: Array,
        disturbances_norm: Array,
        params_scaled: Array,
        control_mask: Array,
        disturbance_mask: Array,
        param_mask: Array,
        system_id: Array,
    ) -> Array:
        if ts.shape[0] <= 1:
            return z0[None, :]

        if self.latent_solver_method == "heun":
            dt_steps = ts[1:] - ts[:-1]

            def step_fn(z_prev: Array, step_inputs: tuple[Array, ...]):
                u_t, u_tp1, d_t, d_tp1, step_dt = step_inputs
                z_next = self.latent_step(
                    z_prev,
                    u_t,
                    u_tp1,
                    d_t,
                    d_tp1,
                    params_scaled,
                    control_mask,
                    disturbance_mask,
                    param_mask,
                    system_id,
                    step_dt,
                )
                return z_next, z_next

            _, z_hist = jax.lax.scan(
                step_fn,
                z0,
                (
                    controls_norm[:-1],
                    controls_norm[1:],
                    disturbances_norm[:-1],
                    disturbances_norm[1:],
                    dt_steps,
                ),
            )
            return jnp.concatenate([z0[None, :], z_hist], axis=0)

        solver = self._latent_solver()
        safe_dt0 = jnp.maximum(
            jnp.abs(ts[1] - ts[0]) * self.latent_solver_dt0_factor,
            jnp.asarray(1e-4, dtype=ts.dtype),
        )

        def drift_fn(t, z, args):
            del args
            return self._latent_drift_at_time(
                t,
                z,
                ts,
                controls_norm,
                disturbances_norm,
                params_scaled,
                control_mask,
                disturbance_mask,
                param_mask,
                system_id,
            )

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(drift_fn),
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=safe_dt0,
            y0=z0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(
                rtol=self.latent_solver_rtol,
                atol=self.latent_solver_atol,
            ),
            max_steps=max(self.latent_solver_max_steps, int(ts.shape[0]) * 16),
        )
        return solution.ys

    def get_parameter_count(self) -> Dict[str, int]:
        return {
            "system_embeddings": _count_params(self.system_embedding_table),
            "family_embeddings": _count_params(self.family_embedding_table),
            "subtype_embeddings": _count_params(self.subtype_embedding_table),
            "law_embeddings": _count_params(self.law_embedding_table),
            "conditioning_embeddings": _count_params(
                (
                    self.conditioning_category_embedding_table,
                    self.conditioning_value_embedding_table,
                )
            ),
            "state_group_encoder": _count_params(self.state_group_encoder),
            "state_group_mixer": _count_params(self.state_group_mixer),
            "group_token_film_scale": _count_params(self.group_token_film_scale),
            "group_token_film_shift": _count_params(self.group_token_film_shift),
            "group_mixer_film_scale": _count_params(self.group_mixer_film_scale),
            "group_mixer_film_shift": _count_params(self.group_mixer_film_shift),
            "decoder_film_scale": _count_params(self.decoder_film_scale),
            "decoder_film_shift": _count_params(self.decoder_film_shift),
            "encoder_mean": _count_params(self.encoder_mean),
            "encoder_logvar": _count_params(self.encoder_logvar),
            "decoder": _count_params(self.decoder),
            "drift": _count_params(self.drift),
            "cde_matrix": _count_params(self.cde_matrix) if self.cde_matrix is not None else 0,
            "encoder_adapter": _count_params(self.encoder_adapter)
            if self.encoder_adapter is not None
            else 0,
            "drift_adapter": _count_params(self.drift_adapter)
            if self.drift_adapter is not None
            else 0,
            "decoder_adapter": _count_params(self.decoder_adapter)
            if self.decoder_adapter is not None
            else 0,
            "calibration_tables": _count_params(
                (
                    self.state_center_delta_table,
                    self.state_scale_log_delta_table,
                    self.control_center_delta_table,
                    self.control_scale_log_delta_table,
                    self.disturbance_center_delta_table,
                    self.disturbance_scale_log_delta_table,
                    self.param_bias_table,
                )
            ),
            "total": _count_params(self),
        }
