"""Shared universal digital twin for mixed-system training.

The universal path intentionally starts simpler than the single-system
`DigitalTwin`: it operates in padded normalized space, uses explicit masks, and
conditions on system identity plus a numeric SystemSpec descriptor. This gives
the repo a real shared-checkpoint baseline without forcing the existing
single-system physics-informed stack into a mismatched interface.
"""

from __future__ import annotations

from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from dte.data.multi_system_dataset import UniversalSystemMetadata


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


class UniversalDigitalTwin(eqx.Module):
    """Shared padded digital twin trained across multiple systems."""

    state_group_encoder: ResidualMLP
    state_group_mixer: ResidualMLP
    encoder_mean: ResidualMLP
    encoder_logvar: ResidualMLP
    decoder: ResidualMLP
    drift: ResidualMLP
    cde_matrix: ResidualMLP | None
    descriptor_proj: eqx.nn.Linear | None

    system_embedding_table: Float[Array, "n_systems embedding_dim"]

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
    use_variational_encoder: bool = eqx.field(static=True)

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
        use_variational_encoder: bool,
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
        self.use_variational_encoder = use_variational_encoder

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
        n_group_kinds = len(metadata.state_group_kind_names)
        (
            key_embed,
            key_desc,
            key_group_kind,
            key_group_enc,
            key_group_mix,
            key_enc_mean,
            key_enc_logvar,
            key_dec,
            key_drift,
            key_cde,
        ) = jax.random.split(key, 10)
        self.system_embedding_table = (
            0.02 * jax.random.normal(key_embed, (n_systems, system_embedding_dim))
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

        context_dim = system_embedding_dim
        group_encoder_input_dim = 2 * self.max_state_dim + context_dim + group_kind_dim
        group_coupling_input_dim = 2 * state_group_token_dim + context_dim + group_kind_dim
        encoder_input_dim = state_group_token_dim + 2 * self.max_control_dim + 2 * self.max_param_dim + context_dim
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

    @classmethod
    def from_config(
        cls,
        config: dict,
        metadata: UniversalSystemMetadata,
        key: PRNGKeyArray,
    ) -> "UniversalDigitalTwin":
        model_cfg = config.get("model", {})
        neural_cde_cfg = model_cfg.get("neural_cde", {})
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
            use_variational_encoder=bool(model_cfg.get("use_variational_encoder", True)),
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

    def trainable_filter_spec(self):
        """Exclude fixed normalization/spec tables from optimizer updates."""
        spec = jax.tree.map(eqx.is_inexact_array, self)
        return eqx.tree_at(
            lambda m: (
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
            ),
            spec,
            replace=(False,) * 15,
        )

    def _system_context(self, system_id: Array) -> Array:
        learned = self.system_embedding_table[system_id]
        if self.descriptor_proj is None:
            return learned
        descriptor = self.descriptor_table[system_id]
        return learned + jax.nn.gelu(self.descriptor_proj(descriptor))

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
    ) -> Array:
        context = self._system_context(system_id)
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
        return jnp.sum(mixed_tokens, axis=0) / denom

    def normalize_states(self, states: Array, system_ids: Array) -> Array:
        center = self.state_center_table[system_ids]
        scale = self.state_scale_table[system_ids]
        if states.ndim == 3:
            center = center[:, None, :]
            scale = scale[:, None, :]
        return (states - center) * scale

    def denormalize_states(self, states: Array, system_ids: Array) -> Array:
        center = self.state_center_table[system_ids]
        scale = self.state_scale_table[system_ids]
        if states.ndim == 3:
            center = center[:, None, :]
            scale = scale[:, None, :]
        return states / jnp.maximum(scale, 1e-6) + center

    def normalize_controls(self, controls: Array, system_ids: Array) -> Array:
        center = self.control_center_table[system_ids]
        scale = self.control_scale_table[system_ids]
        if controls.ndim == 3:
            center = center[:, None, :]
            scale = scale[:, None, :]
        return (controls - center) * scale

    def normalize_disturbances(self, disturbances: Array, system_ids: Array) -> Array:
        center = self.disturbance_center_table[system_ids]
        scale = self.disturbance_scale_table[system_ids]
        if disturbances.ndim == 3:
            center = center[:, None, :]
            scale = scale[:, None, :]
        return (disturbances - center) * scale

    def scale_params(self, params: Array, system_ids: Array) -> Array:
        scale = self.param_scale_table[system_ids]
        if params.ndim == 2:
            return jnp.sign(params) * jnp.log1p(jnp.abs(params)) * scale
        return jnp.sign(params) * jnp.log1p(jnp.abs(params)) * scale[None, :]

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
        state_summary = self._encode_state_groups(state_norm, state_mask, system_id)
        context = self._system_context(system_id)
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
        context = self._system_context(system_id)
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
        group_outputs = jax.vmap(self.decoder)(group_features)
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
        context = self._system_context(system_id)
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
        matrix = self.cde_matrix(features).reshape(self.latent_dim, self.path_dim)
        return matrix @ path_derivative

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
        context = self._system_context(system_id)
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
        control_mid = 0.5 * (control_t + control_tp1)
        disturbance_mid = 0.5 * (disturbance_t + disturbance_tp1)
        path_derivative = jnp.concatenate(
            [
                jnp.array([1.0], dtype=z_prev.dtype),
                (control_tp1 - control_t) / safe_dt,
                (disturbance_tp1 - disturbance_t) / safe_dt,
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
            (controls_norm[:-1], controls_norm[1:], disturbances_norm[:-1], disturbances_norm[1:], dt_steps),
        )
        return jnp.concatenate([z0[None, :], z_hist], axis=0)

    def get_parameter_count(self) -> Dict[str, int]:
        def count_params(module):
            return sum(x.size for x in jax.tree.leaves(eqx.filter(module, eqx.is_array)))

        return {
            "state_group_encoder": count_params(self.state_group_encoder),
            "state_group_mixer": count_params(self.state_group_mixer),
            "encoder_mean": count_params(self.encoder_mean),
            "encoder_logvar": count_params(self.encoder_logvar),
            "decoder": count_params(self.decoder),
            "drift": count_params(self.drift),
            "cde_matrix": count_params(self.cde_matrix) if self.cde_matrix is not None else 0,
            "total": count_params(self),
        }
