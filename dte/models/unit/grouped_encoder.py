"""Grouped typed encoder for ProcessUnitSpec-aware single-system training."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from dte.simulators.base import SystemSpec


def _masked_embedding_mean(embeddings: Array, weights: Array) -> Array:
    weights = weights.astype(embeddings.dtype)
    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=embeddings.dtype))
    return jnp.sum(embeddings * weights[:, None], axis=0) / denom


class ResidualMLP(eqx.Module):
    """Small residual MLP used by the grouped encoder path."""

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


class GroupedStateEncoder(eqx.Module):
    """State-group-aware encoder with the same interface as ``Encoder``."""

    state_group_encoder: ResidualMLP
    state_group_mixer: ResidualMLP
    summary_head: ResidualMLP
    mean_layer: eqx.nn.Linear
    logvar_layer: eqx.nn.Linear

    state_center: Float[Array, "state_dim"]
    state_scale: Float[Array, "state_dim"]
    control_center: Float[Array, "control_dim"]
    control_scale: Float[Array, "control_dim"]
    group_masks: Float[Array, "max_groups state_dim"]
    group_active: Float[Array, "max_groups"]
    group_kind_ids: Array
    group_kind_embedding_table: Float[Array, "n_group_kinds group_kind_dim"]
    state_role_ids: Array
    state_name_ids: Array
    control_role_ids: Array
    control_name_ids: Array
    state_role_embedding_table: Float[Array, "n_state_roles group_kind_dim"]
    control_role_embedding_table: Float[Array, "n_control_roles group_kind_dim"]
    channel_name_embedding_table: Float[Array, "n_channel_names group_kind_dim"]
    law_feature_defaults: Float[Array, "n_law_features"]
    state_channel_proj: eqx.nn.Linear | None
    control_channel_proj: eqx.nn.Linear | None
    law_feature_proj: eqx.nn.Linear | None

    param_scale: float = eqx.field(static=True)
    max_groups: int = eqx.field(static=True)
    group_kind_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    channel_conditioning_enabled: bool = eqx.field(static=True)
    law_conditioning_enabled: bool = eqx.field(static=True)

    def __init__(
        self,
        system_spec: SystemSpec,
        *,
        param_dim: int,
        control_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
        group_token_dim: int,
        group_kind_dim: int,
        group_encoder_layers: int,
        group_mixer_layers: int,
        channel_conditioning_enabled: bool,
        law_conditioning_enabled: bool,
        key: PRNGKeyArray,
    ):
        group_kinds: list[str] = []
        for group in system_spec.state_groups:
            if group.kind not in group_kinds:
                group_kinds.append(group.kind)
        if "generic" not in group_kinds:
            group_kinds.append("generic")
        kind_to_id = {name: idx for idx, name in enumerate(group_kinds)}

        group_masks = jnp.zeros(
            (len(system_spec.state_groups), system_spec.state_dim),
            dtype=jnp.float32,
        )
        group_active = jnp.ones((len(system_spec.state_groups),), dtype=jnp.float32)
        group_kind_ids = jnp.zeros((len(system_spec.state_groups),), dtype=jnp.int32)
        for group_idx, group in enumerate(system_spec.state_groups):
            group_masks = group_masks.at[group_idx, jnp.asarray(group.indices)].set(1.0)
            group_kind_ids = group_kind_ids.at[group_idx].set(kind_to_id[group.kind])

        (
            key_group_embed,
            key_group_enc,
            key_group_mix,
            key_summary,
            key_mean,
            key_logvar,
            key_state_role,
            key_control_role,
            key_channel_name,
            key_state_channel_proj,
            key_control_channel_proj,
            key_law_feature_proj,
        ) = jax.random.split(key, 12)
        self.state_center = jnp.asarray(
            system_spec.normalization.state_center,
            dtype=jnp.float32,
        )
        self.state_scale = jnp.asarray(
            system_spec.normalization.state_scale,
            dtype=jnp.float32,
        )
        self.control_center = jnp.asarray(
            system_spec.normalization.control_center,
            dtype=jnp.float32,
        )
        self.control_scale = jnp.asarray(
            system_spec.normalization.control_scale,
            dtype=jnp.float32,
        )
        self.group_masks = group_masks
        self.group_active = group_active
        self.group_kind_ids = group_kind_ids
        self.group_kind_embedding_table = 0.02 * jax.random.normal(
            key_group_embed,
            (len(group_kinds), group_kind_dim),
        )
        self.law_feature_defaults = jnp.asarray(
            getattr(system_spec, "law_feature_defaults", ()),
            dtype=jnp.float32,
        )
        self.param_scale = system_spec.normalization.param_scale
        self.max_groups = len(system_spec.state_groups)
        self.group_kind_dim = group_kind_dim
        self.context_dim = param_dim + control_dim
        self.channel_conditioning_enabled = bool(channel_conditioning_enabled)
        self.law_conditioning_enabled = bool(law_conditioning_enabled)

        if self.channel_conditioning_enabled:
            state_role_names = list(
                dict.fromkeys(
                    [*getattr(system_spec, "state_role_names", lambda: tuple())(), "generic"]
                )
            )
            control_role_names = list(
                dict.fromkeys(
                    [*getattr(system_spec, "control_role_names", lambda: tuple())(), "generic"]
                )
            )
            channel_names = list(
                dict.fromkeys(
                    [
                        "generic",
                        *getattr(system_spec, "state_channel_names", lambda: tuple())(),
                        *getattr(system_spec, "control_channel_names", lambda: tuple())(),
                    ]
                )
            )
            state_role_to_id = {name: idx for idx, name in enumerate(state_role_names)}
            control_role_to_id = {name: idx for idx, name in enumerate(control_role_names)}
            channel_name_to_id = {name: idx for idx, name in enumerate(channel_names)}
            self.state_role_ids = jnp.asarray(
                [
                    state_role_to_id[getattr(channel, "role", "generic")]
                    for channel in getattr(system_spec, "state_channels", [])
                ],
                dtype=jnp.int32,
            )
            self.state_name_ids = jnp.asarray(
                [
                    channel_name_to_id[str(getattr(channel, "name", "generic"))]
                    for channel in getattr(system_spec, "state_channels", [])
                ],
                dtype=jnp.int32,
            )
            self.control_role_ids = jnp.asarray(
                [
                    control_role_to_id[getattr(channel, "role", "generic")]
                    for channel in getattr(system_spec, "control_channels", [])
                ],
                dtype=jnp.int32,
            )
            self.control_name_ids = jnp.asarray(
                [
                    channel_name_to_id[str(getattr(channel, "name", "generic"))]
                    for channel in getattr(system_spec, "control_channels", [])
                ],
                dtype=jnp.int32,
            )
            self.state_role_embedding_table = 0.02 * jax.random.normal(
                key_state_role,
                (len(state_role_names), group_kind_dim),
            )
            self.control_role_embedding_table = 0.02 * jax.random.normal(
                key_control_role,
                (len(control_role_names), group_kind_dim),
            )
            self.channel_name_embedding_table = 0.02 * jax.random.normal(
                key_channel_name,
                (len(channel_names), group_kind_dim),
            )
            self.state_channel_proj = eqx.nn.Linear(
                group_kind_dim,
                group_kind_dim,
                key=key_state_channel_proj,
            )
            self.control_channel_proj = eqx.nn.Linear(
                group_kind_dim,
                self.context_dim,
                key=key_control_channel_proj,
            )
        else:
            self.state_role_ids = jnp.zeros((system_spec.state_dim,), dtype=jnp.int32)
            self.state_name_ids = jnp.zeros((system_spec.state_dim,), dtype=jnp.int32)
            self.control_role_ids = jnp.zeros((system_spec.control_dim,), dtype=jnp.int32)
            self.control_name_ids = jnp.zeros((system_spec.control_dim,), dtype=jnp.int32)
            self.state_role_embedding_table = jnp.zeros((1, group_kind_dim), dtype=jnp.float32)
            self.control_role_embedding_table = jnp.zeros((1, group_kind_dim), dtype=jnp.float32)
            self.channel_name_embedding_table = jnp.zeros((1, group_kind_dim), dtype=jnp.float32)
            self.state_channel_proj = None
            self.control_channel_proj = None

        if self.law_conditioning_enabled and int(self.law_feature_defaults.size) > 0:
            self.law_feature_proj = eqx.nn.Linear(
                int(self.law_feature_defaults.shape[0]),
                self.context_dim,
                key=key_law_feature_proj,
            )
        else:
            self.law_feature_proj = None

        group_input_dim = 2 * system_spec.state_dim + self.context_dim + group_kind_dim
        group_mix_dim = 2 * group_token_dim + self.context_dim + group_kind_dim
        summary_input_dim = group_token_dim + system_spec.state_dim + self.context_dim

        self.state_group_encoder = ResidualMLP(
            group_input_dim,
            hidden_dim,
            group_token_dim,
            group_encoder_layers,
            key=key_group_enc,
        )
        self.state_group_mixer = ResidualMLP(
            group_mix_dim,
            hidden_dim,
            group_token_dim,
            group_mixer_layers,
            key=key_group_mix,
        )
        self.summary_head = ResidualMLP(
            summary_input_dim,
            hidden_dim,
            hidden_dim,
            n_layers,
            key=key_summary,
        )
        self.mean_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=key_mean)
        self.logvar_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=key_logvar)

    def _normalize_inputs(
        self,
        state: Array,
        params: Array,
        control: Array,
    ) -> tuple[Array, Array, Array]:
        state_norm = (state - self.state_center) * self.state_scale
        control_norm = (control - self.control_center) * self.control_scale
        params_scaled = jnp.sign(params) * jnp.log1p(jnp.abs(params)) * self.param_scale
        return state_norm, params_scaled, control_norm

    def _state_group_channel_embeddings(self) -> Array:
        if not self.channel_conditioning_enabled or self.state_channel_proj is None:
            return jnp.zeros((self.max_groups, self.group_kind_dim), dtype=jnp.float32)
        state_semantics = (
            self.state_role_embedding_table[self.state_role_ids]
            + self.channel_name_embedding_table[self.state_name_ids]
        )
        return jax.vmap(
            lambda mask: jax.nn.gelu(
                self.state_channel_proj(_masked_embedding_mean(state_semantics, mask))
            )
        )(self.group_masks)

    def _control_channel_context(self) -> Array:
        if not self.channel_conditioning_enabled or self.control_channel_proj is None:
            return jnp.zeros((self.context_dim,), dtype=jnp.float32)
        control_semantics = (
            self.control_role_embedding_table[self.control_role_ids]
            + self.channel_name_embedding_table[self.control_name_ids]
        )
        summary = _masked_embedding_mean(
            control_semantics,
            jnp.ones((control_semantics.shape[0],), dtype=jnp.float32),
        )
        return jax.nn.gelu(self.control_channel_proj(summary))

    def _law_context(self) -> Array:
        if self.law_feature_proj is None:
            return jnp.zeros((self.context_dim,), dtype=jnp.float32)
        return jax.nn.gelu(self.law_feature_proj(self.law_feature_defaults))

    def encode(
        self,
        state: Float[Array, "state_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
    ) -> tuple[Float[Array, "latent_dim"], Float[Array, "latent_dim"]]:
        state_norm, params_scaled, control_norm = self._normalize_inputs(state, params, control)
        context = jnp.concatenate([params_scaled, control_norm], axis=-1)
        if self.channel_conditioning_enabled:
            context = context + 0.25 * self._control_channel_context()
        if self.law_conditioning_enabled:
            context = context + 0.25 * self._law_context()

        group_kind_embeddings = self.group_kind_embedding_table[self.group_kind_ids]
        if self.channel_conditioning_enabled:
            group_kind_embeddings = group_kind_embeddings + self._state_group_channel_embeddings()
        context_tokens = jnp.broadcast_to(context, (self.max_groups, context.shape[0]))

        group_features = jnp.concatenate(
            [
                self.group_masks * state_norm[None, :],
                self.group_masks,
                group_kind_embeddings,
                context_tokens,
            ],
            axis=-1,
        )
        group_tokens = jax.vmap(self.state_group_encoder)(group_features)
        active = self.group_active[:, None]
        denom = jnp.maximum(jnp.sum(self.group_active), jnp.asarray(1.0, dtype=group_tokens.dtype))
        pooled = jnp.sum(group_tokens * active, axis=0) / denom
        pooled_tokens = jnp.broadcast_to(pooled, group_tokens.shape)
        mixed = jax.vmap(self.state_group_mixer)(
            jnp.concatenate(
                [group_tokens, pooled_tokens, group_kind_embeddings, context_tokens],
                axis=-1,
            )
        )
        summary = jnp.sum((group_tokens + mixed) * active, axis=0) / denom
        hidden = self.summary_head(jnp.concatenate([summary, state_norm, context], axis=-1))
        z_mean = self.mean_layer(hidden)
        z_logvar = jnp.clip(self.logvar_layer(hidden), -10.0, 5.0)
        return z_mean, z_logvar

    def sample(
        self,
        z_mean: Float[Array, "latent_dim"],
        z_logvar: Float[Array, "latent_dim"],
        key: PRNGKeyArray,
    ) -> Float[Array, "latent_dim"]:
        std = jnp.exp(0.5 * z_logvar)
        eps = jax.random.normal(key, shape=z_mean.shape)
        return z_mean + eps * std

    def __call__(
        self,
        state: Float[Array, "state_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "latent_dim"], Float[Array, "latent_dim"], Float[Array, "latent_dim"]]:
        z_mean, z_logvar = self.encode(state, params, control)
        z = self.sample(z_mean, z_logvar, key)
        return z, z_mean, z_logvar
