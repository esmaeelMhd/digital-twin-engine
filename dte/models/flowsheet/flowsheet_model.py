"""Graph-structured flowsheet model built from a shared unit backbone."""

from __future__ import annotations

from typing import Dict

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from dte.data.datasets.flowsheet_dataset import FlowsheetGraphMetadata


class ResidualMLP(eqx.Module):
    """Small residual MLP used for shared unit and graph updates."""

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
        for idx in range(n_layers):
            in_dim = input_dim if idx == 0 else hidden_dim
            self.layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[idx]))
        final_in_dim = hidden_dim if n_layers > 0 else input_dim
        self.output_layer = eqx.nn.Linear(final_in_dim, output_dim, key=keys[-1])

    def __call__(self, x: Array) -> Array:
        h = x
        for idx, layer in enumerate(self.layers):
            out = jax.nn.gelu(layer(h if idx > 0 else x))
            h = h + out if idx > 0 and h.shape == out.shape else out
        base = h if self.layers else x
        return self.output_layer(base)


class FlowsheetModel(eqx.Module):
    """Shared graph model over a small process flowsheet."""

    unit_backbone: ResidualMLP
    unit_delta_head: eqx.nn.Linear
    graph_update: ResidualMLP
    stream_message_proj: eqx.nn.Linear
    descriptor_proj: eqx.nn.Linear
    channel_context_proj: eqx.nn.Linear | None
    law_feature_proj: eqx.nn.Linear | None
    family_embedding_table: Array
    state_role_embedding_table: Array
    control_role_embedding_table: Array
    disturbance_role_embedding_table: Array
    channel_name_embedding_table: Array

    unit_state_center_table: Array
    unit_state_scale_table: Array
    unit_control_center_table: Array
    unit_control_scale_table: Array
    unit_disturbance_center_table: Array
    unit_disturbance_scale_table: Array
    unit_param_scale_table: Array
    unit_state_mask_table: Array
    unit_control_mask_table: Array
    unit_disturbance_mask_table: Array
    unit_param_mask_table: Array
    unit_descriptor_table: Array
    unit_family_id_table: Array
    unit_state_role_id_table: Array
    unit_control_role_id_table: Array
    unit_disturbance_role_id_table: Array
    unit_state_name_id_table: Array
    unit_control_name_id_table: Array
    unit_disturbance_name_id_table: Array
    unit_law_feature_table: Array
    stream_source_index_table: Array
    stream_target_index_table: Array
    stream_source_var_index_table: Array
    stream_target_var_index_table: Array
    stream_var_mask_table: Array
    stream_delay_table: Array

    n_units: int = eqx.field(static=True)
    n_streams: int = eqx.field(static=True)
    max_state_dim: int = eqx.field(static=True)
    max_control_dim: int = eqx.field(static=True)
    max_disturbance_dim: int = eqx.field(static=True)
    max_param_dim: int = eqx.field(static=True)
    max_stream_vars: int = eqx.field(static=True)
    n_global_controls: int = eqx.field(static=True)
    n_global_disturbances: int = eqx.field(static=True)
    descriptor_dim: int = eqx.field(static=True)
    family_embedding_dim: int = eqx.field(static=True)
    message_dim: int = eqx.field(static=True)
    message_passing_steps: int = eqx.field(static=True)
    channel_conditioning_enabled: bool = eqx.field(static=True)
    law_conditioning_enabled: bool = eqx.field(static=True)

    def __init__(
        self,
        metadata: FlowsheetGraphMetadata,
        *,
        hidden_dim: int,
        message_dim: int,
        family_embedding_dim: int,
        n_layers: int,
        graph_layers: int,
        message_passing_steps: int,
        channel_conditioning_enabled: bool,
        law_conditioning_enabled: bool,
        key: PRNGKeyArray,
    ):
        self.n_units = len(metadata.unit_names)
        self.n_streams = len(metadata.stream_names)
        self.max_state_dim = int(metadata.unit_state_center.shape[1])
        self.max_control_dim = int(metadata.unit_control_center.shape[1])
        self.max_disturbance_dim = int(metadata.unit_disturbance_center.shape[1])
        self.max_param_dim = int(metadata.unit_param_scale.shape[1])
        self.max_stream_vars = int(metadata.stream_var_mask.shape[1])
        self.n_global_controls = len(metadata.global_control_names)
        self.n_global_disturbances = len(metadata.global_disturbance_names)
        self.descriptor_dim = int(metadata.unit_descriptor.shape[1])
        self.family_embedding_dim = family_embedding_dim
        self.message_dim = message_dim
        self.message_passing_steps = max(int(message_passing_steps), 1)
        self.channel_conditioning_enabled = channel_conditioning_enabled
        self.law_conditioning_enabled = law_conditioning_enabled

        self.unit_state_center_table = metadata.unit_state_center
        self.unit_state_scale_table = metadata.unit_state_scale
        self.unit_control_center_table = metadata.unit_control_center
        self.unit_control_scale_table = metadata.unit_control_scale
        self.unit_disturbance_center_table = metadata.unit_disturbance_center
        self.unit_disturbance_scale_table = metadata.unit_disturbance_scale
        self.unit_param_scale_table = metadata.unit_param_scale
        self.unit_state_mask_table = metadata.unit_state_mask
        self.unit_control_mask_table = metadata.unit_control_mask
        self.unit_disturbance_mask_table = metadata.unit_disturbance_mask
        self.unit_param_mask_table = metadata.unit_param_mask
        self.unit_descriptor_table = metadata.unit_descriptor
        self.unit_family_id_table = metadata.unit_family_id
        self.unit_state_role_id_table = metadata.unit_state_role_id
        self.unit_control_role_id_table = metadata.unit_control_role_id
        self.unit_disturbance_role_id_table = metadata.unit_disturbance_role_id
        self.unit_state_name_id_table = metadata.unit_state_name_id
        self.unit_control_name_id_table = metadata.unit_control_name_id
        self.unit_disturbance_name_id_table = metadata.unit_disturbance_name_id
        self.unit_law_feature_table = metadata.unit_law_feature_defaults
        self.stream_source_index_table = metadata.stream_source_index
        self.stream_target_index_table = metadata.stream_target_index
        self.stream_source_var_index_table = metadata.stream_source_var_index
        self.stream_target_var_index_table = metadata.stream_target_var_index
        self.stream_var_mask_table = metadata.stream_var_mask
        self.stream_delay_table = metadata.stream_delay

        (
            key_family,
            key_desc,
            key_state_role,
            key_control_role,
            key_disturbance_role,
            key_channel_name,
            key_channel_context_proj,
            key_law_feature_proj,
            key_stream,
            key_backbone,
            key_delta,
            key_graph,
        ) = jax.random.split(key, 12)
        n_families = max(len(metadata.unit_family_names), 1)
        self.family_embedding_table = 0.02 * jax.random.normal(
            key_family,
            (n_families, family_embedding_dim),
        )
        self.state_role_embedding_table = 0.02 * jax.random.normal(
            key_state_role,
            (max(len(metadata.unit_state_role_names), 1), family_embedding_dim),
        )
        self.control_role_embedding_table = 0.02 * jax.random.normal(
            key_control_role,
            (max(len(metadata.unit_control_role_names), 1), family_embedding_dim),
        )
        self.disturbance_role_embedding_table = 0.02 * jax.random.normal(
            key_disturbance_role,
            (max(len(metadata.unit_disturbance_role_names), 1), family_embedding_dim),
        )
        self.channel_name_embedding_table = 0.02 * jax.random.normal(
            key_channel_name,
            (max(len(metadata.unit_channel_name_names), 1), family_embedding_dim),
        )
        self.descriptor_proj = eqx.nn.Linear(
            self.descriptor_dim,
            family_embedding_dim,
            key=key_desc,
        )
        if self.channel_conditioning_enabled:
            self.channel_context_proj = eqx.nn.Linear(
                family_embedding_dim,
                family_embedding_dim,
                key=key_channel_context_proj,
            )
        else:
            self.channel_context_proj = None
        if self.law_conditioning_enabled and int(self.unit_law_feature_table.shape[1]) > 0:
            self.law_feature_proj = eqx.nn.Linear(
                int(self.unit_law_feature_table.shape[1]),
                family_embedding_dim,
                key=key_law_feature_proj,
            )
        else:
            self.law_feature_proj = None
        self.stream_message_proj = eqx.nn.Linear(
            self.max_stream_vars,
            message_dim,
            key=key_stream,
        )

        global_feature_dim = self.n_global_controls + self.n_global_disturbances + 1
        unit_input_dim = (
            2 * self.max_state_dim
            + 2 * self.max_control_dim
            + 2 * self.max_disturbance_dim
            + 2 * self.max_param_dim
            + message_dim
            + family_embedding_dim
            + global_feature_dim
        )
        graph_input_dim = hidden_dim + message_dim + family_embedding_dim + global_feature_dim
        self.unit_backbone = ResidualMLP(
            unit_input_dim,
            hidden_dim,
            hidden_dim,
            n_layers,
            key=key_backbone,
        )
        self.unit_delta_head = eqx.nn.Linear(hidden_dim, self.max_state_dim, key=key_delta)
        self.graph_update = ResidualMLP(
            graph_input_dim,
            hidden_dim,
            self.max_state_dim,
            graph_layers,
            key=key_graph,
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        metadata: FlowsheetGraphMetadata,
        key: PRNGKeyArray,
    ) -> "FlowsheetModel":
        model_cfg = config.get("model", {})
        return cls(
            metadata,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            message_dim=int(model_cfg.get("message_dim", 16)),
            family_embedding_dim=int(model_cfg.get("family_embedding_dim", 16)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            graph_layers=int(model_cfg.get("graph_layers", 2)),
            message_passing_steps=int(model_cfg.get("message_passing_steps", 2)),
            channel_conditioning_enabled=bool(
                model_cfg.get("channel_conditioning", {}).get("enabled", False)
            ),
            law_conditioning_enabled=bool(
                model_cfg.get("law_conditioning", {}).get("enabled", False)
            ),
            key=key,
        )

    def normalize_states(self, states: Array) -> Array:
        return (states - self.unit_state_center_table) * self.unit_state_scale_table

    def normalize_controls(self, controls: Array) -> Array:
        return (controls - self.unit_control_center_table) * self.unit_control_scale_table

    def normalize_disturbances(self, disturbances: Array) -> Array:
        return (
            disturbances - self.unit_disturbance_center_table
        ) * self.unit_disturbance_scale_table

    def scale_params(self, params: Array) -> Array:
        return (
            jnp.sign(params) * jnp.log1p(jnp.abs(params)) * self.unit_param_scale_table
        )

    def _unit_contexts(self) -> Array:
        family_context = self.family_embedding_table[self.unit_family_id_table]
        descriptor_context = jax.vmap(
            lambda descriptor: jax.nn.gelu(self.descriptor_proj(descriptor))
        )(self.unit_descriptor_table)
        context = family_context + descriptor_context
        if self.channel_context_proj is not None:
            state_semantics = (
                self.state_role_embedding_table[self.unit_state_role_id_table]
                + self.channel_name_embedding_table[self.unit_state_name_id_table]
            )
            control_semantics = (
                self.control_role_embedding_table[self.unit_control_role_id_table]
                + self.channel_name_embedding_table[self.unit_control_name_id_table]
            )
            disturbance_semantics = (
                self.disturbance_role_embedding_table[self.unit_disturbance_role_id_table]
                + self.channel_name_embedding_table[self.unit_disturbance_name_id_table]
            )
            state_summary = jax.vmap(
                lambda emb, mask: jnp.sum(emb * mask[:, None], axis=0)
                / jnp.maximum(jnp.sum(mask), 1.0)
            )(state_semantics, self.unit_state_mask_table)
            control_summary = jax.vmap(
                lambda emb, mask: jnp.sum(emb * mask[:, None], axis=0)
                / jnp.maximum(jnp.sum(mask), 1.0)
            )(control_semantics, self.unit_control_mask_table)
            disturbance_summary = jax.vmap(
                lambda emb, mask: jnp.sum(emb * mask[:, None], axis=0)
                / jnp.maximum(jnp.sum(mask), 1.0)
            )(disturbance_semantics, self.unit_disturbance_mask_table)
            channel_summary = (state_summary + control_summary + disturbance_summary) / 3.0
            context = context + jax.vmap(
                lambda summary: jax.nn.gelu(self.channel_context_proj(summary))
            )(channel_summary)
        if self.law_feature_proj is not None:
            context = context + jax.vmap(
                lambda features: jax.nn.gelu(self.law_feature_proj(features))
            )(self.unit_law_feature_table)
        return context

    def _compute_stream_values(
        self,
        states: Array,
        external_stream_values: Array | None,
    ) -> Array:
        safe_source_index = jnp.maximum(self.stream_source_index_table, 0)
        safe_source_var_index = jnp.maximum(self.stream_source_var_index_table, 0)
        source_states = states[safe_source_index]
        internal_values = jnp.take_along_axis(
            source_states,
            safe_source_var_index,
            axis=-1,
        ) * self.stream_var_mask_table
        if external_stream_values is None:
            external_stream_values = jnp.zeros_like(internal_values)
        external_mask = (self.stream_source_index_table < 0).astype(internal_values.dtype)[:, None]
        return internal_values * (1.0 - external_mask) + external_stream_values * external_mask

    def _compose_stream_values(
        self,
        raw_stream_values: Array,
        external_stream_values: Array | None,
        delayed_stream_values: Array | None,
    ) -> Array:
        if external_stream_values is None:
            external_stream_values = jnp.zeros_like(raw_stream_values)
        if delayed_stream_values is None:
            delayed_stream_values = raw_stream_values

        external_mask = (self.stream_source_index_table < 0).astype(raw_stream_values.dtype)[:, None]
        delayed_mask = (
            (self.stream_source_index_table >= 0)
            & (self.stream_delay_table > 0.0)
        ).astype(raw_stream_values.dtype)[:, None]
        direct_internal_mask = 1.0 - external_mask - delayed_mask
        return (
            external_stream_values * external_mask
            + delayed_stream_values * delayed_mask
            + raw_stream_values * direct_internal_mask
        )

    def _aggregate_incoming_messages(self, stream_values: Array) -> Array:
        projected = jax.vmap(self.stream_message_proj)(stream_values)
        safe_target_index = jnp.maximum(self.stream_target_index_table, 0)
        internal_target_mask = (self.stream_target_index_table >= 0).astype(projected.dtype)[:, None]
        messages = jnp.zeros((self.n_units, self.message_dim), dtype=projected.dtype)
        return messages.at[safe_target_index].add(projected * internal_target_mask)

    def step(
        self,
        states: Array,
        controls: Array,
        disturbances: Array,
        params: Array,
        dt: Array,
        *,
        external_stream_values: Array | None = None,
        delayed_stream_values: Array | None = None,
        global_controls: Array | None = None,
        global_disturbances: Array | None = None,
    ) -> tuple[Array, Array]:
        global_controls = (
            global_controls
            if global_controls is not None
            else jnp.zeros((self.n_global_controls,), dtype=states.dtype)
        )
        global_disturbances = (
            global_disturbances
            if global_disturbances is not None
            else jnp.zeros((self.n_global_disturbances,), dtype=states.dtype)
        )

        state_mask = self.unit_state_mask_table
        control_mask = self.unit_control_mask_table
        disturbance_mask = self.unit_disturbance_mask_table
        param_mask = self.unit_param_mask_table
        contexts = self._unit_contexts()
        global_features = jnp.concatenate(
            [
                global_controls,
                global_disturbances,
                jnp.asarray([dt], dtype=states.dtype),
            ]
        )

        control_norm = self.normalize_controls(controls) * control_mask
        disturbance_norm = self.normalize_disturbances(disturbances) * disturbance_mask
        params_scaled = self.scale_params(params) * param_mask

        def one_iteration(carry):
            current_states, current_streams = carry
            state_norm = self.normalize_states(current_states) * state_mask
            incoming_messages = self._aggregate_incoming_messages(current_streams)

            def update_one(
                state_norm_t: Array,
                control_norm_t: Array,
                disturbance_norm_t: Array,
                params_scaled_t: Array,
                message_t: Array,
                context_t: Array,
                state_mask_t: Array,
                control_mask_t: Array,
                disturbance_mask_t: Array,
                param_mask_t: Array,
                center_t: Array,
                scale_t: Array,
            ) -> Array:
                features = jnp.concatenate(
                    [
                        state_norm_t * state_mask_t,
                        state_mask_t,
                        control_norm_t * control_mask_t,
                        control_mask_t,
                        disturbance_norm_t * disturbance_mask_t,
                        disturbance_mask_t,
                        params_scaled_t * param_mask_t,
                        param_mask_t,
                        message_t,
                        context_t,
                        global_features,
                    ]
                )
                hidden = self.unit_backbone(features)
                base_delta = 0.1 * jnp.tanh(self.unit_delta_head(hidden))
                graph_delta = 0.05 * self.graph_update(
                    jnp.concatenate([hidden, message_t, context_t, global_features])
                )
                next_state_norm = state_norm_t + (base_delta + graph_delta) * state_mask_t
                next_state = next_state_norm / jnp.maximum(scale_t, 1e-6) + center_t
                return next_state * state_mask_t

            next_states = jax.vmap(update_one)(
                state_norm,
                control_norm,
                disturbance_norm,
                params_scaled,
                incoming_messages,
                contexts,
                state_mask,
                control_mask,
                disturbance_mask,
                param_mask,
                self.unit_state_center_table,
                self.unit_state_scale_table,
            )
            raw_next_streams = self._compute_stream_values(next_states, external_stream_values)
            next_streams = self._compose_stream_values(
                raw_next_streams,
                external_stream_values,
                delayed_stream_values,
            )
            return next_states, next_streams

        stream_values = self._compose_stream_values(
            self._compute_stream_values(states, external_stream_values),
            external_stream_values,
            delayed_stream_values,
        )
        carry = (states, stream_values)
        for _ in range(self.message_passing_steps):
            carry = one_iteration(carry)
        return carry

    def rollout(
        self,
        initial_states: Array,
        controls: Array,
        disturbances: Array,
        params: Array,
        ts: Array,
        *,
        global_controls: Array | None = None,
        global_disturbances: Array | None = None,
        external_stream_sequence: Array | None = None,
    ) -> tuple[Array, Array]:
        n_steps = int(ts.shape[0])
        if global_controls is None:
            global_controls = jnp.zeros(
                (n_steps, self.n_global_controls),
                dtype=initial_states.dtype,
            )
        if global_disturbances is None:
            global_disturbances = jnp.zeros(
                (n_steps, self.n_global_disturbances),
                dtype=initial_states.dtype,
            )
        if external_stream_sequence is None:
            external_stream_sequence = jnp.zeros(
                (n_steps, self.n_streams, self.max_stream_vars),
                dtype=initial_states.dtype,
            )

        initial_raw_streams = self._compute_stream_values(
            initial_states,
            external_stream_sequence[0],
        )
        initial_streams = self._compose_stream_values(
            initial_raw_streams,
            external_stream_sequence[0],
            initial_raw_streams,
        )

        raw_history = jnp.zeros(
            (n_steps, self.n_streams, self.max_stream_vars),
            dtype=initial_states.dtype,
        )
        raw_history = raw_history.at[0].set(initial_raw_streams)

        def select_delayed_streams(history: Array, history_index: Array, dt: Array) -> Array:
            safe_dt = jnp.maximum(jnp.abs(dt), 1e-6)
            lag_steps = jnp.maximum(
                jnp.rint(self.stream_delay_table / safe_dt).astype(jnp.int32),
                0,
            )
            delay_indices = jnp.maximum(history_index - lag_steps + 1, 0)
            return history[delay_indices, jnp.arange(self.n_streams)]

        def step_fn(carry, step_idx):
            current_states, history, history_index = carry
            dt = ts[step_idx + 1] - ts[step_idx]
            delayed_stream_values = select_delayed_streams(history, history_index, dt)
            next_states, next_streams = self.step(
                current_states,
                controls[step_idx],
                disturbances[step_idx],
                params,
                dt,
                external_stream_values=external_stream_sequence[step_idx + 1],
                delayed_stream_values=delayed_stream_values,
                global_controls=global_controls[step_idx],
                global_disturbances=global_disturbances[step_idx],
            )
            next_raw_streams = self._compute_stream_values(
                next_states,
                external_stream_sequence[step_idx + 1],
            )
            next_history = history.at[history_index + 1].set(next_raw_streams)
            return (next_states, next_history, history_index + 1), (next_states, next_streams)

        _, (state_hist, stream_hist) = jax.lax.scan(
            step_fn,
            (initial_states, raw_history, jnp.asarray(0, dtype=jnp.int32)),
            jnp.arange(n_steps - 1),
        )
        full_state_hist = jnp.concatenate([initial_states[None, ...], state_hist], axis=0)
        full_stream_hist = jnp.concatenate([initial_streams[None, ...], stream_hist], axis=0)
        return full_state_hist, full_stream_hist

    def get_parameter_count(self) -> Dict[str, int]:
        def count(module) -> int:
            return sum(x.size for x in jax.tree.leaves(eqx.filter(module, eqx.is_inexact_array)))

        return {
            "family_embeddings": count(self.family_embedding_table),
            "unit_backbone": count(self.unit_backbone),
            "unit_delta_head": count(self.unit_delta_head),
            "graph_update": count(self.graph_update),
            "stream_message_proj": count(self.stream_message_proj),
            "descriptor_proj": count(self.descriptor_proj),
            "total": count(self),
        }
