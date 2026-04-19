"""Tests for the grouped universal digital twin path."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import equinox as eqx

from dte.data.datasets.universal_unit_dataset import UniversalSystemMetadata
from dte.models.universal.digital_twin import UniversalDigitalTwin


def _build_metadata() -> UniversalSystemMetadata:
    return UniversalSystemMetadata(
        system_names=("cstr", "heat_exchanger"),
        state_center=jnp.zeros((2, 4), dtype=jnp.float32),
        state_scale=jnp.ones((2, 4), dtype=jnp.float32),
        control_center=jnp.zeros((2, 2), dtype=jnp.float32),
        control_scale=jnp.ones((2, 2), dtype=jnp.float32),
        disturbance_center=jnp.zeros((2, 2), dtype=jnp.float32),
        disturbance_scale=jnp.ones((2, 2), dtype=jnp.float32),
        param_scale=jnp.ones((2, 6), dtype=jnp.float32),
        state_mask=jnp.asarray([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=jnp.float32),
        control_mask=jnp.ones((2, 2), dtype=jnp.float32),
        disturbance_mask=jnp.ones((2, 2), dtype=jnp.float32),
        param_mask=jnp.asarray([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=jnp.float32),
        state_dim=jnp.asarray([4, 2], dtype=jnp.int32),
        control_dim=jnp.asarray([2, 2], dtype=jnp.int32),
        disturbance_dim=jnp.asarray([2, 2], dtype=jnp.int32),
        param_dim=jnp.asarray([6, 5], dtype=jnp.int32),
        system_descriptor=jnp.zeros((2, 25), dtype=jnp.float32),
        state_group_kind_names=("concentration", "thermal", "inventory"),
        state_group_mask=jnp.asarray(
            [
                [[1, 1, 0, 0], [0, 0, 1, 1]],
                [[1, 1, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=jnp.float32,
        ),
        state_group_active=jnp.asarray([[1, 1], [1, 0]], dtype=jnp.float32),
        state_group_kind_id=jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32),
        state_role_names=("concentration", "temperature", "inventory"),
        state_role_id=jnp.asarray(
            [
                [0, 0, 1, 1],
                [1, 1, 2, 2],
            ],
            dtype=jnp.int32,
        ),
        state_lower_bound=jnp.asarray(
            [
                [0.0, 0.0, 250.0, 250.0],
                [250.0, 250.0, -jnp.inf, -jnp.inf],
            ],
            dtype=jnp.float32,
        ),
        state_upper_bound=jnp.asarray(
            [
                [jnp.inf, jnp.inf, 400.0, 400.0],
                [450.0, 450.0, jnp.inf, jnp.inf],
            ],
            dtype=jnp.float32,
        ),
        family_names=("reactor", "thermal"),
        family_id=jnp.asarray([0, 1], dtype=jnp.int32),
        subtype_names=("nonisothermal_cstr", "counter_current"),
        subtype_id=jnp.asarray([0, 1], dtype=jnp.int32),
        law_tag_names=("mass_balance", "energy_balance", "heat_transfer", "generic"),
        law_tag_mask=jnp.asarray(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        conditioning_category_names=(
            "reaction_class",
            "thermo_regime",
            "bio_model_family",
            "operating_regime",
        ),
        conditioning_value_names=(
            "unknown",
            "irreversible_exothermic",
            "single_phase_countercurrent",
            "liquid_well_mixed",
            "nominal_continuous",
            "nominal_heat_exchange",
            "none",
        ),
        conditioning_value_id=jnp.asarray(
            [
                [1, 3, 6, 4],
                [6, 2, 6, 5],
            ],
            dtype=jnp.int32,
        ),
        parameter_law_tag_id=jnp.asarray(
            [
                [0, 0, 1, 2, 1, 3],
                [1, 2, 2, 3, 3, 3],
            ],
            dtype=jnp.int32,
        ),
    )


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
        }
    }


def _count_trainable(model, filter_spec) -> int:
    trainable, _ = eqx.partition(model, filter_spec)
    return sum(
        leaf.size for leaf in jax.tree.leaves(eqx.filter(trainable, eqx.is_inexact_array))
    )


def test_universal_model_grouped_encode_decode_shapes():
    metadata = _build_metadata()
    model = UniversalDigitalTwin.from_config(_build_config(), metadata, jax.random.PRNGKey(0))

    state_norm = jnp.array([0.5, -0.2, 0.1, 0.7], dtype=jnp.float32)
    params_scaled = jnp.ones((6,), dtype=jnp.float32)
    control_norm = jnp.array([0.2, -0.4], dtype=jnp.float32)
    state_mask = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    control_mask = jnp.array([1.0, 1.0], dtype=jnp.float32)
    param_mask = jnp.ones((6,), dtype=jnp.float32)

    z, z_mean, z_logvar = model.encode(
        state_norm,
        params_scaled,
        control_norm,
        state_mask,
        control_mask,
        param_mask,
        jnp.asarray(0, dtype=jnp.int32),
        jax.random.PRNGKey(1),
    )
    decoded = model.decode(
        z,
        params_scaled,
        control_norm,
        state_mask,
        control_mask,
        param_mask,
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert z.shape == (16,)
    assert z_mean.shape == (16,)
    assert z_logvar.shape == (16,)
    assert decoded.shape == (4,)


def test_universal_model_respects_inactive_state_dimensions():
    metadata = _build_metadata()
    model = UniversalDigitalTwin.from_config(_build_config(), metadata, jax.random.PRNGKey(2))

    state_norm = jnp.array([0.3, -0.1, 0.0, 0.0], dtype=jnp.float32)
    params_scaled = jnp.array([1.0, 0.5, -0.2, 0.1, 0.0, 0.0], dtype=jnp.float32)
    control_norm = jnp.array([0.2, 0.4], dtype=jnp.float32)
    state_mask = jnp.array([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    control_mask = jnp.array([1.0, 1.0], dtype=jnp.float32)
    param_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    system_id = jnp.asarray(1, dtype=jnp.int32)

    z, _, _ = model.encode(
        state_norm,
        params_scaled,
        control_norm,
        state_mask,
        control_mask,
        param_mask,
        system_id,
        None,
    )
    decoded = model.decode(
        z,
        params_scaled,
        control_norm,
        state_mask,
        control_mask,
        param_mask,
        system_id,
    )

    assert decoded.shape == (4,)
    assert jnp.allclose(decoded[2:], 0.0)


def test_universal_model_supports_channel_and_law_conditioning_tables():
    metadata = replace(
        _build_metadata(),
        control_role_names=("flow", "temperature"),
        control_role_id=jnp.asarray([[0, 1], [0, 0]], dtype=jnp.int32),
        disturbance_role_names=("concentration", "temperature"),
        disturbance_role_id=jnp.asarray([[0, 1], [1, 1]], dtype=jnp.int32),
        channel_name_names=("generic", "Ca", "Cb", "T", "Tc", "F_in", "Tc_in", "Ca_in", "T_in"),
        state_name_id=jnp.asarray([[1, 2, 3, 4], [3, 4, 0, 0]], dtype=jnp.int32),
        control_name_id=jnp.asarray([[5, 6], [5, 5]], dtype=jnp.int32),
        disturbance_name_id=jnp.asarray([[7, 8], [8, 8]], dtype=jnp.int32),
        law_feature_names=("law_tag::mass_balance", "law_tag::energy_balance"),
        law_feature_defaults=jnp.asarray(
            [[1.0, 1.0], [0.0, 1.0]],
            dtype=jnp.float32,
        ),
    )
    config = _build_config()
    config["model"]["channel_conditioning"] = {"enabled": True}
    config["model"]["law_conditioning"] = {"enabled": True}
    model = UniversalDigitalTwin.from_config(config, metadata, jax.random.PRNGKey(4))

    z, _, _ = model.encode(
        jnp.array([0.2, -0.1, 0.1, 0.3], dtype=jnp.float32),
        jnp.ones((6,), dtype=jnp.float32),
        jnp.array([0.1, -0.2], dtype=jnp.float32),
        jnp.ones((4,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((6,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jax.random.PRNGKey(5),
    )

    assert z.shape == (16,)


def test_universal_model_supports_stiff_aware_latent_solver_rollout():
    metadata = _build_metadata()
    config = _build_config()
    config["model"]["latent_solver"] = {
        "method": "kvaerno5",
        "rtol": 1e-3,
        "atol": 1e-4,
        "dt0_factor": 0.5,
        "max_steps": 256,
    }
    model = UniversalDigitalTwin.from_config(config, metadata, jax.random.PRNGKey(6))

    z0 = jnp.zeros((16,), dtype=jnp.float32)
    ts = jnp.asarray([0.0, 0.05, 0.1], dtype=jnp.float32)
    controls = jnp.asarray(
        [[0.0, 0.0], [0.1, -0.1], [0.15, -0.05]],
        dtype=jnp.float32,
    )
    disturbances = jnp.asarray(
        [[0.0, 0.0], [0.05, 0.0], [0.05, 0.02]],
        dtype=jnp.float32,
    )
    params_scaled = jnp.ones((6,), dtype=jnp.float32)
    control_mask = jnp.ones((2,), dtype=jnp.float32)
    disturbance_mask = jnp.ones((2,), dtype=jnp.float32)
    param_mask = jnp.ones((6,), dtype=jnp.float32)
    system_id = jnp.asarray(0, dtype=jnp.int32)

    z_traj = model.rollout_latent(
        ts,
        z0,
        controls,
        disturbances,
        params_scaled,
        control_mask,
        disturbance_mask,
        param_mask,
        system_id,
    )

    assert model.latent_solver_method == "kvaerno5"
    assert z_traj.shape == (3, 16)
    assert jnp.all(jnp.isfinite(z_traj))

    z_next = model.latent_step(
        z0,
        controls[0],
        controls[1],
        disturbances[0],
        disturbances[1],
        params_scaled,
        control_mask,
        disturbance_mask,
        param_mask,
        system_id,
        ts[1] - ts[0],
    )

    assert z_next.shape == (16,)
    assert jnp.all(jnp.isfinite(z_next))


def test_universal_model_adapter_filter_is_smaller_than_full_filter():
    metadata = _build_metadata()
    model = UniversalDigitalTwin.from_config(_build_config(), metadata, jax.random.PRNGKey(3))

    full_count = _count_trainable(model, model.trainable_filter_spec(mode="full"))
    adapter_count = _count_trainable(model, model.trainable_filter_spec(mode="adapters"))

    assert adapter_count > 0
    assert adapter_count < full_count


def test_universal_model_param_calibration_changes_scaled_params():
    metadata = _build_metadata()
    model = UniversalDigitalTwin.from_config(_build_config(), metadata, jax.random.PRNGKey(4))
    model = eqx.tree_at(
        lambda m: m.param_bias_table,
        model,
        model.param_bias_table.at[0, 1].set(0.5),
    )
    params = jnp.asarray([1.0, 2.0, 0.5, 0.0, 0.0, 0.0], dtype=jnp.float32)

    scaled_without_bias = UniversalDigitalTwin.from_config(
        _build_config(),
        metadata,
        jax.random.PRNGKey(5),
    ).scale_params(params, jnp.asarray(0, dtype=jnp.int32))
    scaled_with_bias = model.scale_params(params, jnp.asarray(0, dtype=jnp.int32))

    assert scaled_with_bias.shape == scaled_without_bias.shape
    assert scaled_with_bias[1] > scaled_without_bias[1]
