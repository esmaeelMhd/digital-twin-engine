"""System registry: maps system names to SystemSpec and ProcessSimulator instances."""

from typing import Dict, Optional

from dte.simulators.base import (
    DecoderConstraint,
    NormalizationSpec,
    ProcessSimulator,
    SystemSpec,
)


def _build_cstr_spec(system_config: dict) -> SystemSpec:
    """Build a SystemSpec for the CSTR from a parsed system YAML config."""
    sys = system_config.get("system", {})
    ic = system_config.get("initial_conditions", {})
    ops = system_config.get("operating_ranges", {})

    state_names = sys.get("state_names", ["Ca", "Cb", "T", "Tc"])
    control_names = sys.get("control_names", ["F_in", "Tc_in"])
    disturbance_names = sys.get("disturbance_names", ["Ca_in", "T_in"])

    norm_cfg = sys.get("normalization", {})
    normalization = NormalizationSpec(
        state_center=norm_cfg.get("state_center", [1.0, 1.0, 325.0, 325.0]),
        state_scale=norm_cfg.get("state_scale", [1.0, 1.0, 0.01, 0.01]),
        control_center=norm_cfg.get("control_center", [55.0, 300.0]),
        control_scale=norm_cfg.get("control_scale", [0.02, 0.05]),
        disturbance_center=norm_cfg.get("disturbance_center", [1.0, 320.0]),
        disturbance_scale=norm_cfg.get("disturbance_scale", [1.0, 0.02]),
        param_scale=norm_cfg.get("param_scale", 0.1),
    )

    raw_constraints = sys.get("decoder_constraints", [
        {"type": "softplus", "indices": [0, 1], "bias": 0.5},
        {"type": "sigmoid_range", "indices": [2, 3], "low": 250.0, "high": 400.0},
    ])
    decoder_constraints = [
        DecoderConstraint(
            type=c["type"],
            indices=c["indices"],
            bias=c.get("bias", 0.5),
            low=c.get("low", 0.0),
            high=c.get("high", 1.0),
        )
        for c in raw_constraints
    ]

    default_initial_state = [
        ic.get("Ca", 0.5),
        ic.get("Cb", 0.5),
        ic.get("T", 350.0),
        ic.get("Tc", 300.0),
    ]

    # Nominal disturbances: midpoints of operating ranges
    ca_in_range = ops.get("Ca_in", [0.5, 2.0])
    t_in_range = ops.get("T_in", [290.0, 350.0])
    default_nominal_disturbance = [
        (ca_in_range[0] + ca_in_range[1]) / 2.0,
        (t_in_range[0] + t_in_range[1]) / 2.0,
    ]

    return SystemSpec(
        name="cstr",
        state_dim=4,
        control_dim=2,
        disturbance_dim=2,
        param_dim=6,
        state_names=state_names,
        control_names=control_names,
        disturbance_names=disturbance_names,
        decoder_constraints=decoder_constraints,
        normalization=normalization,
        default_initial_state=default_initial_state,
        default_nominal_disturbance=default_nominal_disturbance,
        control_ranges={k: list(v) for k, v in ops.items() if k in control_names},
        disturbance_ranges={k: list(v) for k, v in ops.items() if k in disturbance_names},
    )


def _build_heat_exchanger_spec(system_config: dict) -> SystemSpec:
    """Build a SystemSpec for the counter-current heat exchanger."""
    sys = system_config.get("system", {})
    ic = system_config.get("initial_conditions", {})
    ops = system_config.get("operating_ranges", {})

    state_names = sys.get("state_names", ["T_hot", "T_cold"])
    control_names = sys.get("control_names", ["F_hot", "F_cold"])
    disturbance_names = sys.get("disturbance_names", ["T_hot_in", "T_cold_in"])

    norm_cfg = sys.get("normalization", {})
    normalization = NormalizationSpec(
        state_center=norm_cfg.get("state_center", [350.0, 300.0]),
        state_scale=norm_cfg.get("state_scale", [0.01, 0.01]),
        control_center=norm_cfg.get("control_center", [5.0, 5.0]),
        control_scale=norm_cfg.get("control_scale", [0.1, 0.1]),
        disturbance_center=norm_cfg.get("disturbance_center", [380.0, 280.0]),
        disturbance_scale=norm_cfg.get("disturbance_scale", [0.01, 0.01]),
        param_scale=norm_cfg.get("param_scale", 0.1),
    )

    raw_constraints = sys.get("decoder_constraints", [
        {"type": "sigmoid_range", "indices": [0, 1], "low": 250.0, "high": 450.0},
    ])
    decoder_constraints = [
        DecoderConstraint(
            type=c["type"],
            indices=c["indices"],
            bias=c.get("bias", 0.5),
            low=c.get("low", 0.0),
            high=c.get("high", 1.0),
        )
        for c in raw_constraints
    ]

    default_initial_state = [
        ic.get("T_hot", 350.0),
        ic.get("T_cold", 300.0),
    ]

    t_hot_in_range = ops.get("T_hot_in", [360.0, 420.0])
    t_cold_in_range = ops.get("T_cold_in", [270.0, 310.0])
    default_nominal_disturbance = [
        (t_hot_in_range[0] + t_hot_in_range[1]) / 2.0,
        (t_cold_in_range[0] + t_cold_in_range[1]) / 2.0,
    ]

    return SystemSpec(
        name="heat_exchanger",
        state_dim=2,
        control_dim=2,
        disturbance_dim=2,
        param_dim=5,
        state_names=state_names,
        control_names=control_names,
        disturbance_names=disturbance_names,
        decoder_constraints=decoder_constraints,
        normalization=normalization,
        default_initial_state=default_initial_state,
        default_nominal_disturbance=default_nominal_disturbance,
        control_ranges={k: list(v) for k, v in ops.items() if k in control_names},
        disturbance_ranges={k: list(v) for k, v in ops.items() if k in disturbance_names},
    )


# Registry mapping system name -> builder function
_SPEC_BUILDERS = {
    "cstr": _build_cstr_spec,
    "heat_exchanger": _build_heat_exchanger_spec,
}


def get_system_spec(system_config: dict) -> SystemSpec:
    """Build a SystemSpec from a parsed system YAML config.

    The config must contain a ``system.name`` key that identifies the system.
    Falls back to ``"cstr"`` for backwards compatibility when no name is set.
    """
    system_name = system_config.get("system", {}).get("name", "cstr")
    if system_name not in _SPEC_BUILDERS:
        raise ValueError(
            f"Unknown system '{system_name}'. "
            f"Available systems: {list(_SPEC_BUILDERS.keys())}"
        )
    return _SPEC_BUILDERS[system_name](system_config)


def get_simulator(system_name: str, system_config: dict) -> ProcessSimulator:
    """Instantiate and return the ProcessSimulator for ``system_name``."""
    if system_name == "cstr":
        from dte.simulators.cstr import CSTRSimulator, CSTRParams

        cstr_cfg = system_config.get("cstr", {})
        params = CSTRParams(**{k: float(v) for k, v in cstr_cfg.items()})
        return CSTRSimulator(params)

    if system_name == "heat_exchanger":
        from dte.simulators.heat_exchanger import HeatExchangerSimulator, HeatExchangerParams

        hx_cfg = system_config.get("heat_exchanger", {})
        params = HeatExchangerParams(**{k: float(v) for k, v in hx_cfg.items()})
        return HeatExchangerSimulator(params)

    raise ValueError(
        f"Unknown system '{system_name}'. "
        f"Available systems: {list(_SPEC_BUILDERS.keys())}"
    )


def list_systems() -> list:
    """Return the names of all registered systems."""
    return list(_SPEC_BUILDERS.keys())
