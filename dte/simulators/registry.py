"""System registry: maps system names to SystemSpec and ProcessSimulator instances."""

from typing import Callable, Dict, Optional

from dte.core.state_schema import ParameterDescriptor, SignalChannel, StateChannel, TopologyPort
from dte.simulators.base import (
    DecoderConstraint,
    NormalizationSpec,
    ProcessUnitSpec,
    ProcessSimulator,
    StateGroupSpec,
    SystemSpec,
)


def _parse_state_groups(sys: dict, state_dim: int) -> list[StateGroupSpec]:
    raw_groups = sys.get("state_groups")
    if not raw_groups:
        return [
            StateGroupSpec(
                name="all_states",
                kind="generic",
                indices=list(range(state_dim)),
            )
        ]
    return [
        StateGroupSpec(
            name=str(group["name"]),
            kind=str(group.get("kind", "generic")),
            indices=[int(idx) for idx in group.get("indices", [])],
        )
        for group in raw_groups
    ]


def _parse_state_channels(
    sys: dict,
    state_names: list[str],
    state_groups: list[StateGroupSpec],
    decoder_constraints: list[DecoderConstraint],
) -> list[StateChannel]:
    raw_channels = sys.get("state_channels")
    if not raw_channels:
        return []
    if len(raw_channels) != len(state_names):
        raise ValueError(
            "system.state_channels must match the length of system.state_names."
        )

    channels: list[StateChannel] = []
    for idx, (expected_name, channel_cfg) in enumerate(zip(state_names, raw_channels)):
        channel_name = str(channel_cfg.get("name", expected_name))
        if channel_name != expected_name:
            raise ValueError(
                f"system.state_channels[{idx}] name '{channel_name}' does not match "
                f"state_names[{idx}]='{expected_name}'."
            )
        channels.append(
            StateChannel(
                name=channel_name,
                role=str(channel_cfg.get("role", "generic")),
                unit=channel_cfg.get("unit"),
                lower_bound=channel_cfg.get("lower_bound"),
                upper_bound=channel_cfg.get("upper_bound"),
                conserved_group=channel_cfg.get("conserved_group"),
                description=channel_cfg.get("description"),
            )
        )
    return channels


def _parse_signal_channels(
    channel_cfg: list[dict] | None,
    expected_names: list[str],
    label: str,
) -> list[SignalChannel]:
    if not channel_cfg:
        return []
    if len(channel_cfg) != len(expected_names):
        raise ValueError(f"system.{label} must match the configured dimension.")

    channels: list[SignalChannel] = []
    for idx, (expected_name, item) in enumerate(zip(expected_names, channel_cfg)):
        channel_name = str(item.get("name", expected_name))
        if channel_name != expected_name:
            raise ValueError(
                f"system.{label}[{idx}] name '{channel_name}' does not match "
                f"expected '{expected_name}'."
            )
        channels.append(
            SignalChannel(
                name=channel_name,
                role=str(item.get("role", "generic")),
                unit=item.get("unit"),
                lower_bound=item.get("lower_bound"),
                upper_bound=item.get("upper_bound"),
                description=item.get("description"),
            )
        )
    return channels


def _parse_parameter_descriptors(
    sys: dict,
    default_names: list[str],
) -> list[ParameterDescriptor]:
    raw = sys.get("parameter_descriptors")
    if not raw:
        return [
            ParameterDescriptor(name=name)
            for name in default_names
        ]
    if len(raw) != len(default_names):
        raise ValueError("system.parameter_descriptors must match param_dim.")
    return [
        ParameterDescriptor(
            name=str(item.get("name", default_name)),
            unit=item.get("unit"),
            default=item.get("default"),
            law_tag=item.get("law_tag"),
            description=item.get("description"),
        )
        for default_name, item in zip(default_names, raw)
    ]


def _parse_topology_ports(sys: dict) -> list[TopologyPort]:
    raw = sys.get("topology_ports")
    if not raw:
        return []
    return [
        TopologyPort(
            name=str(item["name"]),
            kind=str(item.get("kind", "generic")),
            direction=str(item.get("direction", "bidirectional")),
            description=item.get("description"),
        )
        for item in raw
    ]


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

    state_groups = _parse_state_groups(sys, state_dim=4)
    return ProcessUnitSpec(
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
        state_groups=state_groups,
        state_channels=_parse_state_channels(
            sys,
            state_names,
            state_groups,
            decoder_constraints,
        ),
        control_channels=_parse_signal_channels(
            sys.get("control_channels"),
            control_names,
            "control_channels",
        ),
        disturbance_channels=_parse_signal_channels(
            sys.get("disturbance_channels"),
            disturbance_names,
            "disturbance_channels",
        ),
        parameter_descriptors=_parse_parameter_descriptors(
            sys,
            ["V", "Ea_over_R", "dH_rxn", "UA", "Fc", "Cp"],
        ),
        unit_type=str(sys.get("unit_type", "stirred_tank_reactor")),
        family=str(sys.get("family", "reactor")),
        subtype=sys.get("subtype", "nonisothermal_cstr"),
        law_tags=[str(tag) for tag in sys.get("law_tags", ["mass_balance", "energy_balance", "reaction_kinetics"])],
        topology_ports=_parse_topology_ports(sys),
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
        disturbance_center=norm_cfg.get("disturbance_center", [390.0, 290.0]),
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

    state_groups = _parse_state_groups(sys, state_dim=2)
    return ProcessUnitSpec(
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
        state_groups=state_groups,
        state_channels=_parse_state_channels(
            sys,
            state_names,
            state_groups,
            decoder_constraints,
        ),
        control_channels=_parse_signal_channels(
            sys.get("control_channels"),
            control_names,
            "control_channels",
        ),
        disturbance_channels=_parse_signal_channels(
            sys.get("disturbance_channels"),
            disturbance_names,
            "disturbance_channels",
        ),
        parameter_descriptors=_parse_parameter_descriptors(
            sys,
            ["V_hot", "V_cold", "UA", "rho", "Cp"],
        ),
        unit_type=str(sys.get("unit_type", "heat_exchanger")),
        family=str(sys.get("family", "thermal")),
        subtype=sys.get("subtype", "counter_current"),
        law_tags=[
            str(tag)
            for tag in sys.get("law_tags", ["energy_balance", "heat_transfer"])
        ],
        topology_ports=_parse_topology_ports(sys),
    )


def _build_two_tank_spec(system_config: dict) -> SystemSpec:
    """Build a SystemSpec for the coupled two-tank level process."""
    sys = system_config.get("system", {})
    ic = system_config.get("initial_conditions", {})
    ops = system_config.get("operating_ranges", {})

    state_names = sys.get("state_names", ["h1", "h2"])
    control_names = sys.get("control_names", ["q_in", "valve"])
    disturbance_names = sys.get("disturbance_names", ["d1", "d2"])

    norm_cfg = sys.get("normalization", {})
    normalization = NormalizationSpec(
        state_center=norm_cfg.get("state_center", [1.5, 1.0]),
        state_scale=norm_cfg.get("state_scale", [0.5, 0.5]),
        control_center=norm_cfg.get("control_center", [0.75, 0.825]),
        control_scale=norm_cfg.get("control_scale", [1.0, 1.5]),
        disturbance_center=norm_cfg.get("disturbance_center", [0.075, 0.05]),
        disturbance_scale=norm_cfg.get("disturbance_scale", [2.0, 4.0]),
        param_scale=norm_cfg.get("param_scale", 0.1),
    )

    raw_constraints = sys.get("decoder_constraints", [
        {"type": "sigmoid_range", "indices": [0, 1], "low": 0.0, "high": 5.0},
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
        ic.get("h1", 1.8),
        ic.get("h2", 1.1),
    ]
    d1_range = ops.get("d1", [0.0, 0.4])
    d2_range = ops.get("d2", [0.0, 0.3])
    default_nominal_disturbance = [
        0.5 * (d1_range[0] + d1_range[1]),
        0.5 * (d2_range[0] + d2_range[1]),
    ]

    state_groups = _parse_state_groups(sys, state_dim=2)
    return ProcessUnitSpec(
        name="two_tank",
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
        state_groups=state_groups,
        state_channels=_parse_state_channels(
            sys,
            state_names,
            state_groups,
            decoder_constraints,
        ),
        control_channels=_parse_signal_channels(
            sys.get("control_channels"),
            control_names,
            "control_channels",
        ),
        disturbance_channels=_parse_signal_channels(
            sys.get("disturbance_channels"),
            disturbance_names,
            "disturbance_channels",
        ),
        parameter_descriptors=_parse_parameter_descriptors(
            sys,
            ["A1", "A2", "k12", "kout", "h_max"],
        ),
        unit_type=str(sys.get("unit_type", "coupled_tanks")),
        family=str(sys.get("family", "hydraulic")),
        subtype=sys.get("subtype", "two_tank_level"),
        law_tags=[
            str(tag)
            for tag in sys.get("law_tags", ["mass_balance", "gravity_flow"])
        ],
        topology_ports=_parse_topology_ports(sys),
    )


def _build_cstr_simulator(system_config: dict) -> ProcessSimulator:
    from dte.simulators.cstr import CSTRParams, CSTRSimulator

    cstr_cfg = system_config.get("cstr", {})
    params = CSTRParams(**{k: float(v) for k, v in cstr_cfg.items()})
    return CSTRSimulator(params)


def _build_heat_exchanger_simulator(system_config: dict) -> ProcessSimulator:
    from dte.simulators.heat_exchanger import HeatExchangerParams, HeatExchangerSimulator

    hx_cfg = system_config.get("heat_exchanger", {})
    params = HeatExchangerParams(**{k: float(v) for k, v in hx_cfg.items()})
    return HeatExchangerSimulator(params)


def _build_two_tank_simulator(system_config: dict) -> ProcessSimulator:
    from dte.simulators.two_tank import TwoTankParams, TwoTankSimulator

    two_tank_cfg = system_config.get("two_tank", {})
    params = TwoTankParams(**{k: float(v) for k, v in two_tank_cfg.items()})
    return TwoTankSimulator(params)


# Registry mapping system name -> builder function
_SPEC_BUILDERS = {
    "cstr": _build_cstr_spec,
    "heat_exchanger": _build_heat_exchanger_spec,
    "two_tank": _build_two_tank_spec,
}

_SIMULATOR_BUILDERS: Dict[str, Callable[[dict], ProcessSimulator]] = {
    "cstr": _build_cstr_simulator,
    "heat_exchanger": _build_heat_exchanger_simulator,
    "two_tank": _build_two_tank_simulator,
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
    builder = _SIMULATOR_BUILDERS.get(system_name)
    if builder is None:
        raise ValueError(
            f"Unknown system '{system_name}'. "
            f"Available systems: {list(_SPEC_BUILDERS.keys())}"
        )
    return builder(system_config)


def list_systems() -> list:
    """Return the names of all registered systems."""
    return list(_SPEC_BUILDERS.keys())
