"""Script to run and compare MPC vs PID control."""

import argparse
import os

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import yaml
import jax
import jax.numpy as jnp
import numpy as np

from dte.models.unit.digital_twin import DigitalTwin
from dte.control.mpc import SamplingMPC
from dte.simulators.registry import get_system_spec, get_simulator
from dte.utils.plotting import plot_mpc_results


def compute_performance_metrics(states, setpoints, controls):
    """Compute control performance metrics.
    
    Returns:
        Dictionary with metrics
    """
    # Tracking error (ISE - Integral Squared Error)
    tracking_error = states - setpoints[None, :]
    ise = float(jnp.sum(tracking_error ** 2))
    
    # Control effort
    control_effort = float(jnp.sum(jnp.diff(controls, axis=0) ** 2))
    
    # Settling time (time to reach within 5% of setpoint)
    # Focus on temperature (index 2)
    T_error = jnp.abs(states[:, 2] - setpoints[2])
    tolerance = 0.05 * jnp.abs(setpoints[2])
    settled = T_error < tolerance
    settling_time = jnp.where(settled, jnp.arange(len(states)), len(states)).min()
    
    # Overshoot
    T_overshoot = jnp.max(jnp.abs(states[:, 2] - setpoints[2])) / jnp.abs(setpoints[2]) * 100
    
    return {
        "ise": ise,
        "control_effort": control_effort,
        "settling_time": int(settling_time),
        "overshoot_percent": float(T_overshoot),
    }


def _build_pid_controller(system_spec, setpoints, dt):
    """Return the legacy PID baseline when the system supports it."""

    if system_spec.name != "cstr":
        return None

    from dte.control.pid import CSTRPIDController

    return CSTRPIDController(
        T_setpoint=float(setpoints[2]),
        Ca_setpoint=float(setpoints[0]),
        dt=float(dt),
    )


def main():
    parser = argparse.ArgumentParser(description="Run MPC control comparison")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained digital twin"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mpc_default.yaml",
        help="MPC config"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Model config"
    )
    parser.add_argument(
        "--system_config",
        type=str,
        default="configs/cstr_default.yaml",
        help="System config"
    )
    parser.add_argument(
        "--setpoint_T",
        type=float,
        default=340.0,
        help="Temperature setpoint (K)"
    )
    parser.add_argument(
        "--setpoint_Ca",
        type=float,
        default=0.8,
        help="Concentration setpoint (mol/L)"
    )
    parser.add_argument(
        "--disturbance_scenario",
        type=str,
        default="step",
        choices=["step", "none", "random"],
        help="Disturbance scenario"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=200,
        help="Number of simulation steps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/mpc_results/",
        help="Output directory"
    )
    parser.add_argument(
        "--compare_pid",
        action="store_true",
        help="Compare with PID baseline"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("MPC CONTROL COMPARISON")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Setpoints: T={args.setpoint_T}K, Ca={args.setpoint_Ca} mol/L")
    print(f"Disturbance: {args.disturbance_scenario}")
    print(f"Steps: {args.n_steps}")
    print("="*60)
    
    # Load configs
    with open(args.config, "r") as f:
        mpc_config = yaml.safe_load(f)

    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    with open(args.system_config, "r") as f:
        system_config = yaml.safe_load(f)

    system_spec = get_system_spec(system_config)

    # Initialize
    key = jax.random.PRNGKey(args.seed)
    dt = system_config.get("simulation", {}).get("dt", 0.1)

    # Load digital twin
    print("\nLoading digital twin...")
    model = DigitalTwin.load(
        args.model_path,
        model_config,
        system_spec=system_spec,
        system_config=system_config,
    )

    # Create ground truth simulator
    simulator = get_simulator(system_spec.name, system_config)

    # Initial state and nominal disturbances from system spec
    initial_state = system_spec.default_initial_state_array()
    nominal_disturbance = system_spec.default_nominal_disturbance_array()

    # Setpoints: for CSTR use CLI values, for other systems use spec defaults
    if system_spec.name == "cstr":
        setpoints = jnp.array([args.setpoint_Ca, 0.0, args.setpoint_T, 300.0])
    else:
        setpoints = initial_state  # use initial state as default setpoint

    # Generate disturbances
    dist_dim = system_spec.disturbance_dim
    if args.disturbance_scenario == "step":
        disturbances = jnp.tile(nominal_disturbance[None, :], (args.n_steps, 1))
        if dist_dim > 0:
            step_val = nominal_disturbance[0] * 1.5
            disturbances = disturbances.at[50:, 0].set(step_val)
    elif args.disturbance_scenario == "random":
        key, subkey = jax.random.split(key)
        dist_ranges = list(system_spec.disturbance_ranges.values())
        if dist_ranges:
            lo = jnp.array([r[0] for r in dist_ranges])
            hi = jnp.array([r[1] for r in dist_ranges])
            disturbances = jax.random.uniform(subkey, (args.n_steps, dist_dim)) * (hi - lo) + lo
        else:
            disturbances = jnp.tile(nominal_disturbance[None, :], (args.n_steps, 1))
    else:
        disturbances = jnp.tile(nominal_disturbance[None, :], (args.n_steps, 1))

    # System parameters for MPC (normalized)
    params = jnp.ones(system_spec.param_dim)
    
    # Run MPC
    print("\n" + "="*60)
    print("Running AI-MPC...")
    print("="*60)
    mpc_controller = SamplingMPC(model, {"mpc": mpc_config["mpc"]})
    
    key, subkey = jax.random.split(key)
    mpc_result = mpc_controller.run_closed_loop(
        simulator,
        initial_state,
        disturbances,
        setpoints,
        params,
        args.n_steps,
        dt,
        subkey,
    )
    
    mpc_metrics = compute_performance_metrics(
        mpc_result["states"], setpoints, mpc_result["controls"]
    )
    
    print("\nMPC Performance:")
    print(f"  ISE: {mpc_metrics['ise']:.2f}")
    print(f"  Control effort: {mpc_metrics['control_effort']:.2f}")
    print(f"  Settling time: {mpc_metrics['settling_time']} steps")
    print(f"  Overshoot: {mpc_metrics['overshoot_percent']:.1f}%")
    
    # Plot MPC results
    times = jnp.arange(args.n_steps) * dt
    setpoints_traj = jnp.tile(setpoints[None, :], (args.n_steps, 1))
    
    fig = plot_mpc_results(
        np.array(mpc_result["states"]),
        np.array(mpc_result["controls"]),
        np.array(setpoints_traj),
        np.array(times),
        save_path=os.path.join(args.output_dir, "mpc_results.png")
    )
    
    # Run PID if requested (CSTR only for now)
    if args.compare_pid:
        pid_controller = _build_pid_controller(system_spec, setpoints, dt)
        if pid_controller is None:
            print(f"\nNote: PID baseline is only implemented for CSTR; skipping for '{system_spec.name}'.")
        else:
            pid_result = pid_controller.run_closed_loop(
                simulator,
                initial_state,
                disturbances,
                args.n_steps,
                dt,
            )

            pid_metrics = compute_performance_metrics(
                pid_result["states"], setpoints, pid_result["controls"]
            )

            print("\nPID Performance:")
            print(f"  ISE: {pid_metrics['ise']:.2f}")
            print(f"  Control effort: {pid_metrics['control_effort']:.2f}")
            print(f"  Settling time: {pid_metrics['settling_time']} steps")
            print(f"  Overshoot: {pid_metrics['overshoot_percent']:.1f}%")

            # Plot PID results
            fig = plot_mpc_results(
                np.array(pid_result["states"]),
                np.array(pid_result["controls"]),
                np.array(setpoints_traj),
                np.array(times),
                save_path=os.path.join(args.output_dir, "pid_results.png")
            )

            # Comparison
            print("\n" + "="*60)
            print("COMPARISON: MPC vs PID")
            print("="*60)
            print(f"ISE:            MPC={mpc_metrics['ise']:.2f}  PID={pid_metrics['ise']:.2f}  (lower is better)")
            print(f"Control effort: MPC={mpc_metrics['control_effort']:.2f}  PID={pid_metrics['control_effort']:.2f}")
            print(f"Settling time:  MPC={mpc_metrics['settling_time']}  PID={pid_metrics['settling_time']}  (steps)")
            print(f"Overshoot:      MPC={mpc_metrics['overshoot_percent']:.1f}%  PID={pid_metrics['overshoot_percent']:.1f}%")
        
            improvement = (pid_metrics['ise'] - mpc_metrics['ise']) / pid_metrics['ise'] * 100
            print(f"\n→ MPC achieves {improvement:.1f}% improvement in ISE")
    
    print("\n" + "="*60)
    print(f"Results saved to {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
