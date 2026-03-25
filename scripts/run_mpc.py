"""Script to run and compare MPC vs PID control."""

import argparse
import os

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import yaml
import jax
import jax.numpy as jnp
import numpy as np

from dte.models.digital_twin import DigitalTwin
from dte.control.mpc import SamplingMPC
from dte.control.pid import CSTRPIDController
from dte.simulators.cstr import CSTRSimulator, CSTRParams
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
        "--cstr_config",
        type=str,
        default="configs/cstr_default.yaml",
        help="CSTR config"
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
    
    with open(args.cstr_config, "r") as f:
        cstr_config = yaml.safe_load(f)
    
    # Initialize
    key = jax.random.PRNGKey(args.seed)
    dt = cstr_config["simulation"]["dt"]
    
    # Load digital twin
    print("\nLoading digital twin...")
    model = DigitalTwin.load(args.model_path, model_config)
    
    # Create ground truth simulator
    cstr_params_dict = {k: float(v) for k, v in cstr_config["cstr"].items()}
    cstr_params = CSTRParams(**cstr_params_dict)
    simulator = CSTRSimulator(cstr_params)
    
    # Initial state
    initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
    
    # Setpoints
    setpoints = jnp.array([args.setpoint_Ca, 0.0, args.setpoint_T, 300.0])
    
    # Generate disturbances
    if args.disturbance_scenario == "step":
        # Step change in Ca_in at t=50
        disturbances = jnp.ones((args.n_steps, 2)) * jnp.array([1.0, 320.0])
        disturbances = disturbances.at[50:, 0].set(1.5)  # Step in Ca_in
    elif args.disturbance_scenario == "random":
        key, subkey = jax.random.split(key)
        disturbances = jax.random.uniform(subkey, (args.n_steps, 2))
        disturbances = disturbances * jnp.array([1.0, 30.0]) + jnp.array([0.5, 305.0])
    else:
        disturbances = jnp.ones((args.n_steps, 2)) * jnp.array([1.0, 320.0])
    
    # System parameters for MPC
    params = jnp.ones(6)  # Normalized params
    
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
    
    # Run PID if requested
    if args.compare_pid:
        print("\n" + "="*60)
        print("Running PID baseline...")
        print("="*60)
        
        pid_controller = CSTRPIDController(
            T_setpoint=args.setpoint_T,
            Ca_setpoint=args.setpoint_Ca,
            dt=dt,
        )
        
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
