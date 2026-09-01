"""Model Predictive Control using sampling-based optimization (Cross-Entropy Method)."""

from typing import Dict
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from dte.models.unit.digital_twin import DigitalTwin
from dte.simulators.cstr import CSTRSimulator


def _sequence_cost(
    predicted_states: Float[Array, "horizon state_dim"],
    control_sequence: Float[Array, "horizon control_dim"],
    setpoints: Float[Array, "state_dim"],
    u_prev: Float[Array, "control_dim"],
    state_weights: Float[Array, "state_dim"],
    control_weights: Float[Array, "control_dim"],
    terminal_weight: Float[Array, ""],
) -> Float[Array, ""]:
    """Scalar CEM tracking cost for one candidate control sequence."""

    state_errors = predicted_states - setpoints[None, :]
    state_cost = jnp.sum((state_errors ** 2) * state_weights[None, :])
    control_changes = jnp.diff(
        jnp.concatenate([u_prev[None, :], control_sequence], axis=0),
        axis=0,
    )
    control_cost = jnp.sum((control_changes ** 2) * control_weights[None, :])
    terminal_error = predicted_states[-1] - setpoints
    terminal_cost = terminal_weight * jnp.sum((terminal_error ** 2) * state_weights)
    return state_cost + control_cost + terminal_cost


@eqx.filter_jit
def _candidate_costs(
    model: DigitalTwin,
    candidates: Float[Array, "n_candidates horizon control_dim"],
    current_state: Float[Array, "state_dim"],
    params: Float[Array, "param_dim"],
    setpoints: Float[Array, "state_dim"],
    disturbance_forecast: Float[Array, "horizon disturbance_dim"],
    ts: Float[Array, "horizon"],
    u_prev: Float[Array, "control_dim"],
    state_weights: Float[Array, "state_dim"],
    control_weights: Float[Array, "control_dim"],
    terminal_weight: Float[Array, ""],
) -> Float[Array, "n_candidates"]:
    """Vectorized candidate evaluation; compiled once across CEM iterations."""

    def evaluate_candidate(controls):
        z0, _, _ = model.encode(current_state, params, controls[0], None)
        z_traj = model.rollout_latent(
            ts,
            z0,
            controls,
            params,
            disturbances=disturbance_forecast,
        )
        decode_fn = jax.vmap(lambda z, u: model.decode(z, params, u), in_axes=(0, 0))
        pred_states = decode_fn(z_traj, controls)
        return _sequence_cost(
            pred_states,
            controls,
            setpoints,
            u_prev,
            state_weights,
            control_weights,
            terminal_weight,
        )

    return jax.vmap(evaluate_candidate)(candidates)


class SamplingMPC:
    """Model Predictive Control using random shooting / CEM with the digital twin.
    
    Fully JAX-native. No CasADi needed.
    """
    
    def __init__(
        self,
        model: DigitalTwin,
        config: dict,
    ):
        """Initialize sampling-based MPC.
        
        Args:
            model: Digital Twin model for predictions
            config: MPC configuration
        """
        self.model = model
        self.config = config["mpc"]
        
        # MPC parameters
        self.horizon = self.config["horizon"]
        self.n_candidates = self.config["n_candidates"]
        self.n_elite = self.config["n_elite"]
        self.n_iterations = self.config["n_iterations"]
        self.initial_std = self.config["initial_std"]
        
        # Control bounds
        self.control_bounds = jnp.array([
            self.config["control_bounds"]["F_in"],
            self.config["control_bounds"]["Tc_in"],
        ])
        
        # Cost weights
        self.state_weights = jnp.array(self.config["cost_weights"]["state"])
        self.control_weights = jnp.array(self.config["cost_weights"]["control_effort"])
        self.terminal_weight = self.config["cost_weights"]["terminal"]
        
        # Warm start
        self.prev_solution = None
    
    def compute_cost(
        self,
        predicted_states: Float[Array, "horizon state_dim"],
        control_sequence: Float[Array, "horizon control_dim"],
        setpoints: Float[Array, "state_dim"],
        u_prev: Float[Array, "control_dim"],
    ) -> Float[Array, ""]:
        """Compute cost for a predicted trajectory.
        
        Cost = sum_t [ (x_t - x_ref)^T Q (x_t - x_ref) + (u_t - u_prev)^T R (u_t - u_prev) ]
               + terminal_weight * (x_T - x_ref)^T Q (x_T - x_ref)
        
        Args:
            predicted_states: Predicted state trajectory
            control_sequence: Control sequence
            setpoints: State setpoints
            u_prev: Previous control (for penalizing control effort)
            
        Returns:
            Scalar cost
        """
        return _sequence_cost(
            predicted_states,
            control_sequence,
            setpoints,
            u_prev,
            self.state_weights,
            self.control_weights,
            self.terminal_weight,
        )

    def solve(
        self,
        current_state: Float[Array, "state_dim"],
        params: Float[Array, "param_dim"],
        setpoints: Float[Array, "state_dim"],
        disturbance_forecast: Float[Array, "horizon disturbance_dim"],
        dt: float,
        key: PRNGKeyArray,
        u_prev: Float[Array, "control_dim"],
    ) -> Float[Array, "horizon control_dim"]:
        """Solve MPC optimization using Cross-Entropy Method.
        
        Args:
            current_state: Current state
            params: System parameters
            setpoints: State setpoints
            disturbance_forecast: Predicted disturbance sequence over the horizon
            dt: Time step
            key: PRNG key
            u_prev: Previous control action
            
        Returns:
            Optimal control sequence
        """
        # Initialize mean and std
        if self.prev_solution is not None:
            # Warm start: shift previous solution
            mean = jnp.concatenate([
                self.prev_solution[1:],
                self.prev_solution[-1:],  # Repeat last control
            ], axis=0)
        else:
            # Cold start: middle of bounds
            mean = jnp.tile(
                (self.control_bounds[:, 0] + self.control_bounds[:, 1]) / 2,
                (self.horizon, 1)
            )
        
        std = jnp.ones_like(mean) * self.initial_std
        ts = jnp.arange(self.horizon) * dt

        # CEM iterations
        for _iteration in range(self.n_iterations):
            key, subkey = jax.random.split(key)

            # Sample candidate control sequences
            keys = jax.random.split(subkey, self.n_candidates)
            noise = jax.vmap(lambda k: jax.random.normal(k, shape=mean.shape))(keys)
            candidates = mean[None, :, :] + std[None, :, :] * noise

            # Clip to bounds
            candidates = jnp.clip(
                candidates,
                self.control_bounds[:, 0][None, None, :],
                self.control_bounds[:, 1][None, None, :],
            )

            costs = _candidate_costs(
                self.model,
                candidates,
                current_state,
                params,
                setpoints,
                disturbance_forecast,
                ts,
                u_prev,
                self.state_weights,
                self.control_weights,
                self.terminal_weight,
            )

            # Select elite samples
            elite_indices = jnp.argsort(costs)[:self.n_elite]
            elite_candidates = candidates[elite_indices]

            # Update mean and std
            mean = jnp.mean(elite_candidates, axis=0)
            std = jnp.std(elite_candidates, axis=0) + 1e-6  # Add small value for numerical stability

        return mean
    
    def step(
        self,
        current_state: Float[Array, "state_dim"],
        params: Float[Array, "param_dim"],
        setpoints: Float[Array, "state_dim"],
        disturbance_forecast: Float[Array, "horizon disturbance_dim"],
        dt: float,
        key: PRNGKeyArray,
        u_prev: Float[Array, "control_dim"],
    ) -> Float[Array, "control_dim"]:
        """Compute MPC control action for one step.
        
        Args:
            current_state: Current state
            params: System parameters
            setpoints: State setpoints
            disturbance_forecast: Predicted disturbance sequence over the horizon
            dt: Time step
            key: PRNG key
            u_prev: Previous control action
            
        Returns:
            Control action for this step
        """
        # Solve MPC
        optimal_sequence = self.solve(
            current_state,
            params,
            setpoints,
            disturbance_forecast,
            dt,
            key,
            u_prev,
        )
        
        # Store for warm start
        self.prev_solution = optimal_sequence
        
        # Return first control action
        return optimal_sequence[0]
    
    def run_closed_loop(
        self,
        simulator: CSTRSimulator,
        initial_state: Float[Array, "4"],
        disturbances: Float[Array, "n_steps 2"],
        setpoints: Float[Array, "4"],
        params: Float[Array, "param_dim"],
        n_steps: int,
        dt: float,
        key: PRNGKeyArray,
    ) -> Dict[str, Array]:
        """Run closed-loop MPC simulation.
        
        Args:
            simulator: Ground truth simulator
            initial_state: Initial state
            disturbances: Disturbance trajectory
            setpoints: State setpoints
            params: System parameters
            n_steps: Number of steps
            dt: Time step
            key: PRNG key
            
        Returns:
            Dictionary with trajectories
        """
        # Reset warm start
        self.prev_solution = None
        
        # Initialize storage
        states = jnp.zeros((n_steps, 4))
        controls = jnp.zeros((n_steps, 2))
        costs = jnp.zeros(n_steps)
        
        states = states.at[0].set(initial_state)
        current_state = initial_state
        u_prev = jnp.array([50.0, 300.0])  # Initialize with nominal values
        
        print("\nRunning closed-loop MPC...")
        
        for i in range(n_steps - 1):
            if i % 10 == 0:
                print(f"Step {i}/{n_steps}")
            
            # Compute MPC action
            key, subkey = jax.random.split(key)
            forecast_end = min(i + self.horizon, n_steps)
            disturbance_forecast = disturbances[i:forecast_end]
            if disturbance_forecast.shape[0] < self.horizon:
                pad = jnp.tile(
                    disturbance_forecast[-1][None, :],
                    (self.horizon - disturbance_forecast.shape[0], 1),
                )
                disturbance_forecast = jnp.concatenate([disturbance_forecast, pad], axis=0)

            control = self.step(
                current_state,
                params,
                setpoints,
                disturbance_forecast,
                dt,
                subkey,
                u_prev,
            )
            controls = controls.at[i].set(control)
            
            # Apply to simulator (ground truth)
            control_traj = jnp.tile(control[None, :], (10, 1))
            dist_traj = jnp.tile(disturbances[i][None, :], (10, 1))
            
            result = simulator.simulate(
                current_state,
                control_traj,
                dist_traj,
                t_span=(i * dt, (i + 1) * dt),
                dt=dt / 10,
                n_steps=10,
            )
            
            # Update state
            current_state = result["states"][-1]
            states = states.at[i + 1].set(current_state)
            u_prev = control
        
        # Last control
        key, subkey = jax.random.split(key)
        disturbance_forecast = jnp.tile(disturbances[-1][None, :], (self.horizon, 1))
        control = self.step(
            current_state,
            params,
            setpoints,
            disturbance_forecast,
            dt,
            subkey,
            u_prev,
        )
        controls = controls.at[-1].set(control)
        
        return {
            "states": states,
            "controls": controls,
            "disturbances": disturbances,
            "costs": costs,
        }
