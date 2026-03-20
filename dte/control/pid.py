"""PID controller implementation as a baseline for comparison."""

from typing import Tuple, Optional, Dict
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.cstr import CSTRSimulator


class PIDController:
    """Standard PID controller with anti-windup."""
    
    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        setpoint: float,
        output_limits: Tuple[Optional[float], Optional[float]] = (None, None),
        dt: float = 0.1,
    ):
        """Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            setpoint: Target setpoint
            output_limits: (min, max) output limits
            dt: Time step
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.dt = dt
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
    
    def step(self, measurement: float) -> float:
        """Compute control action.
        
        Args:
            measurement: Current measured value
            
        Returns:
            Control action
        """
        # Compute error
        error = self.setpoint - measurement
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        I = self.Ki * self.integral
        
        # Derivative term (derivative on measurement to avoid setpoint jumps)
        if self.prev_measurement is not None:
            D = -self.Kd * (measurement - self.prev_measurement) / self.dt
        else:
            D = 0.0
        
        # Compute output
        output = P + I + D
        
        # Apply output limits and anti-windup
        min_out, max_out = self.output_limits
        if max_out is not None and output > max_out:
            output = max_out
            # Anti-windup: stop integrator from growing
            self.integral -= error * self.dt
        elif min_out is not None and output < min_out:
            output = min_out
            # Anti-windup: stop integrator from shrinking
            self.integral -= error * self.dt
        
        # Update state
        self.prev_error = error
        self.prev_measurement = measurement
        
        return output


class CSTRPIDController:
    """Dual PID loops for CSTR temperature and concentration control."""
    
    def __init__(
        self,
        T_setpoint: float = 350.0,
        Ca_setpoint: float = 0.5,
        dt: float = 0.1,
        # Temperature loop tuning (manipulating Tc_in)
        Kp_T: float = 5.0,
        Ki_T: float = 0.1,
        Kd_T: float = 0.5,
        # Concentration loop tuning (manipulating F_in)
        Kp_Ca: float = 50.0,
        Ki_Ca: float = 1.0,
        Kd_Ca: float = 5.0,
    ):
        """Initialize CSTR PID controller.
        
        Args:
            T_setpoint: Temperature setpoint (K)
            Ca_setpoint: Concentration setpoint (mol/L)
            dt: Time step
            Kp_T, Ki_T, Kd_T: Temperature PID gains
            Kp_Ca, Ki_Ca, Kd_Ca: Concentration PID gains
        """
        self.dt = dt
        
        # Temperature loop (controls Tc_in)
        self.pid_T = PIDController(
            Kp=Kp_T,
            Ki=Ki_T,
            Kd=Kd_T,
            setpoint=T_setpoint,
            output_limits=(280.0, 320.0),  # Coolant temperature limits
            dt=dt,
        )
        
        # Concentration loop (controls F_in)
        self.pid_Ca = PIDController(
            Kp=Kp_Ca,
            Ki=Ki_Ca,
            Kd=Kd_Ca,
            setpoint=Ca_setpoint,
            output_limits=(10.0, 100.0),  # Flow rate limits
            dt=dt,
        )
    
    def reset(self):
        """Reset both controllers."""
        self.pid_T.reset()
        self.pid_Ca.reset()
    
    def step(self, state: Float[Array, "4"]) -> Tuple[float, float]:
        """Compute control actions.
        
        Args:
            state: Current state [Ca, Cb, T, Tc]
            
        Returns:
            Tuple of (F_in, Tc_in)
        """
        Ca = float(state[0])
        T = float(state[2])
        
        # Compute control actions
        Tc_in = self.pid_T.step(T)
        F_in = self.pid_Ca.step(Ca)
        
        return F_in, Tc_in
    
    def run_closed_loop(
        self,
        simulator: CSTRSimulator,
        initial_state: Float[Array, "4"],
        disturbances: Float[Array, "n_steps 2"],
        n_steps: int,
        dt: float,
    ) -> Dict[str, Float[Array, "..."]]:
        """Run closed-loop simulation.
        
        Args:
            simulator: CSTR simulator
            initial_state: Initial state
            disturbances: Disturbance trajectory [Ca_in, T_in]
            n_steps: Number of steps
            dt: Time step
            
        Returns:
            Dictionary with trajectories
        """
        self.reset()
        
        # Initialize storage
        states = jnp.zeros((n_steps, 4))
        controls = jnp.zeros((n_steps, 2))
        
        # Set initial state
        states = states.at[0].set(initial_state)
        
        # Closed-loop simulation
        current_state = initial_state
        
        for i in range(n_steps - 1):
            # Compute control action
            F_in, Tc_in = self.step(current_state)
            control = jnp.array([F_in, Tc_in])
            controls = controls.at[i].set(control)
            
            # Simulate one step
            control_traj = jnp.tile(control[None, :], (10, 1))  # Small steps
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
        
        # Last control
        F_in, Tc_in = self.step(current_state)
        controls = controls.at[-1].set(jnp.array([F_in, Tc_in]))
        
        return {
            "states": states,
            "controls": controls,
            "disturbances": disturbances,
        }
