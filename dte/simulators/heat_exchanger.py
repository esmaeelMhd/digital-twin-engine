"""Counter-current heat exchanger simulator.

Implements a lumped-parameter counter-current heat exchanger with the ODEs:

    dT_hot/dt  = (F_hot / V_hot)  * (T_hot_in  - T_hot)
                 - (UA / (V_hot  * rho * Cp)) * (T_hot - T_cold)

    dT_cold/dt = (F_cold / V_cold) * (T_cold_in - T_cold)
                 + (UA / (V_cold * rho * Cp)) * (T_hot - T_cold)

State:       [T_hot, T_cold]     -- K
Control:     [F_hot, F_cold]     -- L/min
Disturbance: [T_hot_in, T_cold_in]  -- K
Parameters:  [V_hot, V_cold, UA, rho, Cp]
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import diffrax

from dte.simulators.base import ProcessSimulator, SystemSpec
from dte.simulators.registry import _build_heat_exchanger_spec


@dataclass(frozen=True)
class HeatExchangerParams:
    """Physical parameters for the counter-current heat exchanger."""

    V_hot: float = 50.0     # Hot-side volume (L)
    V_cold: float = 50.0    # Cold-side volume (L)
    UA: float = 5000.0      # Overall heat transfer coefficient * area (J/(min*K))
    rho: float = 1000.0     # Fluid density (g/L) -- same for both sides
    Cp: float = 4.18        # Specific heat capacity (J/(g*K)) -- same for both sides


class HeatExchangerSimulator(ProcessSimulator):
    """Lumped-parameter counter-current heat exchanger simulator."""

    def __init__(self, params: HeatExchangerParams):
        self._params = params
        self._spec: Optional[SystemSpec] = None

    @property
    def spec(self) -> SystemSpec:
        if self._spec is None:
            from dte.simulators.registry import _build_heat_exchanger_spec
            self._spec = _build_heat_exchanger_spec({})
        return self._spec

    @property
    def params(self) -> HeatExchangerParams:
        return self._params

    def dynamics(
        self,
        t: float,
        state: Float[Array, "2"],
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
    ) -> Float[Array, "2"]:
        """ODE right-hand side.

        Args:
            t: Time (unused, kept for diffrax compatibility)
            state: [T_hot, T_cold] in K
            control: [F_hot, F_cold] in L/min
            disturbance: [T_hot_in, T_cold_in] in K
        """
        T_hot = state[0]
        T_cold = state[1]
        F_hot = control[0]
        F_cold = control[1]
        T_hot_in = disturbance[0]
        T_cold_in = disturbance[1]

        p = self._params
        heat_transfer = (p.UA / (p.rho * p.Cp)) * (T_hot - T_cold)

        dT_hot_dt = (F_hot / p.V_hot) * (T_hot_in - T_hot) - heat_transfer / p.V_hot
        dT_cold_dt = (F_cold / p.V_cold) * (T_cold_in - T_cold) + heat_transfer / p.V_cold

        return jnp.array([dT_hot_dt, dT_cold_dt])

    def simulate(
        self,
        initial_state: Float[Array, "2"],
        control_trajectory: Float[Array, "n_steps 2"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        """Simulate a full trajectory using diffrax.

        Args:
            initial_state: [T_hot_0, T_cold_0]
            control_trajectory: (n_steps, 2) array of [F_hot, F_cold]
            disturbance_trajectory: (n_steps, 2) array of [T_hot_in, T_cold_in]
            t_span: (t0, t1)
            dt: Integration timestep
            n_steps: Number of output steps

        Returns:
            dict with 'time', 'states', 'controls'
        """
        t0, t1 = t_span
        ts = jnp.linspace(t0, t1, n_steps)

        control_interp = diffrax.LinearInterpolation(ts, control_trajectory)
        disturbance_interp = diffrax.LinearInterpolation(ts, disturbance_trajectory)

        def _ode(t, y, args):
            u = control_interp.evaluate(t)
            d = disturbance_interp.evaluate(t)
            return self.dynamics(t, y, u, d)

        term = diffrax.ODETerm(_ode)
        solver = diffrax.Heun()
        saveat = diffrax.SaveAt(ts=ts)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=initial_state,
            saveat=saveat,
            max_steps=4096,
        )
        return {
            "time": ts,
            "states": solution.ys,
            "controls": control_trajectory,
        }

    def steady_state(
        self,
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
        initial_guess: Optional[Float[Array, "2"]] = None,
    ) -> Float[Array, "2"]:
        """Compute steady state analytically or via fixed-point iteration.

        For the heat exchanger the steady state has a closed-form solution
        when operating at constant inputs:

            T_hot_ss  = T_hot_in  - (UA/...)*Delta_T_ss
            T_cold_ss = T_cold_in + (UA/...)*Delta_T_ss

        We use 5 seconds of simulation to approximate it.
        """
        if initial_guess is None:
            initial_guess = disturbance  # T_hot, T_cold ≈ inlet temps

        t_end = 200.0
        n_steps = 2000
        controls_traj = jnp.tile(control[None, :], (n_steps, 1))
        disturbances_traj = jnp.tile(disturbance[None, :], (n_steps, 1))

        result = self.simulate(
            initial_guess, controls_traj, disturbances_traj,
            t_span=(0.0, t_end), dt=0.1, n_steps=n_steps
        )
        return result["states"][-1]

    def get_params_vector(self) -> Float[Array, "5"]:
        """Return parameters as a JAX array [V_hot, V_cold, UA, rho, Cp]."""
        p = self._params
        return jnp.array([p.V_hot, p.V_cold, p.UA, p.rho, p.Cp])

    def sample_params(
        self,
        key,
        variation: float = 0.1,
    ) -> HeatExchangerParams:
        """Sample randomised parameters within ±variation of nominals."""
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        p = self._params
        scale = lambda k, nom: float(nom * (1.0 + variation * jax.random.uniform(k, minval=-1, maxval=1)))
        return HeatExchangerParams(
            V_hot=scale(k1, p.V_hot),
            V_cold=scale(k2, p.V_cold),
            UA=scale(k3, p.UA),
            rho=float(p.rho),
            Cp=float(p.Cp),
        )
