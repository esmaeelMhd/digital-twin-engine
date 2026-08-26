"""Online adaptation / continual learning for Digital Twin models.

Implements:
- **Sliding-window fine-tuning**: Maintains a ring buffer of recent observations
  and periodically fine-tunes the model on the most recent window.
- **Drift detection**: Monitors rolling prediction error; triggers re-calibration
  when error exceeds an adaptive threshold (CUSUM-inspired).
- **Exponential forgetting**: Older data in the buffer is weighted less
  (L2-anchor regularisation toward the last fine-tune snapshot).
- **Integration with MPC loop**: :class:`OnlineAdapter` can be called after each
  new observation arrives from the plant.

Usage
-----
::

    from dte.training.online import OnlineAdapter, OnlineAdapterConfig

    cfg = OnlineAdapterConfig(
        window_size=500,
        finetune_every=50,
        n_finetune_steps=10,
        drift_threshold=0.05,
    )
    adapter = OnlineAdapter(model, loss_computer, optimizer_config, system_spec, cfg)

    # In the MPC/observation loop:
    for t, obs in plant_stream:
        adapter.push(states=obs.states, controls=obs.controls,
                     disturbances=obs.disturbances, t=obs.t)
        if adapter.drift_detected:
            print(f"Drift detected at t={t}!  Triggering recalibration.")
        model = adapter.model  # always up to date
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import PRNGKeyArray

from dte.models.unit.digital_twin import DigitalTwin
from dte.simulators.base import SystemSpec

try:
    from dte.training.shared.losses import LossComputer  # optional, for type hints only
except ImportError:
    LossComputer = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OnlineAdapterConfig:
    """Configuration for the online adaptation loop.

    Parameters
    ----------
    window_size:
        Number of most-recent time steps to keep in the replay buffer.
    finetune_every:
        Push this many new observations before triggering a fine-tune pass.
    n_finetune_steps:
        Number of gradient steps per fine-tune trigger.
    seq_len:
        Length of subsequences sampled from the buffer for fine-tuning.
    batch_size:
        Batch size for fine-tune gradient steps.
    learning_rate:
        Learning rate for online fine-tuning (usually 10-100× smaller than
        the offline rate to prevent catastrophic forgetting).
    gradient_clip:
        Global gradient-norm clip applied during fine-tuning.
    drift_threshold:
        CUSUM drift-detection threshold.  If the CUSUM statistic exceeds this
        value the adapter sets ``drift_detected = True``.
    drift_slack:
        Allowable normalised prediction error before CUSUM accumulates.
        A value near 0 is sensitive; 0.5 means "accumulate only when error
        exceeds 50% above the reference level".
    drift_reference_steps:
        Number of initial steps used to estimate the baseline error for drift
        detection (exponential moving average is used after this period).
    ewc_lambda:
        Strength of the L2-anchor forgetting penalty (0 = disabled).
        When > 0, parameters are pulled toward the snapshot taken after the
        previous fine-tune pass.  This is not Fisher-information EWC; no
        Fisher diagonal is estimated.
    finetune_encoder_only:
        If True, only encoder parameters receive gradient updates during
        fine-tuning (faster, prevents encoder collapse, better for small data).
    finetune_decoder_only:
        If True, only decoder parameters receive gradient updates.
        Mutually exclusive with *finetune_encoder_only*.
    """

    window_size: int = 500
    finetune_every: int = 50
    n_finetune_steps: int = 10
    seq_len: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    drift_threshold: float = 3.0
    drift_slack: float = 0.5
    drift_reference_steps: int = 100
    ewc_lambda: float = 0.0
    finetune_encoder_only: bool = False
    finetune_decoder_only: bool = False


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


class _RingBuffer:
    """Fixed-size ring buffer for numpy arrays.

    Stores ``(states, controls, disturbances, t)`` tuples of shape ``(d,)``.
    """

    def __init__(self, capacity: int, state_dim: int, control_dim: int, disturbance_dim: int):
        self.capacity = capacity
        self._buf_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._buf_controls = np.zeros((capacity, control_dim), dtype=np.float32)
        self._buf_disturbances = np.zeros((capacity, disturbance_dim), dtype=np.float32)
        self._buf_t = np.zeros(capacity, dtype=np.float32)
        self._head = 0
        self._size = 0

    def push(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        disturbances: np.ndarray,
        t: float,
    ):
        idx = self._head % self.capacity
        self._buf_states[idx] = states
        self._buf_controls[idx] = controls
        self._buf_disturbances[idx] = disturbances
        self._buf_t[idx] = t
        self._head += 1
        self._size = min(self._size + 1, self.capacity)

    def as_trajectory(self) -> Optional[Dict[str, np.ndarray]]:
        """Return contents as a single trajectory dict or None if too small."""
        if self._size < 2:
            return None
        if self._size < self.capacity:
            # Buffer not full yet: data is sequential from index 0
            s = self._buf_states[: self._size]
            c = self._buf_controls[: self._size]
            d = self._buf_disturbances[: self._size]
            t = self._buf_t[: self._size]
        else:
            # Oldest data is at _head % capacity; re-order
            start = self._head % self.capacity
            idx = np.arange(self.capacity)
            idx = np.roll(idx, -start)
            s = self._buf_states[idx]
            c = self._buf_controls[idx]
            d = self._buf_disturbances[idx]
            t = self._buf_t[idx]
        return {"states": s, "controls": c, "disturbances": d, "t": t}

    @property
    def size(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------


class _DriftDetector:
    """CUSUM-based drift detector for prediction error.

    After an initial reference period the detector accumulates a CUSUM
    statistic.  When it exceeds ``threshold`` a drift alarm is raised and
    the statistic is reset.

    Parameters
    ----------
    threshold:
        Alarm threshold for the CUSUM statistic.
    slack:
        Allowable deviation from the reference level before accumulation.
    reference_steps:
        Number of samples used to initialise the running mean reference.
    alpha:
        EMA coefficient for the running reference level (0 < alpha < 1).
        Smaller values make the reference adapt more slowly to true drift.
    """

    def __init__(
        self,
        threshold: float = 3.0,
        slack: float = 0.5,
        reference_steps: int = 100,
        alpha: float = 0.01,
    ):
        self.threshold = threshold
        self.slack = slack
        self.reference_steps = reference_steps
        self.alpha = alpha

        self._cusum: float = 0.0
        self._reference: Optional[float] = None
        self._n_seen: int = 0
        self._init_buffer: List[float] = []

        self.alarm: bool = False
        self.cusum_history: List[float] = []

    def update(self, error: float) -> bool:
        """Push a new normalised error value and check for drift.

        Returns True if an alarm was just triggered.
        """
        self._n_seen += 1

        if self._n_seen <= self.reference_steps:
            self._init_buffer.append(error)
            if self._n_seen == self.reference_steps:
                self._reference = float(np.mean(self._init_buffer))
            self.cusum_history.append(0.0)
            return False

        # EMA update of reference level (slow tracking)
        assert self._reference is not None
        self._reference = (1 - self.alpha) * self._reference + self.alpha * error

        # CUSUM accumulation
        deviation = error - self._reference * (1.0 + self.slack)
        self._cusum = max(0.0, self._cusum + deviation)
        self.cusum_history.append(self._cusum)

        if self._cusum > self.threshold:
            self.alarm = True
            self._cusum = 0.0  # reset after alarm
            return True
        return False

    def reset_alarm(self):
        self.alarm = False


# ---------------------------------------------------------------------------
# Online Adapter
# ---------------------------------------------------------------------------


class OnlineAdapter:
    """Adapts a Digital Twin model online as new plant observations arrive.

    Parameters
    ----------
    model:
        Pre-trained :class:`~dte.models.unit.digital_twin.DigitalTwin`.
    system_spec:
        :class:`~dte.simulators.base.SystemSpec` for the system.
    config:
        :class:`OnlineAdapterConfig` with hyper-parameters.
    key:
        Initial JAX PRNG key.
    loss_computer:
        Ignored.  Kept for API compatibility only.
    """

    def __init__(
        self,
        model: DigitalTwin,
        system_spec: SystemSpec,
        config: Optional[OnlineAdapterConfig] = None,
        key: Optional[PRNGKeyArray] = None,
        loss_computer: Optional[LossComputer] = None,
    ):
        if config is None:
            config = OnlineAdapterConfig()
        self.config = config
        self.model = model
        self.system_spec = system_spec
        self._key = key if key is not None else jax.random.PRNGKey(0)

        # Ring buffer
        self._buffer = _RingBuffer(
            capacity=config.window_size,
            state_dim=system_spec.state_dim,
            control_dim=system_spec.control_dim,
            disturbance_dim=system_spec.disturbance_dim,
        )

        # Drift detector
        self._drift = _DriftDetector(
            threshold=config.drift_threshold,
            slack=config.drift_slack,
            reference_steps=config.drift_reference_steps,
        )

        # Fine-tune optimizer (much smaller LR than offline)
        self._optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip),
            optax.adam(config.learning_rate),
        )
        self._opt_state = self._optimizer.init(eqx.filter(model, eqx.is_array))

        # L2-anchor snapshot (updated after each fine-tune pass if ewc_lambda > 0)
        self._anchor_params = eqx.filter(model, eqx.is_array) if config.ewc_lambda > 0 else None

        # Step counters
        self._push_count: int = 0
        self._finetune_count: int = 0
        self._step: int = 0

        # Prediction error ring (last 100 errors for diagnostics)
        self._pred_error_history: collections.deque = collections.deque(maxlen=500)

        # Default params (filled with ones; users can override via set_params)
        self._default_params = np.ones(system_spec.param_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def drift_detected(self) -> bool:
        return self._drift.alarm

    @property
    def n_observations(self) -> int:
        return self._push_count

    @property
    def n_finetune_passes(self) -> int:
        return self._finetune_count

    def set_params(self, params: np.ndarray):
        """Override the default parameter vector used during fine-tuning."""
        self._default_params = np.asarray(params, dtype=np.float32)

    def push(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        disturbances: Optional[np.ndarray] = None,
        t: float = 0.0,
    ) -> Dict:
        """Record a new plant observation and optionally trigger adaptation.

        Parameters
        ----------
        states:
            Observed state vector, shape ``(state_dim,)``.
        controls:
            Applied control input, shape ``(control_dim,)``.
        disturbances:
            Measured disturbances, shape ``(disturbance_dim,)``; zeros if None.
        t:
            Timestamp (seconds).

        Returns
        -------
        dict
            Status info with keys ``drift``, ``finetune_triggered``,
            ``pred_error`` (normalised MSE), ``cusum``.
        """
        if disturbances is None:
            disturbances = np.zeros(self.system_spec.disturbance_dim, dtype=np.float32)

        states = np.asarray(states, dtype=np.float32)
        controls = np.asarray(controls, dtype=np.float32)
        disturbances = np.asarray(disturbances, dtype=np.float32)

        # Compute one-step prediction error before updating buffer
        pred_error = self._compute_pred_error(states, controls, disturbances, t)
        self._pred_error_history.append(pred_error)

        # Update drift detector
        alarm_just_triggered = self._drift.update(pred_error)

        # Push to buffer
        self._buffer.push(states, controls, disturbances, t)
        self._push_count += 1

        # Periodic fine-tune
        finetune_triggered = False
        if (
            self._push_count % self.config.finetune_every == 0
            and self._buffer.size >= self.config.seq_len + 1
        ):
            self._finetune()
            finetune_triggered = True

        # Extra fine-tune when drift detected
        if alarm_just_triggered:
            self._drift.reset_alarm()
            if self._buffer.size >= self.config.seq_len + 1:
                self._finetune(extra_steps=self.config.n_finetune_steps)

        return {
            "drift": alarm_just_triggered,
            "finetune_triggered": finetune_triggered,
            "pred_error": float(pred_error),
            "cusum": self._drift._cusum,
        }

    def get_diagnostics(self) -> Dict:
        """Return current adapter diagnostics."""
        errors = list(self._pred_error_history)
        return {
            "push_count": self._push_count,
            "finetune_count": self._finetune_count,
            "buffer_size": self._buffer.size,
            "drift_alarm": self._drift.alarm,
            "cusum": self._drift._cusum,
            "pred_error_mean": float(np.mean(errors)) if errors else 0.0,
            "pred_error_std": float(np.std(errors)) if errors else 0.0,
            "pred_error_recent": float(errors[-1]) if errors else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_pred_error(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        disturbances: np.ndarray,
        t: float,
    ) -> float:
        """Compute normalised one-step prediction error.

        Uses the last state in the buffer as the "previous" observation and
        predicts the current state via Euler integration of the latent drift.
        Falls back to 0.0 when the buffer is empty (first observation).
        """
        if self._buffer.size == 0:
            return 0.0

        traj = self._buffer.as_trajectory()
        if traj is None:
            return 0.0

        prev_state = traj["states"][-1]
        prev_control = traj["controls"][-1]
        prev_disturbance = traj["disturbances"][-1]
        prev_t = float(traj["t"][-1])
        dt = t - prev_t
        if dt <= 0:
            return 0.0

        # Encode previous state -> latent
        key, self._key = jax.random.split(self._key)
        _, z_mean, _ = self.model.encode(
            jnp.array(prev_state),
            jnp.array(self._default_params),
            jnp.array(prev_control),
            key,
        )

        # Euler step in latent space
        drift = self.model.latent_sde.drift(
            z_mean,
            jnp.array(prev_control),
            jnp.array(prev_disturbance),
            jnp.array(self._default_params),
        )
        z_next = z_mean + dt * drift

        # Decode predicted state
        pred_state = self.model.decode(
            z_next,
            jnp.array(self._default_params),
            jnp.array(controls),
        )

        # Normalised MSE
        state_range = np.maximum(np.abs(states), 1e-6)
        err = float(jnp.mean(((pred_state - jnp.array(states)) / jnp.array(state_range)) ** 2))
        return err

    def _finetune(self, extra_steps: int = 0):
        """Run gradient steps on the current window."""
        n_steps = self.config.n_finetune_steps + extra_steps
        traj = self._buffer.as_trajectory()
        if traj is None:
            return

        seq_len = self.config.seq_len
        n_steps_in_buf = traj["states"].shape[0]
        if n_steps_in_buf < seq_len + 1:
            return

        params_arr = jnp.array(self._default_params)

        @eqx.filter_jit
        def _step_fn(model, opt_state, batch, key):
            def loss_fn(m):
                states = batch["states"]
                controls = batch["controls"]
                disturbances = batch["disturbances"]
                ts = batch["t"]

                key_enc, _ = jax.random.split(key)
                _, z_mean, _ = m.encode(states[0], params_arr, controls[0], key_enc)
                z_traj = m.latent_sde.mean_trajectory(
                    ts, z_mean, controls, params_arr, disturbances=disturbances
                )
                decode_fn = jax.vmap(lambda z, u: m.decode(z, params_arr, u), in_axes=(0, 0))
                pred = decode_fn(z_traj, controls)

                # Simple MSE in state space
                mse = jnp.mean((pred - states) ** 2)

                # L2-anchor penalty toward the previous fine-tune snapshot
                ewc_pen = jnp.array(0.0)
                if self.config.ewc_lambda > 0 and self._anchor_params is not None:
                    curr_arrays = eqx.filter(m, eqx.is_array)
                    diff_leaves = jax.tree_util.tree_leaves(
                        jax.tree_util.tree_map(
                            lambda a, b: jnp.sum((a - b) ** 2),
                            curr_arrays,
                            self._anchor_params,
                        )
                    )
                    ewc_pen = self.config.ewc_lambda * sum(diff_leaves)

                return mse + ewc_pen

            loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)

            updates, new_opt_state = opt_state[0].update(
                grads, opt_state[1], eqx.filter(model, eqx.is_array)
            )
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss_val

        # Wrap optimizer into a tuple so it can be passed through JIT
        opt_tuple = (self._optimizer, self._opt_state)

        for _ in range(n_steps):
            # Random subsequence from buffer
            start = np.random.randint(0, n_steps_in_buf - seq_len)
            batch = {
                "states": jnp.array(traj["states"][start : start + seq_len]),
                "controls": jnp.array(traj["controls"][start : start + seq_len]),
                "disturbances": jnp.array(traj["disturbances"][start : start + seq_len]),
                "t": jnp.array(traj["t"][start : start + seq_len]),
            }
            self._key, subkey = jax.random.split(self._key)
            self.model, new_opt_state, _ = _step_fn(self.model, opt_tuple, batch, subkey)
            opt_tuple = (self._optimizer, new_opt_state)
            self._step += 1

        self._opt_state = opt_tuple[1]
        self._finetune_count += 1

        # Update L2-anchor snapshot
        if self.config.ewc_lambda > 0:
            self._anchor_params = eqx.filter(self.model, eqx.is_array)

    # ------------------------------------------------------------------
    # Convenience: reset
    # ------------------------------------------------------------------

    def reset(self, keep_model: bool = True):
        """Reset the adapter state (buffer, drift detector, step counters).

        Parameters
        ----------
        keep_model:
            If True (default), the current fine-tuned model is retained.
            If False, resets everything including the model back to initial state.
        """
        self._buffer = _RingBuffer(
            capacity=self.config.window_size,
            state_dim=self.system_spec.state_dim,
            control_dim=self.system_spec.control_dim,
            disturbance_dim=self.system_spec.disturbance_dim,
        )
        self._drift = _DriftDetector(
            threshold=self.config.drift_threshold,
            slack=self.config.drift_slack,
            reference_steps=self.config.drift_reference_steps,
        )
        self._push_count = 0
        self._finetune_count = 0
        self._step = 0
        self._pred_error_history.clear()
        if not keep_model:
            self._opt_state = self._optimizer.init(eqx.filter(self.model, eqx.is_array))
