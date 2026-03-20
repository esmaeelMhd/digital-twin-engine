"""Streamlit dashboard for Digital Twin Engine demo."""

import streamlit as st
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path

from dte.models.digital_twin import DigitalTwin
from dte.control.mpc import SamplingMPC
from dte.control.pid import CSTRPIDController
from dte.simulators.cstr import CSTRSimulator, CSTRParams


# Page config
st.set_page_config(
    page_title="Digital Twin Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("🏭 Digital Twin Engine")
st.markdown("### AI-Powered Process Control with Physics-Informed Neural SDEs")

# Cache model loading
@st.cache_resource
def load_model():
    """Load trained model (cached)."""
    # Try to load a trained model if it exists
    model_path = "outputs/test_train/final_model.eqx"
    config_path = "configs/training_default.yaml"
    
    if Path(model_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = DigitalTwin.load(model_path, config)
        return model, config
    else:
        st.warning("No trained model found. Please train a model first.")
        return None, None


@st.cache_resource
def load_simulator():
    """Load CSTR simulator."""
    with open("configs/cstr_default.yaml", "r") as f:
        config = yaml.safe_load(f)
    cstr_params_dict = {k: float(v) for k, v in config["cstr"].items()}
    params = CSTRParams(**cstr_params_dict)
    return CSTRSimulator(params), config


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("Model")
    model_name = st.selectbox("Select Model", ["CSTR v1"])
    
    # Operating parameters
    st.subheader("Operating Parameters")
    V = st.slider("Reactor Volume (L)", 50.0, 200.0, 100.0, 10.0)
    UA = st.slider("Heat Transfer (kJ/min/K)", 30.0, 80.0, 50.0, 5.0) * 1000
    
    # Setpoints
    st.subheader("Setpoints")
    T_setpoint = st.slider("Temperature (K)", 300.0, 400.0, 340.0, 5.0)
    Ca_setpoint = st.slider("Concentration (mol/L)", 0.1, 2.0, 0.8, 0.1)
    
    # Control mode
    st.subheader("Control")
    control_mode = st.radio(
        "Control Mode",
        ["Open Loop", "PID", "AI-MPC"],
        index=2,
    )
    
    # Disturbance scenario
    st.subheader("Disturbances")
    disturbance_scenario = st.radio(
        "Scenario",
        ["None", "Step in Ca_in", "Step in T_in", "Random"],
        index=1,
    )
    
    # Simulation length
    n_steps = st.slider("Simulation Steps", 50, 500, 200, 50)
    
    # Run button
    run_simulation = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

# Main area
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Simulation",
    "🔬 Digital Twin vs Reality",
    "📈 Performance Comparison",
    "ℹ️ Model Info"
])

with tab1:
    st.header("Live Process Simulation")
    
    if run_simulation:
        # Load model and simulator
        model, model_config = load_model()
        simulator, cstr_config = load_simulator()
        
        if model is None:
            st.error("Please train a model first using: `python scripts/train.py`")
            st.stop()
        
        with st.spinner("Running simulation..."):
            # Setup
            key = jax.random.PRNGKey(42)
            dt = cstr_config["simulation"]["dt"]
            
            # Initial state
            initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
            
            # Setpoints
            setpoints = jnp.array([Ca_setpoint, 0.0, T_setpoint, 300.0])
            
            # Generate disturbances
            if disturbance_scenario == "Step in Ca_in":
                disturbances = jnp.ones((n_steps, 2)) * jnp.array([1.0, 320.0])
                disturbances = disturbances.at[n_steps//4:, 0].set(1.5)
            elif disturbance_scenario == "Step in T_in":
                disturbances = jnp.ones((n_steps, 2)) * jnp.array([1.0, 320.0])
                disturbances = disturbances.at[n_steps//4:, 1].set(330.0)
            elif disturbance_scenario == "Random":
                key, subkey = jax.random.split(key)
                disturbances = jax.random.uniform(subkey, (n_steps, 2))
                disturbances = disturbances * jnp.array([1.0, 30.0]) + jnp.array([0.5, 305.0])
            else:
                disturbances = jnp.ones((n_steps, 2)) * jnp.array([1.0, 320.0])
            
            # Run simulation based on control mode
            if control_mode == "AI-MPC":
                with open("configs/mpc_default.yaml", "r") as f:
                    mpc_config = yaml.safe_load(f)
                mpc_controller = SamplingMPC(model, {"mpc": mpc_config["mpc"]})
                params = jnp.ones(6)
                key, subkey = jax.random.split(key)
                result = mpc_controller.run_closed_loop(
                    simulator, initial_state, disturbances, setpoints, params, n_steps, dt, subkey
                )
            elif control_mode == "PID":
                pid_controller = CSTRPIDController(T_setpoint=T_setpoint, Ca_setpoint=Ca_setpoint, dt=dt)
                result = pid_controller.run_closed_loop(simulator, initial_state, disturbances, n_steps, dt)
            else:  # Open Loop
                controls = jnp.tile(jnp.array([50.0, 300.0])[None, :], (n_steps, 1))
                result = simulator.simulate(
                    initial_state, controls, disturbances, (0.0, n_steps * dt), dt, n_steps
                )
                result["controls"] = controls
        
        # Plot results
        times = np.arange(n_steps) * dt
        states = np.array(result["states"])
        controls = np.array(result["controls"])
        
        # Create plotly figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Ca (Concentration)", "Cb (Product)", "T (Temperature)", 
                          "Tc (Coolant Temp)", "F_in (Flow Rate)", "Tc_in (Coolant In)"),
            vertical_spacing=0.1,
        )
        
        state_names = ["Ca", "Cb", "T", "Tc"]
        for i, name in enumerate(state_names):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(
                go.Scatter(x=times, y=states[:, i], mode='lines', name=name, line=dict(color='blue', width=2)),
                row=row, col=col
            )
            if i in [0, 2]:  # Show setpoints for Ca and T
                setpoint_val = Ca_setpoint if i == 0 else T_setpoint
                fig.add_trace(
                    go.Scatter(x=times, y=[setpoint_val]*len(times), mode='lines', 
                             name=f'{name} setpoint', line=dict(color='red', dash='dash')),
                    row=row, col=col
                )
        
        # Add controls
        control_names = ["F_in", "Tc_in"]
        for i, name in enumerate(control_names):
            fig.add_trace(
                go.Scatter(x=times, y=controls[:, i], mode='lines', name=name, line=dict(color='green', width=2)),
                row=3, col=i+1
            )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            T_error = np.abs(states[:, 2] - T_setpoint).mean()
            st.metric("Avg T Error", f"{T_error:.2f} K")
        with col2:
            Ca_error = np.abs(states[:, 0] - Ca_setpoint).mean()
            st.metric("Avg Ca Error", f"{Ca_error:.3f} mol/L")
        with col3:
            control_effort = np.sum(np.diff(controls, axis=0)**2)
            st.metric("Control Effort", f"{control_effort:.2f}")
        with col4:
            st.metric("Simulation Time", f"{times[-1]:.1f} s")
    else:
        st.info("👈 Configure parameters in the sidebar and click 'Run Simulation'")

with tab2:
    st.header("Digital Twin vs Ground Truth")
    st.info("Compare digital twin predictions with actual simulator behavior")
    st.markdown("*Coming soon: Side-by-side comparison with prediction errors*")

with tab3:
    st.header("Controller Performance Comparison")
    st.info("Compare PID vs AI-MPC performance metrics")
    st.markdown("*Coming soon: Performance metrics table and charts*")

with tab4:
    st.header("Model Information")
    
    model, model_config = load_model()
    
    if model:
        st.subheader("Architecture")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Type:** Physics-Informed Latent Neural SDE")
            st.markdown("**Framework:** JAX + Equinox + Diffrax")
            st.markdown(f"**Latent Dimension:** {model_config['model']['latent_dim']}")
            st.markdown(f"**Hidden Dimension:** {model_config['model']['hidden_dim']}")
        
        with col2:
            param_counts = model.get_parameter_count()
            st.markdown(f"**Total Parameters:** {param_counts['total']:,}")
            st.markdown(f"**Encoder:** {param_counts['encoder']:,}")
            st.markdown(f"**Decoder:** {param_counts['decoder']:,}")
            st.markdown(f"**Latent SDE:** {param_counts['latent_sde']:,}")
        
        st.subheader("Training Configuration")
        st.json(model_config)
    else:
        st.warning("No model loaded. Train a model first.")

# Footer
st.markdown("---")
st.markdown(
    "**Powered by Digital Twin Engine** | Physics-Informed Latent Neural SDEs | "
    "[GitHub](https://github.com/your-repo)"
)
