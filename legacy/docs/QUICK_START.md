# Quick Start Guide

This guide will walk you through the complete workflow from setup to dashboard deployment.

## Prerequisites

- Python 3.10-3.14
- Node.js `^20.19.0 || >=22.12.0` for the Vite frontend
- 8GB+ GPU (RTX 4070 or similar recommended)
- ~10GB free disk space for data

## Step-by-Step Instructions

### 1. Environment Setup (5 minutes)

```bash
# Navigate to project directory
cd digital-twin-engine

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package and dependencies
pip install -e .
```

**Verify installation:**
```bash
python -c "import dte; print('✓ Installation successful!')"
```

### 2. Generate Training Data (30-60 minutes)

Generate 10,000 diverse CSTR trajectories:

```bash
python scripts/generate_data.py \
  --n_trajectories 10000 \
  --n_steps 1000 \
  --output_dir data/cstr/ \
  --seed 42
```

**What this does:**
- Samples random CSTR parameters
- Generates diverse control signals (PRBS, chirp, multi-step)
- Simulates ground truth trajectories
- Adds measurement noise
- Saves to HDF5 format with normalization stats

**Expected output:** `data/cstr/train_data.h5` (~400MB)

**Quick test (2 minutes):**
```bash
python scripts/generate_data.py \
  --n_trajectories 100 \
  --n_steps 100 \
  --output_dir data/test/
```

Alternative built-in systems:
```bash
python scripts/generate_data.py --config configs/heat_exchanger_default.yaml --output_dir data/heat_exchanger/
python scripts/generate_data.py --config configs/two_tank_default.yaml --output_dir data/two_tank/
```

### 3. Train the Digital Twin (2-4 hours on GPU)

Train the physics-informed neural SDE:

```bash
python scripts/train.py \
  --config configs/training_default.yaml \
  --data_dir data/cstr/ \
  --output_dir outputs/cstr_v1/ \
  --n_epochs 100 \
  --batch_size 64 \
  --seed 42
```

**What this does:**
- Loads and splits data (80/20 train/val)
- Initializes encoder, decoder, latent SDE
- Trains with multiple loss terms:
  - Trajectory reconstruction
  - KL divergence (VAE)
  - Mass conservation
  - Energy conservation
- Saves best model based on validation loss

**Monitor training:**
- Watch console for loss values
- Best model saved to `outputs/cstr_v1/best_model.eqx`
- Training history saved to `training_history.json`

**Quick test (5 minutes):**
```bash
python scripts/train.py \
  --data_dir data/test/ \
  --output_dir outputs/test_train/ \
  --n_epochs 5 \
  --batch_size 8
```

### 4. Evaluate Model (10 minutes)

Evaluate prediction accuracy and physics constraint satisfaction:

```bash
python scripts/evaluate.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --config outputs/cstr_v1/config.yaml \
  --data_dir data/cstr/ \
  --output_dir outputs/cstr_v1/eval/ \
  --n_samples 20 \
  --n_trajectories 20
```

**Outputs:**
- Trajectory comparison plots
- Prediction error analysis
- Conservation violation metrics
- Ensemble uncertainty calibration
- Latent space visualization

**Key metrics to check:**
- 1-step MSE < 0.01 (normalized)
- Mass balance violation < 0.01
- Uncertainty calibration ~95% within ±2σ

### 5. Run MPC Control (15 minutes)

Compare AI-MPC vs PID baseline:

```bash
python scripts/run_mpc.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --model_config outputs/cstr_v1/config.yaml \
  --setpoint_T 340.0 \
  --setpoint_Ca 0.8 \
  --disturbance_scenario step \
  --n_steps 200 \
  --compare_pid
```

**What this does:**
- Runs closed-loop MPC using digital twin
- Runs PID baseline for comparison
- Computes performance metrics:
  - ISE (Integral Squared Error)
  - Settling time
  - Overshoot
  - Control effort
- Generates comparison plots

**Expected:** MPC should achieve 20-40% improvement over PID

### 6. Launch Browser Demo Frontend

Start the FastAPI service first:

```bash
uvicorn dte.api.service:app --host 0.0.0.0 --port 8000
```

Then start the React/Vite frontend:

```bash
nvm use
cd frontend
npm install
npm run dev
```

Opens in browser at `http://localhost:5173`

If `nvm use` fails because `nvm` is not installed, upgrade Node manually first. `node v18.x` is not supported by the current frontend toolchain.

### 7. Launch Dashboard (Instant)

Start the interactive Streamlit demo:

```bash
streamlit run app/dashboard.py
```

Opens in browser at `http://localhost:8501`

**Features:**
- Interactive parameter tuning
- Real-time simulation
- Three control modes: Open Loop, PID, AI-MPC
- Multiple disturbance scenarios
- Live performance metrics

## Troubleshooting

### Issue: CUDA not found

**Solution:** Install CPU-only JAX:
```bash
pip install --upgrade jax jaxlib  # Without [cuda12]
```

### Issue: Out of memory during training

**Solutions:**
1. Reduce batch size: `--batch_size 32`
2. Reduce sequence length in `configs/training_default.yaml`: `seq_len: 30`
3. Reduce model size: `hidden_dim: 64`, `latent_dim: 8`

### Issue: Training is slow

**Check:**
```python
python -c "import jax; print(jax.devices())"
# Should show GPU device
```

If only CPU, verify CUDA installation.

### Issue: NaN losses during training

**Causes:**
- Learning rate too high
- Physics loss weights too large
- Unstable trajectories in data

**Solutions:**
1. Reduce learning rate: `peak_lr: 1.0e-4`
2. Reduce physics weights: `mass_balance: 0.01`, `energy_balance: 0.01`
3. Re-generate data with more stable parameters

## Next Steps

1. **Tune hyperparameters**: Adjust learning rate, loss weights, model size
2. **Generate more data**: Increase to 50K trajectories for better generalization
3. **Optimize MPC**: Tune horizon, candidate count, cost weights
4. **Add more physics**: Implement momentum conservation, reaction kinetics
5. **Multi-reactor systems**: Extend to reactor networks

## File Outputs Reference

| Path | Description | Size |
|------|-------------|------|
| `data/cstr/train_data.h5` | Training dataset | ~400MB |
| `outputs/cstr_v1/best_model.eqx` | Best trained model | ~1MB |
| `outputs/cstr_v1/config.yaml` | Training config | <1KB |
| `outputs/cstr_v1/training_history.json` | Loss curves | <100KB |
| `outputs/cstr_v1/eval/*.png` | Evaluation plots | ~2MB |
| `outputs/mpc_results/*.png` | MPC comparison plots | ~1MB |

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| 1-step MSE | < 0.01 | 0.005 |
| 10-step MSE | < 0.05 | 0.02 |
| Full-seq MSE | < 0.1 | 0.08 |
| Mass violation | < 0.01 | 0.005 |
| Energy violation | < 1.0 | 0.5 |
| MPC solve time | < 500ms | 200ms |

## Citation

If you use this software in your research, please cite:

```bibtex
@software{digital_twin_engine,
  title = {Digital Twin Engine: Physics-Informed Latent Neural SDEs},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/your-repo}
}
```

## License

Proprietary. All rights reserved.
