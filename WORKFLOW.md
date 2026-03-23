# Digital Twin Engine - Complete Workflow

This document provides a step-by-step workflow from initial setup to production deployment.

---

## 🚀 Phase 1: Environment Setup (10 minutes)

### 1.1 Clone and Setup Virtual Environment

```bash
# Navigate to project
cd digital-twin-engine

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .
```

### 1.2 Verify Installation

```bash
# Run verification script
python scripts/verify_install.py

# Expected output: "✓ ALL CHECKS PASSED"
```

### 1.3 Run Tests (Optional but Recommended)

```bash
# Run full test suite (30 tests)
pytest tests/ -v

# Expected: 30 passed in ~20 seconds
```

**Checkpoint:** ✅ All packages installed, tests passing

---

## 📊 Phase 2: Data Generation (30-60 minutes)

### 2.1 Quick Test Data (2 minutes)

```bash
# Generate small dataset for testing
python scripts/generate_data.py \
  --n_trajectories 100 \
  --n_steps 100 \
  --output_dir data/test/ \
  --seed 42
```

**Output:** `data/test/train_data.h5` (~4MB)

### 2.2 Production Data (30-60 minutes)

```bash
# Generate full training dataset
python scripts/generate_data.py \
  --n_trajectories 10000 \
  --n_steps 1000 \
  --output_dir data/cstr/ \
  --seed 42
```

**Output:** `data/cstr/train_data.h5` (~400MB)

**What's happening:**
- Random CSTR parameter sampling
- PRBS + chirp + multi-step control signals
- Ground truth simulation with Diffrax
- Measurement noise addition
- Normalization statistics computation

**Checkpoint:** ✅ Training data generated, no NaN values

---

## 🧠 Phase 3: Model Training (2-4 hours GPU / 8-12 hours CPU)

### 3.1 Quick Training Test (5 minutes)

```bash
# Train on test data to verify pipeline
python scripts/train.py \
  --data_dir data/test/ \
  --output_dir outputs/test_train/ \
  --n_epochs 5 \
  --batch_size 8
```

**Expected:** Loss decreases, no errors

### 3.2 Production Training (2-4 hours)

```bash
# Full training run
python scripts/train.py \
  --config configs/training_default.yaml \
  --data_dir data/cstr/ \
  --output_dir outputs/cstr_v1/ \
  --n_epochs 100 \
  --batch_size 64 \
  --seed 42
```

**Monitor training:**
```
Epoch 1/100: train_loss=2.456, val_loss=2.389
Epoch 10/100: train_loss=0.856, val_loss=0.823
Epoch 50/100: train_loss=0.234, val_loss=0.256
Epoch 100/100: train_loss=0.089, val_loss=0.095
```

**Outputs:**
- `outputs/cstr_v1/best_model.eqx` - Best model checkpoint
- `outputs/cstr_v1/final_model.eqx` - Final model
- `outputs/cstr_v1/training_history.json` - Loss curves
- `outputs/cstr_v1/config.yaml` - Training configuration

**Loss Components:**
- **Reconstruction**: Trajectory matching (~60% of total)
- **KL Divergence**: VAE regularization (~20% of total)
- **Trajectory**: Long-term consistency (~10% of total)
- **Mass Balance**: Conservation law (~5% of total)
- **Energy Balance**: Conservation law (~5% of total)

**Checkpoint:** ✅ Model trained, validation loss < 0.1

---

### 3.3 Optional: Autoresearch Loop (5 minutes per experiment)

If you want to apply the `karpathy/autoresearch` idea to this repo, use the bounded experiment harness:

```bash
python scripts/autoresearch.py \
  --config configs/autoresearch_default.yaml \
  --description baseline \
  --data_dir data/test/
```

For production experiments, point `--data_dir` at `data/cstr/`.

**What the harness does:**
- Runs `scripts/train.py` with a fixed wall-clock budget
- Forces regular validation so each run emits a comparable `best_val_loss`
- Stores run logs and model artifacts under `outputs/autoresearch/runs/<run_id>/`
- Appends the result to `outputs/autoresearch/results.tsv`
- Promotes the run into `outputs/autoresearch/baseline/` only if it improves the metric

**Agent workflow:** See `program.md` for the autonomous keep/discard loop instructions.

**Checkpoint:** ✅ Baseline established, experiments can iterate autonomously

---

## 📈 Phase 4: Model Evaluation (10 minutes)

### 4.1 Run Evaluation

```bash
python scripts/evaluate.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --config outputs/cstr_v1/config.yaml \
  --data_dir data/cstr/ \
  --output_dir outputs/cstr_v1/eval/ \
  --n_samples 20 \
  --n_trajectories 20
```

**Generated Plots:**
1. `trajectory_0.png`, `trajectory_1.png`, `trajectory_2.png` - Prediction vs ground truth
2. `error_0.png`, `error_1.png`, `error_2.png` - Absolute and relative errors
3. `conservation_0.png`, `conservation_1.png`, `conservation_2.png` - Physics violations
4. `ensemble_prediction.png` - Uncertainty quantification

**Key Metrics to Check:**
```
Prediction Accuracy (MSE in normalized space):
  1-step MSE:     0.0023 ± 0.0012  ✓ (target: < 0.01)
  10-step MSE:    0.0145 ± 0.0089  ✓ (target: < 0.05)
  Full-seq MSE:   0.0678 ± 0.0234  ✓ (target: < 0.1)

Physics Constraint Satisfaction:
  Mass balance violation:
    Mean: 0.0034 ± 0.0012  ✓ (target: < 0.01)
    Max:  0.0089 ± 0.0023  ✓ (target: < 0.02)
  Energy balance violation:
    Mean: 0.45 ± 0.12      ✓ (target: < 1.0)
    Max:  1.23 ± 0.34      ✓ (target: < 2.0)

Uncertainty Calibration:
  % within ±2σ: 94.2%  ✓ (target: ~95%)
```

**Checkpoint:** ✅ Model meets accuracy targets, physics constraints satisfied

---

## 🎮 Phase 5: Control Experiments (15 minutes)

### 5.1 MPC Performance Test

```bash
# Run MPC with step disturbance
python scripts/run_mpc.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --model_config outputs/cstr_v1/config.yaml \
  --setpoint_T 340.0 \
  --setpoint_Ca 0.8 \
  --disturbance_scenario step \
  --n_steps 200 \
  --compare_pid \
  --output_dir outputs/mpc_results/
```

**Expected Results:**
```
MPC Performance:
  ISE: 45.23
  Control effort: 12.45
  Settling time: 35 steps
  Overshoot: 8.3%

PID Performance:
  ISE: 68.91
  Control effort: 18.72
  Settling time: 52 steps
  Overshoot: 15.7%

→ MPC achieves 34.4% improvement in ISE
```

**Generated Plots:**
- `outputs/mpc_results/mpc_results.png` - AI-MPC performance
- `outputs/mpc_results/pid_results.png` - PID baseline performance

### 5.2 Try Different Scenarios

```bash
# Random disturbances
python scripts/run_mpc.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --model_config outputs/cstr_v1/config.yaml \
  --disturbance_scenario random \
  --compare_pid

# Different setpoints
python scripts/run_mpc.py \
  --model_path outputs/cstr_v1/best_model.eqx \
  --model_config outputs/cstr_v1/config.yaml \
  --setpoint_T 360.0 \
  --setpoint_Ca 0.5 \
  --disturbance_scenario step
```

**Checkpoint:** ✅ MPC outperforms PID, stable control achieved

---

## 🎨 Phase 6: Interactive Dashboard (Launch Anytime)

### 6.1 Start Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

**Access:** Opens automatically at `http://localhost:8501`

### 6.2 Dashboard Features

**Left Sidebar:**
- Operating parameters (V, UA)
- Setpoints (T, Ca)
- Control mode selection (Open Loop / PID / AI-MPC)
- Disturbance scenarios
- Simulation length

**Main Tabs:**

1. **📊 Live Simulation**
   - Real-time state trajectories
   - Control actions
   - Performance metrics
   - Interactive parameter tuning

2. **🔬 Digital Twin vs Reality**
   - Side-by-side comparison
   - Prediction errors

3. **📈 Performance Comparison**
   - PID vs AI-MPC metrics
   - Settling time, overshoot, ISE

4. **ℹ️ Model Info**
   - Architecture details
   - Parameter counts
   - Training configuration

**Demo Workflow in Dashboard:**
1. Select "AI-MPC" control mode
2. Choose "Step in Ca_in" disturbance
3. Click "▶️ Run Simulation"
4. Observe smooth setpoint tracking
5. Switch to "PID" and compare performance

**Checkpoint:** ✅ Dashboard running, interactive demos working

---

## 🔬 Phase 7: Exploratory Analysis (Optional)

### 7.1 Launch Jupyter Notebook

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

**Notebook Contents:**
1. CSTR Simulator testing
2. Data generation examples
3. Digital twin loading and usage
4. Prediction examples
5. Uncertainty quantification demos

### 7.2 Custom Analysis

Create your own notebooks for:
- Sensitivity analysis
- Latent space visualization
- Controller tuning
- Physics constraint analysis

**Checkpoint:** ✅ Ready for custom experiments

---

## 🔧 Phase 8: Hyperparameter Tuning (Optional)

### 8.1 Tune Model Architecture

Edit `configs/training_default.yaml`:

```yaml
model:
  latent_dim: 16    # Try: 8, 16, 32
  hidden_dim: 128   # Try: 64, 128, 256
  encoder_layers: 3 # Try: 2, 3, 4
```

### 8.2 Tune Training Parameters

```yaml
training:
  peak_lr: 3.0e-4   # Try: 1e-4, 3e-4, 1e-3
  batch_size: 64    # Try: 32, 64, 128
  seq_len: 50       # Try: 30, 50, 100

loss_weights:
  reconstruction: 1.0
  kl: 0.1           # Try: 0.01, 0.1, 1.0
  trajectory: 0.5
  mass_balance: 0.1  # Try: 0.01, 0.1, 0.5
  energy_balance: 0.1
```

### 8.3 Tune MPC Parameters

Edit `configs/mpc_default.yaml`:

```yaml
mpc:
  horizon: 10          # Try: 5, 10, 20
  n_candidates: 500    # Try: 100, 500, 1000
  n_elite: 50          # Try: 20, 50, 100
  n_iterations: 3      # Try: 2, 3, 5
```

**Retrain and Compare:**
```bash
python scripts/train.py --output_dir outputs/cstr_v2/
python scripts/evaluate.py --model_path outputs/cstr_v2/best_model.eqx
```

**Checkpoint:** ✅ Optimized model for your use case

---

## 📦 Phase 9: Production Deployment (Optional)

### 9.1 Export Model

```python
# Model is already serialized as .eqx file
# Load in production:
from dte.models.digital_twin import DigitalTwin
model = DigitalTwin.load("outputs/cstr_v1/best_model.eqx", config)
```

### 9.2 Deploy Dashboard

```bash
# Option 1: Local deployment
streamlit run app/dashboard.py --server.port 8501

# Option 2: Docker deployment (create Dockerfile)
docker build -t digital-twin-engine .
docker run -p 8501:8501 digital-twin-engine

# Option 3: Cloud deployment (Streamlit Cloud, AWS, etc.)
```

### 9.3 API Service (Optional)

Create a FastAPI service:

```python
from fastapi import FastAPI
from dte.models.digital_twin import DigitalTwin

app = FastAPI()
model = DigitalTwin.load("best_model.eqx", config)

@app.post("/predict")
def predict(state, controls, params):
    result = model.predict(state, controls, ...)
    return {"prediction": result}
```

**Checkpoint:** ✅ Production system deployed

---

## 🔄 Phase 10: Continuous Improvement

### 10.1 Regular Monitoring

- Track prediction accuracy on new data
- Monitor physics constraint violations
- Log control performance metrics

### 10.2 Model Updates

```bash
# Generate new data with updated parameters
python scripts/generate_data.py --n_trajectories 5000 --output_dir data/update/

# Retrain or fine-tune
python scripts/train.py --data_dir data/update/ --load_model outputs/cstr_v1/best_model.eqx
```

### 10.3 A/B Testing

- Compare new model versions
- Measure control improvements
- Update production model when validated

---

## 🎯 Quick Reference Commands

```bash
# Complete workflow in one go (for testing)
python scripts/verify_install.py && \
python scripts/generate_data.py --n_trajectories 100 --output_dir data/test/ && \
python scripts/train.py --data_dir data/test/ --n_epochs 5 --output_dir outputs/test/ && \
python scripts/evaluate.py --model_path outputs/test/final_model.eqx --config outputs/test/config.yaml --data_dir data/test/ && \
streamlit run app/dashboard.py
```

---

## 📊 Success Metrics Summary

| Phase | Metric | Target | Check |
|-------|--------|--------|-------|
| Data Generation | No NaN values | 100% | ✓ |
| Training | Val loss | < 0.1 | ✓ |
| Evaluation | 1-step MSE | < 0.01 | ✓ |
| Evaluation | Mass violation | < 0.01 | ✓ |
| Evaluation | Energy violation | < 1.0 | ✓ |
| Control | MPC vs PID improvement | > 20% | ✓ |
| Dashboard | Interactive demo | Working | ✓ |

---

## 🆘 Troubleshooting

### Issue: Out of memory during training
**Solution:** Reduce batch size or sequence length
```bash
python scripts/train.py --batch_size 32 --seq_len 30
```

### Issue: Training too slow
**Solution:** Check GPU availability
```python
import jax; print(jax.devices())  # Should show GPU
```

### Issue: NaN losses
**Solution:** Reduce learning rate or physics weights
```yaml
peak_lr: 1.0e-4
mass_balance: 0.01
```

### Issue: Poor MPC performance
**Solution:** Increase horizon or candidates
```yaml
horizon: 20
n_candidates: 1000
```

---

## 📚 Next Steps

1. **Extend to other reactors**: Modify `dte/simulators/` for different processes
2. **Add more physics**: Implement momentum, reaction kinetics
3. **Multi-reactor systems**: Extend to reactor networks
4. **Real-time integration**: Connect to plant data
5. **Advanced control**: Implement learning-based MPC, adaptive control

---

## 📖 Additional Resources

- `README.md` - Project overview and features
- `QUICK_START.md` - Condensed getting started guide
- `plan.md` - Original technical specification
- `notebooks/01_exploration.ipynb` - Interactive examples
- `tests/` - Unit tests showing usage examples

---

**🎉 You're now ready to use the Digital Twin Engine!**

For questions or issues, refer to the test suite in `tests/` for usage examples.
