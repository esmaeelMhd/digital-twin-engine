# AI Coding Agent Instructions: Initializing `flux-attention-engine`

**Context for the AI Agent:**
You are tasked with executing a critical architectural pivot. We are moving away from a dense, padded, Universal ODE architecture (`digital-twin-engine`) to a Graph-native, Port-Hamiltonian Attention architecture (`flux-attention-engine`). 

Follow these instructions sequentially to initialize the new repository, port the ground-truth physics simulators, and write the core mathematical primitives.

---

## Step 1: Repository & Environment Initialization

1. **Create the new workspace:**
   Run in the terminal:
   ```bash
   cd /home/ismayil
   mkdir flux-attention-engine
   cd flux-attention-engine
   git init
   ```

2. **Create the base directory structure:**
   ```bash
   mkdir -p flux/core flux/simulators flux/physics flux/data scripts tests
   touch flux/__init__.py flux/core/__init__.py flux/simulators/__init__.py flux/physics/__init__.py flux/data/__init__.py
   ```

3. **Port and Adapt `pyproject.toml`:**
   * Copy `/home/ismayil/digital-twin-engine/pyproject.toml` to `/home/ismayil/flux-attention-engine/pyproject.toml`.
   * **Edit the copied file:** Change the project name from `digital-twin-engine` to `flux-attention-engine`.
   * Add new dependencies: `rdkit`, `openai`, `anthropic`, `networkx`.
   * Remove any legacy specific dependencies if they are purely for the old dashboard (e.g., `streamlit` can stay or go, but keep it minimal).

4. **Initialize the Virtual Environment:**
   ```bash
   cd /home/ismayil/flux-attention-engine
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

---

## Step 2: Porting the Ground-Truth Simulators

We need the first-principles ODEs to generate ground-truth data for the graph networks.

1. **Copy the Simulator Modules:**
   Copy the following files from `/home/ismayil/digital-twin-engine/dte/simulators/` to `/home/ismayil/flux-attention-engine/flux/simulators/`:
   * `base.py`
   * `registry.py`
   * `cstr.py`
   * `two_tank.py`
   * `heat_exchanger.py`

2. **Copy the Physics Constraints:**
   Copy the following from `/home/ismayil/digital-twin-engine/dte/physics/` to `/home/ismayil/flux-attention-engine/flux/physics/`:
   * `base.py`
   * `registry.py`
   * `cstr.py`
   * `two_tank.py`
   * `heat_exchanger.py`

3. **Copy the Data Generator:**
   * Copy `/home/ismayil/digital-twin-engine/dte/data/generation_generic.py` to `flux/data/generation_generic.py`.
   * Copy `/home/ismayil/digital-twin-engine/scripts/generate_data.py` to `scripts/generate_data.py`.

4. **Refactor Imports (Crucial):**
   * Use a tool (or `sed` via terminal) in the new repository to replace all instances of `from dte.` or `import dte.` with `from flux.` or `import flux.`.
   * Ensure `scripts/generate_data.py` runs correctly without crashing on import errors.

---

## Step 3: Write the Graph & Attention Primitives (Net-New Code)

Now, you will write the core mathematical engine for the new architecture.

1. **Create `flux/core/graph.py`:**
   Write a JAX/Equinox dataclass named `PhysicalGraphTuple`. It must contain:
   * `X`: Float array of shape `(N_nodes, D_extensive)` representing extensive capacities (mass, energy).
   * `P`: Float array of shape `(N_nodes, D_intensive)` representing intensive potentials (pressure, temperature).
   * `M`: Boolean or Float array of shape `(N_nodes, N_nodes)` representing the physical adjacency matrix (edges/pipes).
   * `edge_controls`: Float array of shape `(N_nodes, N_nodes, D_edge)` for valve states/pipe resistances.
   * `node_controls`: Float array of shape `(N_nodes, D_node)` for heaters, agitators, or reagents.

2. **Create `flux/core/attention.py`:**
   Implement `PortHamiltonianAttention(eqx.Module)`. This handles **Transport** (inter-node flow).
   * **Inputs:** `P` (Intensive potentials), `M` (Adjacency), `edge_controls` (Valves).
   * **Mechanism:**
     * Project `edge_controls` into a conductance matrix `C = Softplus(Linear(edge_controls))`.
     * Calculate potential differences: `DeltaP = P[:, None, :] - P[None, :, :]`.
     * Calculate unnormalized flux: `F = C * M * DeltaP`.
     * **The Skew-Symmetric Guarantee:** `A = F - F.transpose(0, 1, 2)`.
     * Calculate transport rates: `transport_dX_dt = sum(A, axis=1)`.
   * **Output:** `transport_dX_dt` (shape: `(N_nodes, D_extensive)`).

3. **Create `flux/core/dynamics.py`:**
   Implement `NodeDynamicsNetwork(eqx.Module)`. This handles **Transformation** (intra-node reactions, microbiology, heating).
   * **Inputs:** `X` (Extensive states), `P` (Intensive potentials), `node_controls` (Heaters, chillers).
   * **Mechanism:**
     * Pass `[X, P, node_controls]` through a localized MLP.
     * Output must represent the generation/consumption rates of mass/energy within the tank.
   * **Output:** `source_dX_dt` (shape: `(N_nodes, D_extensive)`).

4. **Create `flux/core/model.py`:**
   Implement `PlantGraphModel(eqx.Module)` to tie it all together.
   * **Call Method:** `total_dX_dt = self.attention(...) + self.dynamics(...)`.

---

## Step 4: Write the Mathematical Validation Tests

1. **Create `tests/test_attention.py`:**
   Write a rigorous `pytest` suite for `PortHamiltonianAttention`.
   * **Test 1 (The First Law):** Initialize a random 5-node graph with random potentials and a random adjacency matrix. Pass it through `PortHamiltonianAttention`. Assert that `jnp.sum(dX_dt) == 0.0` (up to floating-point tolerance). This proves that the architecture literally cannot create or destroy mass.
   * **Test 2 (Topology Masking):** Ensure that if `M[i, j] == 0`, no flow occurs between node `i` and node `j`, regardless of their pressure differences.

---

## Completion Criteria
When these steps are complete, the `flux-attention-engine` repository should be fully initialized, installable, and have a passing `pytest` suite proving the absolute mathematical conservation of the Port-Hamiltonian Attention mechanism.