# The Flux Attention Pivot: Execution Plan

## 1. The Strategic Reality (Why We Are Pivoting)
The current `digital-twin-engine` (DTE) architecture relies on padding disparate systems (e.g., CSTR, Two-Tank) into dense, fixed-size tensors (`max_state_dim`) and using soft physics penalties via standard Neural ODEs. 

**Fatal Flaws of DTE:**
1.  **Topological Inflexibility:** It cannot handle arbitrary $N$-node graphs or dynamic recycles without manual, hardcoded dimension matching.
2.  **Physics Leakage:** It relies on optimization constraints (`loss_weights.positivity`, conservation penalties). Over long rollouts on stiff systems, optimization errors accumulate, violating the First Law of Thermodynamics.
3.  **The "Sunk Cost" Trap:** It requires massive engineering overhead (like `convergence_agent.py` and curriculum schedules) just to maintain numerical stability on padded arrays.

**The Solution:** 
We are abandoning the "Universal Padded Backbone" in favor of **Port-Hamiltonian Attention** (PHA)—a graph-native architecture that enforces mass/energy conservation by definition (skew-symmetric attention matrices) and naturally generalizes to unseen graph topologies.

---

## 2. Repository Strategy

*   **Repository 1: `digital-twin-engine` (Frozen / Legacy)**
    *   **Role:** Synthesizer and Ground-Truth Baseline.
    *   **Action:** Cease all training and agentic convergence loops. Preserve the `dte/simulators/` and data generation scripts. We will use these mathematically robust ODEs to generate the training data for the new architecture.
*   **Repository 2: `flux-attention-engine` (Net-New)**
    *   **Role:** The production codebase for the PHA framework and the "AI Plant Compiler."
    *   **Action:** Build from scratch. No legacy padding, no monolithic configuration files.

---

## 3. The 8-Week Execution Roadmap

### Phase 1: Core Mathematical Proof (Weeks 1-2)
*Objective: Build the isolated `PortHamiltonianAttention` layer and mathematically prove 100% conservation.*

*   **Task 1.1:** Initialize the new `flux-attention-engine` repository (JAX, Equinox, Diffrax).
*   **Task 1.2:** Define the Graph Representation. States are strictly divided into Extensive variables ($X$: mass, energy) and Intensive variables ($P$: pressure, temperature, concentration).
*   **Task 1.3:** Implement the core architectural split: **Transport vs. Transformation**.
    *   **The PHA Layer (Transport):** $A = \sigma(C \odot M \cdot (P_i - P_j) - [C \odot M \cdot (P_i - P_j)]^T)$. Computes `transport_dX_dt` (routes flow between nodes, controlled by edge features like valves).
    *   **The Node Dynamics Layer (Transformation):** An MLP that computes `source_dX_dt` (reactions, microbiological growth, adsorption within a node, controlled by node features like heaters).
    *   **Total Update:** $\frac{dX}{dt} = \text{transport\_dX\_dt} + \text{source\_dX\_dt}$.
*   **Task 1.4:** Build the "Hello World" 2-node experiment. Prove fluid routes from Node A to Node B with `0.0%` mass drift over 10,000 steps.

### Phase 2: Data Generation & Zero-Shot Scale (Weeks 3-4)
*Objective: Train the engine to route flow across graphs and prove it generalizes to larger, unseen graphs.*

*   **Task 2.1 (In DTE repo):** Write a multi-tank network generator utilizing the existing `TwoTankSimulator` logic, but expanded to a randomized 5-Node chaotic graph. Generate 10,000 trajectories.
*   **Task 2.2:** Build the lightweight Graph DataLoader in the new repo (returning Node Features, Edge Features, Adjacency $M$).
*   **Task 2.3:** Train the PHA engine on the 5-node dataset.
*   **Task 2.4 (The NeurIPS Result):** At inference time, construct a 50-node graph adjacency matrix. Feed it to the trained model. Validate that the PHA engine accurately routes flow zero-shot across the larger topology.

### Phase 3: The Chemistry & Ingestion MVP (Weeks 5-6)
*Objective: Remove manual physics definitions and manual adjacency matrix construction.*

*   **Task 3.1:** Implement the Multi-Modal VLM ingestion pipeline. Write prompts instructing a VLM (Claude 3.5 / GPT-4o) to trace a PDF P&ID and output a standardized JSON representing the nodes and edges.
*   **Task 3.2:** Write the Python parser that converts the VLM JSON directly into the JAX Adjacency Matrix $M$.
*   **Task 3.3:** Implement a simplified Chemistry Embedding layer. Create a dictionary mapping basic SMILES strings (e.g., Water, Methane) to fixed intensive property offsets.

### Phase 4: The "AI Plant Compiler" Demo (Weeks 7-8)
*Objective: Build the 60-second "Jaw-Dropping" web interface for commercial pitching.*

*   **Task 4.1:** Stand up a FastAPI backend that wraps the VLM parser and the JAX PHA engine.
*   **Task 4.2:** Build a React/Vite frontend with a drag-and-drop PDF upload zone.
*   **Task 4.3:** Integrate an interactive graph visualizer (e.g., React Flow).
*   **Task 4.4:** Wire the graph UI to the simulation output. Allow the user to visually "close" an edge (valve) in the UI, and watch the dynamic pressure/flow response propagate through the plant in real-time.

---

## 4. Immediate Next Steps (Day 1 Actions)

1.  Halt all running training loops in `digital-twin-engine`.
2.  Initialize the new repository `flux-attention-engine`.
3.  Write `flux_attention_engine/core/attention.py`. Do not write training boilerplate; write the core skew-symmetric math and a unit test proving it does not leak mass.