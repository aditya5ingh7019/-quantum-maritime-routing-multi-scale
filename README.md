# Maritime TSP — QAOA-Assisted Routing (Multi-Scale Study)

This repository contains the research implementation for:

**"Quantum Coverage and Solution Quality in QAOA-Assisted Greedy Maritime Routing: A Multi-Scale Empirical Study"**

---

## Authors

* Aditya Singh
* Rajiv Pandey
* Pooja Srivastava

---

## Overview

This project investigates how **quantum influence coverage** affects solution quality in hybrid quantum-classical optimization.

We implement a **QAOA-assisted greedy TSP solver** using real maritime distances and evaluate performance under:

| Regime | Ports | MAX_QUBITS | Quantum Coverage |
|--------|-------|------------|------------------|
| **Full coverage** | 8 | 8 | 100% |
| **Full coverage** | 12 | 12 | 100% |
| **Partial coverage** | 16 | 8 | 53.3% |
| **Partial coverage** | 20 | 8 | 42.1% |

---

## Core Contribution

Traditional hybrid quantum algorithms are often evaluated only on final solution quality.

This work introduces a **new experimental lens**:

> **Quantum Influence Coverage (%) vs Solution Quality**

We empirically show how **partial quantum participation (due to qubit limits)** impacts optimization performance across **four distinct scales**.

---

## Core Results (QUALITY Run)

### 8-Port Regime (Full Quantum Coverage)
| Metric | Value |
|--------|-------|
| QAOA Mean Cost | ~54,896 km |
| Approximation Ratio vs LKH-3 | **1.282** |
| Quantum Influence | **100%** |

### 12-Port Regime (Full Quantum Coverage)
| Metric | Value |
|--------|-------|
| QAOA Mean Cost | ~81,708 km |
| Approximation Ratio vs LKH-3 | **1.631** |
| Quantum Influence | **100%** |

### 16-Port Regime (Partial Quantum Coverage)
| Metric | Value |
|--------|-------|
| QAOA Mean Cost | ~101,793 km |
| Approximation Ratio vs LKH-3 | **1.906** |
| Quantum Influence | **53.3%** |

### 20-Port Regime (Partial Quantum Coverage)
| Metric | Value |
|--------|-------|
| QAOA Mean Cost | ~128,952 km |
| Approximation Ratio vs LKH-3 | **2.013** |
| Quantum Influence | **42.1%** |

---

## Extended Studies

### 20-Port Study with K=16 Qubits (Higher Quantum Capacity)

A side study was conducted to evaluate performance with increased qubit capacity (16 qubits) on the 20-port network.

**Configuration:**
- max_qubits: 16 | layers: 3 | steps: 100 | alpha: 0.5 | n_ports: 20
- seeds: [42, 123, 456, 789] | Quantum Influence: 100% (16/16 ports covered)

**Execution log (4 seeds, ~6 min each, ~25 min total):**

- seed=42: 148,548 km | ratio=2.3186 | Q=16 C=3 | 7.8 min
- seed=123: 127,940 km | ratio=1.9970 | Q=16 C=3 | 5.9 min ← Best
- seed=456: 147,099 km | ratio=2.2960 | Q=16 C=3 | 6.2 min
- seed=789: 133,038 km | ratio=2.0765 | Q=16 C=3 | 5.9 min

**Key Results (20 ports, K=16):**

| Metric | Value |
|--------|-------|
| QAOA-Greedy Mean | 139,156 ± 8,868 km |
| CV | 6.4% |
| Best seed | 127,940 km |
| Worst seed | 148,548 km |
| Mean ρ vs LKH-3 | 2.1720 |
| LKH-3 baseline | 64,067 km |
| Quantum Influence | 100% |

---

**Port dataset (30 ports):**
- **Original 20:** Mumbai, Chennai, Kolkata, Kochi, Visakhapatnam, Goa, Tuticorin, Singapore, Colombo, Jebel Ali, Port Klang, Shanghai, Busan, Rotterdam, Fujairah, Port Hedland, Yokohama, Durban, Sydney, Hamburg
- **New 10:** Cape Town, Lagos, Mombasa, Los Angeles, Houston, Santos, Antwerp, Piraeus, New York, Tokyo

**Configuration:**
- max_qubits: 20 | layers: 3 | steps: 100 | alpha: 0.5 | n_ports: 30
- seeds: [42, 123, 456, 789] | Expected Φ_Q: ~66.7%

**Key Results (30 ports, K=20):**

| Metric | Value |
|--------|-------|
| Nearest Neighbour | 107,157 km |
| HC 2-opt | 93,439 km |
| Or-opt / 3-opt | 93,439 km |
| QAOA-Greedy Mean | 251,684 ± 18,032 km |
| Best seed (123) | 229,059 km |
| Mean ρ vs Or-opt | 2.6936 |
| Quantum Influence | 69.0% |

---

## Figures

**Important:** Not all figures are used from this main cell due to labelling issues, but they are regenerated using real values and saved JSON file.

- `route_map_12port.png` - Used from the current cell main run (maritime_tsp_multi_scale)
- `ablation_12port.png` - Used from the current cell main run (maritime_tsp_multi_scale)
- `p_layer_study_8port` - Figure used in the paper from the third last cell titled: "P-LAYER STUDY" (maritime_tsp_multi_scale)
- `coverage_quality_curve.png` - Used in the paper from last cell run title: "Paper Figures Generation Script for Maritime TSP with QAOA-Greedy with added 30 ports and 20 qubit side study from notebook `maritime_30port_20qubit_side_study` (maritime_tsp_multi_scale)
- `noise_12port.png` - Used in the paper with fixed labels from the next cell Complete Figure and Table Regeneration Script (maritime_tsp_multi_scale)
- `all_regimes_boxplot.png` - Used in the paper with fixed labels from the next cell titled: "Complete Figure and Table Regeneration Script" (maritime_tsp_multi_scale)
- `ablation_12port.png` - Used in the paper with fixed labels from the last cell run title: "Paper Figures Generation Script for Maritime TSP with QAOA-Greedy
Fixed version for all figures including noise plots" (maritime_tsp_multi_scale)

---

## Central Finding

> **Higher quantum coverage → Better approximation ratios**

The relationship is **linear** (R² > 0.95), demonstrating that quantum influence is a key driver of hybrid algorithm performance.

| Quantum Coverage | Approximation Ratio |
|-----------------|-------------------|
| 100% (8-port) | 1.282 |
| 100% (12-port) | 1.631 |
| 53.3% (16-port) | 1.906 |
| 42.1% (20-port) | 2.013 |

---

## Features

* Hybrid QAOA + Greedy TSP construction
* Real-world maritime distances (via `searoute`)
* **Four experimental regimes** (8, 12, 16, 20 ports)
* Classical baselines:
  * Nearest Neighbor
  * 2-opt Hill Climbing
  * LKH-3 (optional)
* Comprehensive experiments:
  * Ablation study (layers × alpha)
  * Statistical evaluation (multi-seed, 8-16 seeds)
  * Noise sensitivity analysis (p = 0.001 to 0.05)
  * Qubit-cap sensitivity (MAX_QUBITS = 4, 6, 8, 10, 12)
* **Publication-ready plots** (11 figure types)

---

## Installation

```bash
pip install -r requirements.txt

## Usage

Run Full Experiment (QUALITY Preset):

```bash
python maritime_tsp_multiscale.py
```

Quick Test Mode (FAST Preset):

```bash
python maritime_tsp_multiscale.py --preset FAST
```

Skip 20-Port Regime:

```bash
python maritime_tsp_multiscale.py --skip_20port
```
Run High-Level Experiments:

```bash
python maritime_tsp_multiscale.py --preset FINAL
```
## Preset Modes

| Mode     | Seeds | Steps (Main) | Noise Study | Estimated Runtime | Purpose                  |
|----------|-------|--------------|-------------|-------------------|--------------------------|
| **FAST** | 2     | 50           | Skipped     | ~30 minutes       | Quick testing & debugging |
| **QUALITY** | 8  | 150          | Included    | ~70 minutes       | Research / Paper results |
| **FINAL**   | 16 | 300          | Included    | ~14 hours         | Publication-ready        |

## Output Files

All outputs are saved in the **current working directory**.

### Figures (30+ files)

| Figure Type                    | Files                          | Purpose                              |
|-------------------------------|--------------------------------|--------------------------------------|
| Route Maps                    | `route_map_*.png` (4)          | Geographic route visualization      |
| Algorithm Comparison          | `algo_comparison_*.png` (4)    | Baseline performance comparison     |
| Ablation Heatmaps             | `ablation_*.png` (4)           | Hyperparameter sensitivity          |
| Statistical Distributions     | `stats_*.png` (4)              | Seed variability                    |
| Noise Sensitivity             | `noise_*.png` (4)              | Hardware noise impact               |
| Qubit Cap Sensitivity         | `qcap_*.png` (4)               | Budget constraints                  |
| Multi-Regime Summary          | `multi_regime_summary.png`     | All algorithms comparison           |
| All-Regimes Boxplot           | `all_regimes_boxplot.png`      | Cost distributions                  |
| Coverage Evolution            | `coverage_evolution.png`       | Quantum vs Classical decisions      |
| Qubit Cap Combined            | `qubit_cap_sensitivity.png`    | Combined scaling analysis           |
| **Coverage-Quality Curve**    | `coverage_quality_curve.png`   | **Central paper figure** ⭐          |

### CSV Tables (20+ files)

| File Type                  | Count | Description                          |
|---------------------------|-------|--------------------------------------|
| `baselines_*.csv`         | 4     | Classical baseline results           |
| `ablation_*.csv`          | 4     | Hyperparameter search                |
| `stats_*.csv`             | 4     | Per-seed statistical records         |
| `noise_*.csv`             | 4     | Noise sensitivity data               |
| `qcap_*.csv`              | 4     | Qubit cap sensitivity                |

### Master CSV Files

| File                            | Description                                      |
|--------------------------------|--------------------------------------------------|
| `cross_regime_summary.csv`     | Comparison across all regimes                    |
| `statistical_tests.csv`        | T-test results                                   |
| `hyperparameter_optimal.csv`   | Best hyperparameter configurations               |
| `regime_runtimes.csv`          | Wall-clock execution times                       |
| `all_ablation_results.csv`     | Combined hyperparameter search                   |
| `all_statistical_records.csv`  | Raw seed data                                    |
| `coverage_quality_points.csv`  | Raw data for central figure                      |

### JSON Summary
- `maritime_tsp_results_*.json` — Complete experiment results in JSON format

---

## Methodology (Simplified)

1. **Build distance matrix** using real navigable sea routes via the `searoute` library.

2. **Construct TSP tour incrementally** using a hybrid greedy approach:
   - Use **QAOA** (XY mixer) for probabilistic port selection when the number of remaining ports ≤ `MAX_QUBITS`
   - Fall back to **Classical Nearest Neighbor** when the candidate set exceeds the qubit budget

3. **Comprehensive evaluation** across:
   - Multiple random seeds (8–16)
   - Noise levels (`p = 0.001` to `0.05`)
   - Qubit budgets (4 to 12)
   - Hyperparameters (QAOA layers `p=1–4`, look-ahead weight `α=0.0–0.7`)

---

## Technologies Used

| Technology          | Purpose                          |
|---------------------|----------------------------------|
| **Python**          | Core language                    |
| **PennyLane**       | Quantum circuit simulation       |
| **PennyLane-Lightning** | Fast statevector simulation   |
| **NumPy / SciPy**   | Numerical computing & statistics |
| **Matplotlib**      | Visualization & plotting         |
| **Pandas**          | Data processing & CSV export     |
| **searoute**        | Real maritime sea-route distances|
| **LKH-3**           | High-quality classical baseline  |

---

## Limitations

- Simulation-based (no real quantum hardware used — synthetic depolarizing noise is added)
- QAOA circuit depth is limited due to NISQ constraints (`p ≤ 8`)
- Runtime scales exponentially with the number of ports/qubits
- Currently tested only up to 20 ports due to classical simulation memory limits

---

## Future Work

- Execution on real quantum hardware (IBM Quantum, IonQ, etc.)
- Exploration of improved mixers and ansatz designs
- Integration with real-world maritime logistics optimization systems
- Scaling to larger port networks (>20 nodes)
- Dynamic and adaptive qubit allocation strategies

---

## Citation

```bibtex
@article{singh2024quantum,
  title={Quantum Coverage and Solution Quality in QAOA-Assisted Greedy Maritime Routing: A Multi-Scale Empirical Study},
  author={Singh, Aditya and Pandey, Rajiv and Srivastava, Pooja},
  year={2024}
}

```

---

## License

MIT License

---

## Contact

For questions or collaboration:

* GitHub Issues
* LinkedIn (Aditya Singh)
