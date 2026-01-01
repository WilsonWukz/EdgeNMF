# EdgeNMF: Noise-Resilient and Lightweight Feature Extraction for Edge AI

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange)]()

![L1_IRLS](https://github.com/user-attachments/assets/d263e0f9-a65b-4454-8df2-c1cbcfc53bf7)

This repository contains the **official implementation** of the paper:

> **"Noise-Resilient and Lightweight Feature Extraction for Edge AI: A Robust $L_1$-Norm Regularized NMF Framework"** (Under Review, 2025)

## 📖 Overview

Deploying reliable feature extraction on **resource-constrained edge nodes** (e.g., IoT gateways, Raspberry Pi) is challenging due to the dual constraints of **limited memory** and **unpredictable sensor noise** (e.g., impulsive salt-and-pepper noise).

To address this, **we propose a robust $L_1$-Regularized NMF framework** based on the Iteratively Reweighted Least Squares (IRLS) strategy. Unlike standard deep learning models that are computationally prohibitive, or traditional $L_2$-based NMFs that are fragile to outliers, our framework:

1.  **Achieves a "Robust Plateau":** Effectively suppresses sparse outliers under moderate noise levels ($0.1 \le p \le 0.3$).
2.  **Prevents OOM Failures:** Reduces peak memory usage by approximately **35%** compared to standard Multiplicative Update (MU) methods, making it safe for embedded deployment.

This repository provides the complete source code for the proposed framework, along with the reproduction scripts for the comparative experiments against state-of-the-art baselines.

---

## 📌 Key Contributions & Findings



[Image of edge computing architecture diagram]


**1. Robustness Analysis (The "Robust Plateau"):**
Our experiments on ORL and Extended Yale B datasets demonstrate that **$L_1$-NMF (Ours)** maintains high clustering accuracy even when 30% of pixels are corrupted, whereas standard baselines collapse rapidly.

**2. Engineering Efficiency (The Memory Advantage):**
We identify a critical **engineering trade-off**. While the IRLS solver incurs higher inference latency, its implementation significantly optimizes memory allocation. By avoiding the large intermediate matrix operations common in standard MU, our framework prevents **Out-of-Memory (OOM)** crashes on limited-SRAM devices.

**3. The "Edge Sweet Spot":**
**We visualize** the trade-off between accuracy and latency. Our framework is identified as the optimal solution that balances high noise resilience with feasible computational costs for edge gateways.

<img width="100%" alt="Fig1_Robustness" src="https://github.com/user-attachments/assets/8319efd8-acb2-4aa5-bf20-84837a3e7e28" />

---

## 🛠️ Installation

### Prerequisites
* **Hardware:** Recommended 8GB+ RAM for full reproduction (Simulating Edge environment).
* **Python:** 3.9 or higher (Tested on 3.12).

### Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/WilsonWukz/EdgeNMF.git](https://github.com/WilsonWukz/EdgeNMF.git)
    cd EdgeNMF
    ```

2.  **Install dependencies:**
    We recommend using a virtual environment (Conda or venv).
    ```bash
    pip install -r requirements.txt
    ```
    *Core dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `openpyxl`, `tracemalloc`.*

3.  **Prepare Datasets:**
    * Download the **ORL** and **Extended Yale B** datasets.
    * Place them in `data/ORL/` and `data/YaleB/` respectively.
    * *(Note: The raw images will be automatically resized and corrupted by our preprocessing pipeline during runtime.)*

---

## 🚀 Reproduction Pipeline

To fully reproduce the experimental results presented in the manuscript, follow this **Batch Training -> Visualization** pipeline.

### Step 1: Run Benchmarks (Proposed Framework vs. Baselines)
We provide a master automation script (`auto.py`) that sequentially executes the proposed algorithm and all baselines across varying noise levels.

**This script will:**
1.  Inject salt-and-pepper noise ($p \in [0, 0.5]$) with consistent random seeds.
2.  Execute the **Proposed Framework** (`l1_*.py`).
3.  Execute **Baselines** (`l2_*.py`, `mu_*.py`, `hc_*.py`, `stack_*.py`).
5.  **Automatically aggregate** metrics (ACC, NMI, RRE, Time, Memory) into `ALL_ALGO_SUMMARY.xlsx`.

```bash
# Execute the full benchmark pipeline
python auto.py

```

> **Note:** This process simulates extensive edge inference and may take time depending on your hardware.

### Step 2: Generate Paper Figures

Once the data is aggregated, run the visualization script to produce the exact figures used in the submission.

```bash
# Generate Fig 1 (Robustness), Fig 2 (Efficiency), and Fig 3 (Trade-off)
python visualization.py

```

**Outputs:**

* `Fig1_Robustness.png`: Illustrates the superiority of the L1-Norm framework.
* `Fig2_Efficiency.png`: Highlights the 35% memory reduction.
* `Fig3_Tradeoff.png`: Visualizes the "Edge Sweet Spot."

---

## 📂 Repository Structure

The codebase is organized to separate the **Proposed Method** from **Baselines**.

```text
EdgeNMF/

├── visualization.py        # Generates paper-ready figures
├── requirements.txt        # Dependency list
├── noise.py                # Define the noise
├── nmf.py                   # Define the NMFs
├── data/                   # Dataset directory
├── measure_performance.py    # Define Metrics
├── train/
│   ├── auto.py                 # Master script for full reproduction
│   ├── l1_orl.py           # [Ours] L1-Regularized Robust NMF (ORL)
│   ├── l1_yale.py          # [Ours] L1-Regularized Robust NMF (YaleB)
│   ├── l2_*.py             # Standard L2-NMF
│   ├── mu_*.py             # Lee-Seung Multiplicative Update
│   ├── hc_*.py             # Hypersurface Cost NMF
│   └── stack_*.py          # Stacked NMF
│
├── load_data.py        # Noise injection & Preprocessing
└── evaluation.py       # Metrics (ACC, NMI, RRE)

```

*(Note: The actual file structure may vary slightly; the above reflects the logical organization of the experiments.)*

---

## 📜 Citation

If you use this code or our framework in your research, please cite our paper:

```bibtex
@article{Wu2025EdgeNMF,
  title={Noise-Resilient and Lightweight Feature Extraction for Edge AI: A Robust $L_1$-Norm Regularized NMF Framework},
  author={Wu, Kezhao and Li, Jiayi and Yan, Yaodong and Cai, Ruilin},
  journal={Under Review},
  year={2025}
}

```
*Maintained by [WilsonWukz](https://www.google.com/search?q=https://github.com/WilsonWukz).*
