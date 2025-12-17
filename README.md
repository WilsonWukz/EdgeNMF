# EdgeNMF: Lightweight and Noise-Resilient Feature Extraction for Edge Computing

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange)]()

![L1_IRLS](https://github.com/user-attachments/assets/d263e0f9-a65b-4454-8df2-c1cbcfc53bf7)

This repository contains the **official implementation** of the paper:

> **"Lightweight and Noise-Resilient Feature Extraction for Edge Computing: A Comparative Study of Robust NMF Algorithms"** (2025)

In this work, **we present** a comprehensive benchmark of Non-negative Matrix Factorization (NMF) algorithms tailored for **resource-constrained edge devices** (e.g., IoT sensors, Raspberry Pi). **We systematically evaluate** robustness against **impulsive sensor noise** (salt-and-pepper) and computational efficiency.

**We highlight** the superiority of **$L_1$-Regularized NMF (IRLS)**, demonstrating that it offers a **35% reduction in memory usage** and maintains a **"Robust Plateau"** under moderate noise, making it the optimal choice for Edge AI.

---

## 📌 Key Features & Findings

**We focus** our analysis on three critical dimensions of Edge AI:

1.  **Robustness Analysis (Fig. 1):** **We demonstrate** that $L_1$-NMF maintains a "Robust Plateau" ($0.1 \le p \le 0.3$), significantly outperforming standard $L_2$ baselines which collapse under sensor noise.
2.  **Computational Efficiency (Fig. 2):** **We prove** that $L_1$-NMF is not only robust but also lightweight, achieving inference speeds comparable to fast baselines while reducing peak memory usage by ~35%.
3.  **The "Edge Sweet Spot" (Fig. 3):** **We visualize** the trade-off between accuracy and latency. **We identify** $L_1$-NMF as the only algorithm that balances high noise resilience with low computational cost.
4.  **Hardware-Aware Profiling:** **We integrate** `tracemalloc` and timer utilities to accurately simulate edge hardware constraints.

<img width="3000" height="1800" alt="Fig1_Robustness" src="https://github.com/user-attachments/assets/8319efd8-acb2-4aa5-bf20-84837a3e7e28" />

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/WilsonWukz/EdgeNMF.git](https://github.com/WilsonWukz/EdgeNMF.git)
    cd EdgeNMF
    ```

2.  **Install dependencies:**
    **We recommend** using a virtual environment (Conda or venv).
    ```bash
    pip install -r requirements.txt
    ```
    *Core dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `openpyxl`.*

3.  **Prepare Datasets:**
    - Place the **ORL** dataset in `data/ORL/`.
    - Place the **Extended Yale B** dataset in `data/YaleB/`.
    *(Note: Our pipeline includes preprocessing steps to auto-resize and inject noise.)*

---

## 🚀 Usage Guide: The Reproduction Pipeline

To fully reproduce **our results**, follow this pipeline: **Run Experiments -> Aggregate Data -> Visualize**.

### Step 1: Run NMF Benchmarks
This script runs 5 NMF variants (L2, MU, L1, HC, StackMU) across varying noise levels ($p \in [0, 0.5]$) and random seeds. It records Accuracy (ACC), NMI, Time, and Peak Memory.

```bash
# Run the main benchmark loop
# This will generate Excel files in the 'results/' directory
python main_benchmark.py
