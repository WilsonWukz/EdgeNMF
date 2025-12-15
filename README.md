# EdgeNMF: Lightweight and Noise-Resilient Feature Extraction for Edge Computing

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Preprint](https://img.shields.io/badge/Submitted%20to-Sensors-red)](https://www.mdpi.com/journal/sensors)

![L1_IRLS](https://github.com/user-attachments/assets/d263e0f9-a65b-4454-8df2-c1cbcfc53bf7)


This repository contains the **official implementation** of the paper:

> **"Lightweight and Noise-Resilient Feature Extraction for Edge Computing: A Comparative Study of Robust NMF Algorithms"**
> *Submitted to Sensors (2025)*

This project provides a comprehensive benchmark of Non-negative Matrix Factorization (NMF) algorithms tailored for **resource-constrained edge devices** (e.g., IoT sensors, Raspberry Pi). It systematically evaluates robustness against **impulsive sensor noise** (salt-and-pepper) and computational efficiency.

My implementation highlights the superiority of **$L_1$-Regularized NMF (IRLS)**, which offers a **35% reduction in memory usage** and a **"Robust Plateau"** under moderate noise, making it the optimal choice for Edge AI.

---

## 📌 Key Features & Findings

This project focuses on three critical dimensions of Edge AI:

1.  **Robustness Analysis (Fig. 1):** Demonstrating that $L_1$-NMF maintains a "Robust Plateau" ($0.1 \le p \le 0.3$), significantly outperforming standard $L_2$ baselines which collapse under sensor noise.
2.  **Computational Efficiency (Fig. 2):** Proving that $L_1$-NMF is not only robust but also lightweight, achieving inference speeds comparable to fast baselines while reducing peak memory usage by ~35%.
3.  **The "Edge Sweet Spot" (Fig. 3):** Visualizing the trade-off between accuracy and latency. $L_1$-NMF is identified as the only algorithm that balances high noise resilience with low computational cost.
4.  **Hardware-Aware Profiling:** Integrated `tracemalloc` and timer utilities to accurately simulate edge hardware constraints.
<img width="3000" height="1800" alt="Fig1_Robustness" src="https://github.com/user-attachments/assets/8319efd8-acb2-4aa5-bf20-84837a3e7e28" />

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/WilsonWukz/EdgeNMF.git](https://github.com/WilsonWukz/EdgeNMF.git)
    cd EdgeNMF
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (Conda or venv).
    ```bash
    pip install -r requirements.txt
    ```
    *Core dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `openpyxl`.*

3.  **Prepare Datasets:**
    - Place the **ORL** dataset in `data/ORL/`.
    - Place the **Extended Yale B** dataset in `data/YaleB/`.
    *(Note: The code includes a preprocessing pipeline to auto-resize and inject noise.)*

---

## 🚀 Usage Guide: The Reproduction Pipeline

To fully reproduce the paper's results, follow this pipeline: **Run Experiments -> Aggregate Data -> Visualize**.

### Step 1: Run NMF Benchmarks
This script runs 5 NMF variants (L2, MU, L1, HC, StackMU) across varying noise levels ($p \in [0, 0.5]$) and random seeds. It records Accuracy (ACC), NMI, Time, and Peak Memory.

```bash
# Run the main benchmark loop
# This will generate Excel files in the 'results/' directory
python main_benchmark.py
````

> **Note:** The script automatically handles noise injection and data preprocessing (`reduce=3` for edge simulation).

### Step 2: Data Aggregation

After the experiments finish, aggregate the raw logs into a summary table for plotting.

```bash
python aggregate_results.py
```

  * **Output:** `ALL_ALGO_SUMMARY.xlsx` containing mean/std for all metrics.

### Step 3: Visualization (Reproducing Paper Figures)

Generate the three key figures used in the "Experimental Results" section.

```bash
python plot_figures.py
```

  * **Outputs:**
      * `Fig1_Robustness.png`: ACC vs. Noise Level (Highligting the robust plateau).
      * `Fig2_Efficiency.png`: Bar charts for Inference Time and Peak Memory.
      * `Fig3_Tradeoff.png`: The Scatter plot showing the "Edge Sweet Spot".

-----

## 📂 Repository Structure

  * `algorithms/`: Core implementations of NMF variants.
      * `l1_nmf.py`: **Our proposed robust L1-NMF (IRLS).**
      * `l2_nmf.py`: Standard Gradient Descent NMF.
      * `mu_nmf.py`: Lee-Seung Multiplicative Update.
      * `hc_nmf.py`: Hypersurface Cost NMF.
      * `stack_nmf.py`: Stacked Multiplicative Updates.
  * `utils/`: Helper functions.
      * `dataloader.py`: Handles image loading, resizing, and salt-and-pepper noise injection.
      * `metrics.py`: Evaluates ACC, NMI, and RRE.
  * `main_benchmark.py`: Main entry point for running experiments.
  * `plot_figures.py`: Script to reproduce paper figures.
  * `results/`: Directory for storing experimental logs and Excel files.
  * `data/`: Directory for raw datasets.

-----

## 📜 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{EdgeNMF2025,
  title={Lightweight and Noise-Resilient Feature Extraction for Edge Computing: A Comparative Study of Robust NMF Algorithms},
  author={Wu, Kezhao and Li, Jiayi and Yan, Yaodong and Cai, Ruilin},
  journal={Submitted to Sensors},
  year={2025}
}
```
*Maintained by [WilsonWukz](https://www.google.com/search?q=https://github.com/WilsonWukz).*
