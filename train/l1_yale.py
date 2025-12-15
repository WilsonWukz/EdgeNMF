import numpy as np
import pandas as pd

try:
    import cupy as cp
    gpu_available = True
except ImportError:
    gpu_available = False

from load_data import load_data
from noise import load_data_salt_pepper
from measure_performance import measure_performance
from nmf import L1_nmf_corrected
from evaluation import relative_reconstruction_error, evaluate_clustering

Algo = 'L1'
Data = 'Yale'
dataset = '../data/CroppedYaleB'
reduce = 3
X_clean, Y = load_data(dataset, reduce=reduce)
eps = 1e-8
rank = 40
steps = 500
lambdas = [(0.0,0.0), (0.01,0.005)]
kmeans_init = 5
verbose_nmf = False
seeds = [0,1,2,3,4]
pr_lst = [(0,0), (0.1,0.1), (0.2,0.2), (0.3,0.3), (0.4,0.4), (0.5,0.5)]
tol = 1e-3
record_L1_Yale = []

for seed in seeds:
    np.random.seed(seed)
    if gpu_available:
        cp.random.seed(seed)

    for lambda_h, lambda_w in lambdas:
        for p, r in pr_lst:
            X_noisy, Y = load_data_salt_pepper(dataset, reduce, p, r)

            if gpu_available:
                X_train = cp.asarray(X_noisy)
            else:
                X_train = X_noisy

            (W, H, WH, step), timeusing, peak_mem = measure_performance(
                L1_nmf_corrected,
                X_train,
                rank=rank,
                steps=steps,
                tol=tol,
                lambda_h=lambda_h,
                lambda_w=lambda_w,
                eps=eps
            )

            if gpu_available:
                if hasattr(H, 'get'): H = H.get()
                if hasattr(W, 'get'): W = W.get()
                if hasattr(WH, 'get'): WH = WH.get()

            rre = relative_reconstruction_error(X_noisy, WH)
            acc, nmi = evaluate_clustering(H, Y)
            record_L1_Yale.append(((p,r), (lambda_h,lambda_w), acc, nmi, rre, step, seed, timeusing, peak_mem, Algo, Data))

cols = ['(p,r)', 'lambdas', 'acc', 'nmi', 'rre', 'step', 'seed', 'time', 'memory', 'Algo', 'Data']
df_L1_Yale = pd.DataFrame(record_L1_Yale, columns=cols)

summary_lambdas = df_L1_Yale.groupby(['(p,r)', 'lambdas']).agg(
    ACC_mean=('acc', 'mean'),
    Time_mean=('time', 'mean'),
    Memory_mean=('memory', 'mean')
).reset_index()
print(summary_lambdas)

summary = df_L1_Yale.groupby(['(p,r)']).agg(
    ACC_mean=('acc', 'mean'),
    ACC_std=('acc', 'std'),
    NMI_mean=('nmi', 'mean'),
    NMI_std=('nmi', 'std'),
    RRE_mean=('rre', 'mean'),
    RRE_std=('rre', 'std'),
    Time_mean=('time', 'mean'),
    Time_std=('time', 'std'),
    Memory_mean=('memory', 'mean'),
    Memory_std=('memory', 'std'),
    Algo=('Algo', 'first'),
    Data=('Data', 'first')
).reset_index()
print("\n=== Experimental Summary (with Edge Metrics) ===")
print(summary)
df_L1_Yale.to_excel(f'{Algo}_{Data}_Raw.xlsx', index=True)
summary.to_excel(f'{Algo}_{Data}_Summary.xlsx', index=False)
print(f"\n[Success] Results saved to '{Algo}_{Data}_Raw.xlsx' and '{Algo}_{Data}_Summary.xlsx'")