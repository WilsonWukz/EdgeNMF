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
from nmf import StackedNMF
from evaluation import relative_reconstruction_error, evaluate_clustering

Algo = 'StackedNMF'
Data = 'YaleB'
dataset = '../data/CroppedYaleB'
reduce = 3
X, Y = load_data(dataset, reduce=reduce)
seeds = [0,1,2,3,4]
pr_lst = [(0,0),(0.1,0.1),(0.2,0.2),(0.3,0.3),(0.4,0.4),(0.5,0.5)]
record_SNMF_Yale = []
layer_ranks = [10, 6]
steps_per_layer = 10000
tol = 1e-3
eps = 1e-9

for seed in seeds:
    np.random.seed(seed)
    if gpu_available:
        cp.random.seed(seed)

    for p, r in pr_lst:
        X_noisy, Y = load_data_salt_pepper(dataset, reduce, p, r)

        if gpu_available:
            X_train = cp.asarray(X_noisy)
        else:
            X_train = X_noisy

        (H_final, PH, step), timeusing, peak_mem = measure_performance(
            StackedNMF, X_train, layer_ranks=layer_ranks, steps_per_layer=steps_per_layer, tol=tol, eps=eps
        )

        if gpu_available:
            if hasattr(H_final, 'get'): H_final = H_final.get()
            if hasattr(PH, 'get'): PH = PH.get()

        rre = relative_reconstruction_error(X_noisy, PH)
        acc, nmi = evaluate_clustering(H_final, Y)
        record_SNMF_Yale.append(((p,r), acc, nmi, rre, tol, step, seed, timeusing, peak_mem, Algo, Data))

df_SNMF_Yale = pd.DataFrame(record_SNMF_Yale, columns=['(p,r)','acc','nmi','rre','tol','step','seed', 'time', 'memory', 'Algo', 'Data'])
print(df_SNMF_Yale)
summary = df_SNMF_Yale.groupby(['(p,r)']).agg(
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

df_SNMF_Yale.to_excel(f'{Algo}_{Data}_Raw.xlsx', index=True)
summary.to_excel(f'{Algo}_{Data}_Summary.xlsx', index=False)
print(f"\n[Success] Results saved to '{Algo}_{Data}_Raw.xlsx' and '{Algo}_{Data}_Summary.xlsx'")