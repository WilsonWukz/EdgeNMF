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
from nmf import L2Lee_Seung
from evaluation import relative_reconstruction_error, evaluate_clustering


Algo = 'MU'
Data = 'ORL'
dataset = '../data/ORL'
reduce = 3
X, Y = load_data(dataset, reduce=reduce)
K = len(set(Y))
seeds = [0,1,2,3,4]
pr_lst = [(0,0),(0.1,0.1),(0.2,0.2),(0.3,0.3),(0.4,0.4),(0.5,0.5)]
record_MU_ORL = []
steps = 10000
tol = 1e-3
epsilon = 1e-9

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

        (W, H, WH, step), timeusing, peak_mem = measure_performance(
            L2Lee_Seung, X_train, K, steps, tol, epsilon, verbose=False
        )

        if gpu_available:
            if hasattr(H, 'get'): H = H.get()
            if hasattr(W, 'get'): W = W.get()
            if hasattr(WH, 'get'): WH = WH.get()

        rre = relative_reconstruction_error(X_noisy, WH)
        acc, nmi = evaluate_clustering(H, Y)
        record_MU_ORL.append(((p,r), acc, nmi, rre, tol, step, seed, timeusing, peak_mem, Algo, Data))

df_MU_ORL = pd.DataFrame(record_MU_ORL, columns=['(p,r)','acc','nmi','rre','tol','step','seed', 'time', 'memory', 'Algo', 'Data'])
print(df_MU_ORL)
summary = df_MU_ORL.groupby(['(p,r)']).agg(
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
df_MU_ORL.to_excel(f'{Algo}_{Data}_Raw.xlsx', index=True)
summary.to_excel(f'{Algo}_{Data}_Summary.xlsx', index=False)
print(f"\n[Success] Results saved to '{Algo}_{Data}_Raw.xlsx' and '{Algo}_{Data}_Summary.xlsx'")