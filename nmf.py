try:
    import cupy as np
    print("NMF using GPU (CuPy)")
except ImportError:
    import numpy as np
    print("NMF using CPU (NumPy)")

def L2_nmf(X, K, lr=0.001, steps=5000, tol=1e-2, verbose=True):
    """
    Non-negative Matrix Factorization (NMF) for
    the basic L2 (Frobenius) objective function.

    Parameters:
    1. X: Input matrix (M x N), must be non-negative.
    2. K: Rank of factorization (number of latent features).
    3. steps: Maximum number of iterations.
    4. tol: Tolerance for early stopping based on error change.
    5. verbose: If True, prints error every 10 steps.

    Returns:
    W: Basis matrix (M x K).
    H: Coefficient matrix (K x N).
    WH: The final reconstructed matrix W @ H.
    """
    M, N = X.shape
    # X = X / np.linalg.norm(X, "fro")
    W = np.random.rand(M, K) * np.sqrt(np.mean(X))
    H = np.random.rand(K, N) * np.sqrt(np.mean(X))

    tol = tol
    prev_e = 0

    for step in range(steps):
        WH = W @ H
        e = np.linalg.norm(X - WH, "fro") ** 2

        dW = -2 * X @ H.T + 2 * W @ (H @ H.T)
        dH = -2 * W.T @ X + 2 * (W.T @ W) @ H

        W -= lr * dW
        H -= lr * dH

        # enforce non-negativity
        W = np.maximum(W, 0)
        H = np.maximum(H, 0)

        if step % 10 == 0:
            print(f'step:{step}, e:{e:.4f}')
        if step > 0:
            rel_change = abs(e - prev_e) / (prev_e + 1e-12)
            if rel_change < tol:
                print(f'Converged at step {step}, e = {e:.4f}, rel_change={rel_change:.3e}')
                break
        prev_e = e

    return W, H, WH, step + 1


def L2Lee_Seung(X, K, steps=10000, tol=1e-2, epsilon=1e-9, verbose=True):
    """
    Non-negative Matrix Factorization (NMF) using Lee & Seung's
    Multiplicative Update (MU) rules for the L2 (Frobenius) objective function.

    Parameters:
    1. X: Input matrix (M x N), must be non-negative.
    2. K: Rank of factorization (number of latent features).
    3. steps: Maximum number of iterations.
    4. tol: Tolerance for early stopping based on error change.
    5. verbose: If True, prints error every 10 steps.

    Returns:
    W: Basis matrix (M x K).
    H: Coefficient matrix (K x N).
    reconstruction: The final reconstructed matrix W @ H.
    """
    M, N = X.shape
    avg_X = np.mean(X)
    W = np.random.rand(M, K) * np.sqrt(avg_X)
    H = np.random.rand(K, N) * np.sqrt(avg_X)

    prev_e = 0
    epsilon = epsilon
    for step in range(steps):
        WH = W @ H
        e = np.linalg.norm(X - WH, "fro") ** 2

        numerator_H = W.T @ X
        denominator_H = W.T @ W @ H

        epsilon = epsilon
        H *= (numerator_H / (denominator_H + epsilon))

        numerator_W = X @ H.T
        denominator_W = W @ H @ H.T
        W *= (numerator_W / (denominator_W + epsilon))

        if verbose and step % 10 == 0:
            print(f'step:{step}, e:{e:.4f}')

        if step > 0:
            rel_change = abs(e - prev_e) / (prev_e + 1e-12)
            if rel_change < tol:
                print(f'Converged at step {step}, e = {e:.4f}, rel_change={rel_change:.3e}')
                break
        prev_e = e

    return W, H, WH, step + 1


def L1_nmf_corrected(X, rank=40, steps=500, tol=1e-4,
                     lambda_h=0.0, lambda_w=0.0, eps=1e-8):
    m, n = X.shape
    norm_X = np.linalg.norm(X)
    lambda_h = lambda_h / norm_X
    lambda_w = lambda_w / norm_X

    W = np.abs(np.random.randn(m, rank)).astype(np.float32)
    H = np.abs(np.random.randn(rank, n)).astype(np.float32)

    for irls_iter in range(10):
        WH = W @ H
        E_mat = X - WH
        weight = 1.0 / np.sqrt(E_mat ** 2 + eps)

        prev_E = np.inf
        for step in range(steps):
            num_H = W.T @ (weight * X)
            den_H = W.T @ (weight * WH)
            H *= num_H / (den_H + lambda_h + eps)

            WH = W @ H
            num_W = (weight * X) @ H.T
            den_W = (weight * WH) @ H.T
            W *= num_W / (den_W + lambda_w + eps)

            weighted_error = weight * (X - W @ H)
            E_sq = np.sum(weighted_error ** 2)
            if abs(E_sq - prev_E) / (prev_E + 1e-12) < tol:
                break
            prev_E = E_sq

    return W, H, W @ H, step + 1


def Hypersurface_nmf(X, rank=30, steps=100, tol=1e-3, delta=0.1, eps=1e-8, verbose=False):
    """
    Corrected Hypersurface Cost-based NMF:
        f(E) = sum( sqrt(1 + (E/delta)^2) - 1 )
    """
    m, n = X.shape
    W = np.abs(np.random.randn(m, rank)).astype(np.float32)
    H = np.abs(np.random.randn(rank, n)).astype(np.float32)

    prev_cost = float('inf')

    for step in range(steps):
        WH = W @ H
        E = X - WH

        denom = np.sqrt(delta ** 2 + E ** 2)
        weights = 1.0 / (denom + eps)

        num_H = W.T @ (X * weights)
        den_H = W.T @ (WH * weights) + eps
        H *= num_H / den_H

        WH = W @ H
        E = X - WH
        denom = np.sqrt(delta ** 2 + E ** 2)
        weights = 1.0 / (denom + eps)

        num_W = (X * weights) @ H.T
        den_W = (WH * weights) @ H.T + eps
        W *= num_W / den_W

        W = np.clip(W, eps, None)
        H = np.clip(H, eps, None)

        current_cost = np.sum(np.sqrt(1 + (X - W @ H) ** 2 / delta ** 2) - 1)

        if verbose and (step % max(1, steps // 10) == 0 or step == steps - 1):
            print(f"step {step + 1}/{steps} | Cost={current_cost:.6f} | Δ={abs(current_cost - prev_cost):.3e}")

        if step > 0:
            rel_change = abs(current_cost - prev_cost) / (prev_cost + 1e-12)
            if rel_change < tol:
                if verbose:
                    print(f'Converged at step {step} | Cost={current_cost:.4f} | rel_change={rel_change:.3e}')
                break

        prev_cost = current_cost

    return W, H, W @ H, step + 1

def nmf_mu(X, K, steps=200, eps=1e-9, tol=None, verbose=False):
    M, N = X.shape
    avg_X = np.mean(X)
    W = np.random.rand(M, K) * np.sqrt(avg_X)
    H = np.random.rand(K, N) * np.sqrt(avg_X)

    prev_e = 0
    epsilon = eps
    for step in range(steps):
        WH = W @ H
        e = np.linalg.norm(X - WH, "fro") ** 2

        numerator_H = W.T @ X
        denominator_H = W.T @ W @ H

        epsilon = epsilon
        H *= (numerator_H / (denominator_H + epsilon))

        numerator_W = X @ H.T
        denominator_W = W @ H @ H.T
        W *= (numerator_W / (denominator_W + epsilon))

        if verbose and step % 10 == 0:
            print(f'step:{step}, e:{e:.4f}')

        if step > 0:
            rel_change = abs(e - prev_e) / (prev_e + 1e-12)
            if rel_change < tol:
                print(f'Converged at step {step}, e = {e:.4f}, rel_change={rel_change:.3e}')
                break
        prev_e = e

    return W, H, step + 1


def stacked_nmf_pretrain(X, layer_ranks, steps_per_layer=200, tol=1e-3, eps=1e-9, verbose=False):
    Y = X.copy()
    Ws = []
    H_final = None
    for i, K in enumerate(layer_ranks):
        if verbose:
            print(f"Pretraining layer {i + 1}/{len(layer_ranks)}: factorizing matrix shape {Y.shape} -> rank {K}")
        W_i, H_i, step = nmf_mu(Y, K, steps=steps_per_layer, tol=tol, eps=eps, verbose=verbose)
        Ws.append(W_i)
        # for next layer, use H_i as data
        Y = H_i
        H_final = H_i  # final H will be last H_i
    return Ws, H_final, step


def reconstruct_from_stacked(Ws, H):
    """
    Compute full reconstruction WH = W1 W2 ... WL H.
    Ws: list of W_i, shapes must chain: W1(m,K1), W2(K1,K2), ...
    H: final H (K_L, n)
    """
    P = Ws[0]
    for W in Ws[1:]:
        P = P @ W
    return P @ H


def StackedNMF(X, layer_ranks=[10, 6], steps_per_layer=100, tol=1e-3, eps=1e-9):
    Ws, H_final, step = stacked_nmf_pretrain(X, layer_ranks, steps_per_layer=steps_per_layer, tol=tol, eps=eps,
                                             verbose=False)
    PH = reconstruct_from_stacked(Ws, H_final)
    return H_final, PH, step