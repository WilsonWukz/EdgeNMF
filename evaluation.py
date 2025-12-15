import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter

def relative_reconstruction_error(X, WH):
    """Compute the relative reconstruction error (RRE)."""
    return np.linalg.norm(X - WH, 'fro') / (np.linalg.norm(X, 'fro') + 1e-12)


def assign_cluster_label(X, Y):
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    return Y_pred


def evaluate_clustering(H, Y_true):
    Y_pred = assign_cluster_label(H.T, Y_true)
    acc = accuracy_score(Y_true, Y_pred)
    nmi = normalized_mutual_info_score(Y_true, Y_pred)
    print('Acc(NMI) = {:.4f} ({:.4f})'.format(acc, nmi))
    return acc, nmi