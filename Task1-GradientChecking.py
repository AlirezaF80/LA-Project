import numpy as np

from Task1 import compute_dC_dw, compute_dC_db, C


def compute_dC_dw_numeric(X, y, w, b, epsilon=1e-6):
    dC_dw = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += epsilon
        w_n[i] -= epsilon
        dC_dw[i] = (C(X, y, w_p, b) - C(X, y, w_n, b)) / (2 * epsilon)
    return dC_dw


def compute_dC_db_numeric(X, y, w, b, epsilon=1e-6):
    return (C(X, y, w, b + epsilon) - C(X, y, w, b - epsilon)) / (2 * epsilon)


data = np.load('data5d.npz')
X = data['X']
y = data['y']
w = np.ones((X.shape[1], 1))
b = 0

analytic_dC_dw = compute_dC_dw(X, y, w, b)
numeric_dC_dw = compute_dC_dw_numeric(X, y, w, b)
print('dC/dw absolute error = ', np.linalg.norm(analytic_dC_dw - numeric_dC_dw))
print('dC/dw relative error = ', np.linalg.norm(analytic_dC_dw - numeric_dC_dw) / np.linalg.norm(numeric_dC_dw))

analytic_dC_db = compute_dC_db(X, y, w, b)
numeric_dC_db = compute_dC_db_numeric(X, y, w, b)
print('dC/db absolute error = ', np.linalg.norm(analytic_dC_db - numeric_dC_db))
print('dC/db relative error = ', np.linalg.norm(analytic_dC_db - numeric_dC_db) / np.linalg.norm(numeric_dC_db))
