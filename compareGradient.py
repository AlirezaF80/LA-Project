import numpy as np

from main import compute_dC_dw, compute_dC_dw_numeric, compute_dC_db, compute_dC_db_numeric

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
