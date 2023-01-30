import numpy as np

from utilFunctions import plot_data_3D, training_error


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Phi(X, w, b):
    return sigmoid(np.dot(X, w) + b).reshape(-1)


def C(X, y, w, b):
    N = len(y)
    y_pred = Phi(X, w, b)
    A = y * np.log(y_pred)
    B = (1 - y) * np.log(1 - y_pred)
    cost = -1 / N * np.sum(A + B)
    return cost


def compute_dC_dw(X, y, w, b):
    N = len(y)
    A = Phi(X, w, b)
    dC_dw = 1 / N * np.dot(X.T, A - y)
    return dC_dw.reshape(w.shape)


def compute_dC_db(X, y, w, b):
    N = len(y)
    A = Phi(X, w, b)
    dC_db = 1 / N * np.sum(A - y)
    return dC_db


def train(X, y, w, b, learning_rate, threshold, num_iterations):
    for i in range(num_iterations):
        dC_dw = compute_dC_dw(X, y, w, b)
        dC_db = compute_dC_db(X, y, w, b)
        w = w - learning_rate * dC_dw
        b = b - learning_rate * dC_db
        if i % 20 == 0:
            plot_data_3D(X, y, Phi(X, w, b), True)
            print(f"Cost after iteration {i}: {C(X, y, w, b)}")
            print(f"Training error after iteration {i}: {training_error(y, Phi(X, w, b))}")
        if np.linalg.norm(dC_dw) < threshold:
            break
    return w, b
