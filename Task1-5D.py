import numpy as np

from Task1 import train, C, Phi
from utilFunctions import plot_data_3D

if __name__ == '__main__':
    raw_data = np.load('data5d.npz')
    X = raw_data['X']  # shape (N, d)
    y = raw_data['y']  # shape (N,)
    w = np.random.randn(X.shape[1], 1)  # shape (d, 1)
    b = np.random.randn(1)  # shape (1,)

    w, b = train(X, y, w, b, learning_rate=0.1, threshold=0.015, num_iterations=200)
    print("Final cost: %f" % C(X, y, w, b))
    print(f"Final parameters: w = {w}, b = {b}")
    plot_data_3D(X, y, Phi(X, w, b))
