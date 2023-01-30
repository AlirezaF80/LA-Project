import numpy as np
from matplotlib import pyplot as plt

from Task1 import train, Phi, C
from utilFunctions import plot_data_2D


def PCA(X, k):
    """
    Perform PCA on the data X, using SVD
    :param X: data matrix, each row is a data point
    :param k: number of components to keep
    :return: reduced data matrix
    """
    N, d = X.shape
    U, S, V = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k])


if __name__ == '__main__':
    raw_data = np.load('data5d.npz')
    X = raw_data['X']  # shape(N, d)
    y = raw_data['y']  # shape(N,)
    print("X.shape = %s, y.shape = %s" % (X.shape, y.shape))

    # reduce the dimensionality of the data to 2D
    X -= np.mean(X, axis=0)
    X_reduced = PCA(X, 2)
    N, d = X_reduced.shape
    print("X_reduced.shape = %s" % (X_reduced.shape,))

    w = np.random.randn(d, 1)  # shape (d, 1)
    b = np.random.randn(1)  # shape (1,)

    # train the model
    w, b = train(X_reduced, y, w, b, learning_rate=0.1, threshold=0.012, num_iterations=200)
    print("Final cost: %f" % C(X_reduced, y, w, b))
    print("Final parameters: w = %s, b = %s" % (w, b))
    plot_data_2D(X_reduced, y, Phi(X_reduced, w, b), w, b, False)
