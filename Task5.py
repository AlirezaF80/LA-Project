import numpy as np


def LDA(X, k):
    """
    Performs Linear Discriminant Analysis on the data X
    :param X: data matrix (N, d)
    :param k: number of components to keep
    :return: reduced data matrix (N, k)
    """
    # TODO: implement LDA
    # 1. Compute the mean vector of each class
    # 2. Compute the within-class scatter matrix Sw
    # 3. Compute the between-class scatter matrix Sb
    # 4. Compute the eigenvectors of the matrix Sb^-1 Sw
    # 5. Get the first k eigenvectors
    # 6. Project the data onto the new subspace
    # 7. Return the projected data
    means = np.mean(X, axis=0)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[0]):
        Sw += np.dot((X[i] - means).T, (X[i] - means))
    for i in range(X.shape[0]):
        Sb += X.shape[0] * np.dot((means - X[i]).T, (means - X[i]))
    eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    matrix_w = np.hstack((eig_pairs[i][1].reshape(X.shape[1], 1) for i in range(k)))
    return np.dot(X, matrix_w)


if __name__ == '__main__':
    data = np.load('data5d.npz')
    X = data['X']
    y = data['y']
    X_reduced = LDA(X, 2)
    print(X_reduced.shape)
