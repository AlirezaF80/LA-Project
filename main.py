import numpy as np
from matplotlib import pyplot as plt


# In this task, you should implement a logistic regression model for a binary (two-class)
# classification problem. You cannot use machine learning libraries such as Scikit-learn,
# Tensorflow, Keras, or PyTorch.

# using sum of squares as cost function, calculate the gradient of the cost function
# with respect to the parameters w and b
# Calculate the Gradients of the cost function with respect to the parameters w and b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Phi(X, w, b):
    return sigmoid(np.dot(X, w) + b)


def C(X, y, w, b):
    N = len(y)
    A = Phi(X, w, b)
    p1 = y * np.log(A)
    p2 = (1 - y) * np.log(1 - A)
    cost = -1 / N * np.sum(p1 + p2)
    return cost


def compute_dC_dw(X, y, w, b):
    # analytical gradient, which comes from the derivative of the cost function
    # with respect to the parameter w
    # dC = np.dot(X, A - y)
    N = len(y)
    A = Phi(X, w, b)
    dC_dw = 1 / N * np.dot(X.T, (A - y))
    return np.sum(dC_dw, axis=1).reshape(-1, 1)


def compute_dC_db(X, y, w, b):
    N = len(y)
    A = Phi(X, w, b)
    dC_db = 1 / N * np.sum(A - y)
    return dC_db


def compute_dC_dw_numeric(X, y, w, b, epsilon=1e-6):
    # the idea is to calculate the gradient numerically
    # by using the definition of the derivative
    # note that w is a vector, so we need to calculate the gradient
    # for each element of w
    dC_dw = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += epsilon
        w_n[i] -= epsilon
        dC_dw[i] = (C(X, y, w_p, b) - C(X, y, w_n, b)) / (2 * epsilon)
    return dC_dw


def compute_dC_db_numeric(X, y, w, b, epsilon=1e-6):
    # the idea is to calculate the gradient numerically
    # by using the definition of the derivative
    return (C(X, y, w, b + epsilon) - C(X, y, w, b - epsilon)) / (2 * epsilon)


def plot_data(X, y, pred_y, animation=False, fig=plt.figure()):
    """
    Plot the data points and the decision boundary, used for animating the training process
    :param X:
    :param y:
    :param pred_y:
    :return:
    """
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')
    ax.scatter(X[:, 0], X[:, 1], pred_y, c='b', marker='o')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    if animation:
        plt.pause(0.1)
    else:
        plt.show()


def train(X, y, w, b, learning_rate=0.01, num_iterations=1000):
    for i in range(num_iterations):
        dC_dw = compute_dC_dw(X, y, w, b)
        dC_db = compute_dC_db(X, y, w, b)
        w = w - learning_rate * dC_dw
        b = b - learning_rate * dC_db
        if i % 100 == 0:
            plot_data(X, y, Phi(X, w, b), True)
            print("Cost after iteration %i: %f" % (i, C(X, y, w, b)))
    return w, b


if __name__ == '__main__':
    raw_data = np.load('data2d.npz')
    X = raw_data['X']  # shape (N, d)
    y = raw_data['y']  # shape (N,)
    w = np.ones((X.shape[1], 1))  # shape (d, 1)
    b = 0  # scalar

    # train the model
    w = np.random.randn(X.shape[1], 1)
    w, b = train(X, y, w, b, learning_rate=0.5, num_iterations=1000)
    pred_y = Phi(X, w, b)
    plot_data(X, y, pred_y, False)
