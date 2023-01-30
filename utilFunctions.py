import numpy as np
from matplotlib import pyplot as plt


def plot_data_3D(X, y, pred_y, animation=False, fig=plt.figure()):
    """
    Plot the data points, used for animating the training process
    """
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    # different colors for different classes
    # use unfilled markers for misclassified points, filled markers for correctly classified points
    for i in range(len(y)):
        color = 'blue' if y[i] == 0 else 'red'
        pred_y_i = 1 if pred_y[i] >= 0.5 else 0
        if pred_y_i != y[i]:  # misclassified
            marker = 'o'  # unfilled marker
        else:  # correctly classified
            marker = 'x'
        ax.scatter(X[i, 0], X[i, 1], pred_y[i], color=color, marker=marker, edgecolors='black')
    # plot true data in green
    ax.scatter(X[:, 0], X[:, 1], y, c='g', marker='o')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    if animation:
        plt.pause(0.01)
    else:
        plt.show()


def plot_data_2D(X, y, y_pred, w, b, draw_line=True, animation=False):
    plt.clf()
    plt.figure()

    for i in range(len(y)):  # plotting the data points
        color = 'blue' if y[i] == 0 else 'red'
        pred_y_i = 1 if y_pred[i] >= 0.5 else 0
        if pred_y_i != y[i]:  # misclassified
            marker = 'o'
        else:  # correctly classified
            marker = 'x'
        plt.scatter(X[i, 0], X[i, 1], c=color, marker=marker)

    if draw_line:
        # plot the separating line, w[0] * x1 + w[1] * x2 + b = 0
        x1 = np.linspace(-1, 1, 100)
        x2 = -(w[0] * x1 + b) / w[1]
        plt.plot(x1, x2, 'g')

    plt.xlabel('X1')
    plt.ylabel('X2')

    if animation:
        plt.pause(0.01)
    else:
        plt.show()


def training_error(y, pred_y):
    y = np.array(y).reshape(-1)
    pred_y = np.array(pred_y).reshape(-1)
    pred_y[pred_y >= 0.5] = 1
    pred_y[pred_y < 0.5] = 0
    not_eq = pred_y != y
    return np.sum(not_eq) / len(y)
