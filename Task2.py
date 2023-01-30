import numpy as np
import tensorflow as tf

from utilFunctions import plot_data_3D, training_error


def sigmoid(z):
    return 1 / (1 + tf.exp(-z))


def Phi(X, w, b):
    return sigmoid(tf.matmul(X, w) + b)


def C(X, y, w, b):
    y_pred = Phi(X, w, b)
    y_pred = tf.reshape(y_pred, tf.shape(y))  # reshape to (N,)
    N = tf.cast(tf.shape(y)[0], tf.float32)
    cost = -(1 / N) * tf.reduce_sum(y * tf.math.log(y_pred) + (1 - y) * tf.math.log(1 - y_pred))
    return cost


def grad_C(X, y, w, b):
    with tf.GradientTape() as tape:
        loss = C(X, y, w, b)
    return tape.gradient(loss, [w, b])


def train(X, y, w, b, learning_rate, threshold, num_iterations):
    for i in range(num_iterations):
        dw, db = grad_C(X, y, w, b)
        w.assign_sub(learning_rate * dw)
        b.assign_sub(learning_rate * db)
        cost = C(X, y, w, b)
        if i % 20 == 0:
            print(f'Cost at iteration {i}: {cost}')
            print(f'Training error: {training_error(y, Phi(X, w, b))}')
            plot_data_3D(X, y, Phi(X, w, b), True)
        if tf.norm(dw) < threshold:
            break
    return w, b


if __name__ == '__main__':
    raw_data = np.load('data5d.npz')
    X = raw_data['X']  # shape(N, d)
    y = raw_data['y']  # shape(N, )
    N, d = X.shape

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    w = tf.Variable(tf.random.normal((d, 1)), dtype=tf.float32)
    b = tf.Variable(tf.random.normal((1,)), dtype=tf.float32)

    w, b = train(X, y, w, b, learning_rate=0.2, threshold=0.025, num_iterations=200)
    print(f'Final cost: {C(X, y, w, b)}')
    print(f'Final parameters: w = {w}, b = {b}')
    plot_data_3D(X, y, Phi(X, w, b), False)
