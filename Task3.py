import tensorflow as tf
import numpy as np

# Load the iris data set
iris_data = np.loadtxt('iris.data', delimiter=',', dtype=str)

# Separate the numerical data and target labels into two separate arrays
X = iris_data[:, :-1].astype(np.float64)
y = iris_data[:, -1]

# Convert the categorical target labels to numerical values
target_labels = np.unique(y)
num_labels = len(target_labels)
y_onehot = np.zeros((y.shape[0], num_labels))
for i, label in enumerate(target_labels):
    y_onehot[y == label, i] = 1

# Update the target labels array
y = y_onehot


# Define the neural network model
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = tf.Variable(tf.random.normal([input_dim, hidden_dim], dtype=tf.float64), name='weights_1',
                              dtype=tf.float64)
        self.b1 = tf.Variable(tf.zeros([hidden_dim], dtype=tf.float64), name='bias_1', dtype=tf.float64)
        self.W2 = tf.Variable(tf.random.normal([hidden_dim, hidden_dim], dtype=tf.float64), name='weights_2',
                              dtype=tf.float64)
        self.b2 = tf.Variable(tf.zeros([hidden_dim], dtype=tf.float64), name='bias_2', dtype=tf.float64)
        self.W3 = tf.Variable(tf.random.normal([hidden_dim, output_dim], dtype=tf.float64), name='weights_3',
                              dtype=tf.float64)
        self.b3 = tf.Variable(tf.zeros([output_dim], dtype=tf.float64), name='bias_3', dtype=tf.float64)

    def sigmoid(self, x):
        return tf.math.sigmoid(x)

    def softmax(self, x):
        exp_x = tf.math.exp(x)
        return exp_x / tf.reduce_sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        z1 = tf.matmul(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = tf.matmul(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = tf.matmul(a2, self.W3) + self.b3
        a3 = self.softmax(z3)
        return a3


# Initialize the model
input_dim = 4
hidden_dim = 8
output_dim = 3
model = NeuralNetwork(input_dim, hidden_dim, output_dim)

# Define the cost function
cost = lambda y, y_pred: tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred), axis=1))


def train(X, y, model, cost, learning_rate=0.05, epochs=1000):
    # Train the model
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model.forward(X)
            current_cost = cost(y, y_pred)
        grads = tape.gradient(current_cost, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3])
        model.W1.assign_sub(learning_rate * grads[0])
        model.b1.assign_sub(learning_rate * grads[1])
        model.W2.assign_sub(learning_rate * grads[2])
        model.b2.assign_sub(learning_rate * grads[3])
        model.W3.assign_sub(learning_rate * grads[4])
        model.b3.assign_sub(learning_rate * grads[5])
        if epoch % 100 == 0:
            print("Epoch: {}, Cost: {}".format(epoch, current_cost.numpy()))
    return model


# Train the model
trained_model = train(X, y, model, cost)

# Evaluate the model
y_pred = model.forward(X)
correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print("Accuracy: {}".format(accuracy.numpy()))
