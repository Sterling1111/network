import math
import time
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1.0 - x)
    return 1.0 / (1 + np.exp(-x))


class Network(object):
    def __init__(self, network_layer_sizes):
        np.random.seed(1)
        self.num_layers = len(network_layer_sizes)
        self.network_layer_sizes = network_layer_sizes
        self.biases = [2 * np.random.random((x, 1)) - 1 for x in network_layer_sizes[1:]]
        self.weights = [2 * np.random.random((y, x)) - 1
                        for x, y in zip(network_layer_sizes[:-1], network_layer_sizes[1:])]
        self.activations = [np.zeros((x, 1)) for x in network_layer_sizes[:]]
        self.weight_derivatives = [np.zeros((y, x))
                                   for x, y in zip(network_layer_sizes[:-1], network_layer_sizes[1:])]
        self.bias_derivatives = [np.zeros((x, 1)) for x in network_layer_sizes[1:]]

    def forward_propagate(self, inputs):
        activations = inputs.reshape(inputs.shape[0], 1)
        self.activations[0] = activations
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            net_inputs = np.dot(weight, activations) + bias
            activations = sigmoid(net_inputs)
            self.activations[i + 1] = activations
        return activations

    def backward_propagate(self, difference):
        for i in reversed(range(len(self.weight_derivatives))):
            a_i_plus_1 = self.activations[i + 1]
            # this follows from theorem 1 and 2. In first pass difference is del_a C
            # in subsequent passes difference is left part of hadamard from theorem 2
            delta = np.multiply(difference, sigmoid(a_i_plus_1, True))
            a_i = self.activations[i]
            # this follows from theorem 4.
            self.weight_derivatives[i] = np.dot(delta, a_i.T)
            # this follows from theorem 3.
            self.bias_derivatives[i] = delta
            # this follows from theorem 2 we are building left-hand side of hadamard product
            difference = np.dot(self.weights[i].T, delta)

    def gradient_descent(self, learning_rate=1.0):
        for i in range(len(self.weights)):
            self.weights[i] -= self.weight_derivatives[i] * learning_rate
            self.biases[i] -= self.bias_derivatives[i] * learning_rate

    def learn(self, inputs, targets, learning_rate=1.0, epsilon=.001, epochs=math.inf, seconds=math.inf):
        epsilon *= 2
        start_time = time.time()
        while True:
            if epochs < 1:
                print('maximum epochs cycled')
                return
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > seconds:
                print('maximum training time achieved')
                return
            error = 0
            weight_derivatives = [np.zeros((y, x))
                                  for x, y in zip(self.network_layer_sizes[:-1], self.network_layer_sizes[1:])]
            bias_derivatives = [np.zeros((x, 1)) for x in self.network_layer_sizes[1:]]
            for input, target in zip(inputs, targets):
                output = self.forward_propagate(input)
                difference = (output.T - target).T
                error += np.square(difference)
                self.backward_propagate(difference)
                weight_derivatives = [np.add(x, y) for x, y in zip(weight_derivatives, self.weight_derivatives)]
                bias_derivatives = [np.add(x, y) for x, y in zip(bias_derivatives, self.bias_derivatives)]
            self.weight_derivatives = [np.divide(x, len(weight_derivatives)) for x in weight_derivatives]
            self.bias_derivatives = [np.divide(x, len(bias_derivatives)) for x in bias_derivatives]
            epochs -= 1
            if np.sum(error) < epsilon:
                print('minimum error achieved')
                return
            self.gradient_descent(learning_rate)


def forward_propagate(network):
    print(network.forward_propagate(np.array([1, 1])))
    print(network.forward_propagate(np.array([0, 0])))


if __name__ == '__main__':
    digits = load_digits()
    # Its easy to create an arbitrary neural net. Simply pass to the constructor a list which contains the number
    # of neurons in each layer. In out example we will have a net of 2 input, 2 hidden, and 1 output neuron.
    net = Network([64, 16, 16, 10])
    net2 = Network([2, 2, 1])
    net2.learn(np.array([[0, 0], [1, 1]]), np.array([[0], [1]]))
    forward_propagate(net2)
    targets = np.empty((200, 10))
    for i, elem in enumerate(digits.target[0:200]):
        targets[i] = np.zeros(10)
        targets[i][elem] = 1
    inputs = digits.data[0:200]
    for i, elem in enumerate(inputs):
        inputs[i] = inputs[i] / 16

    # pass the training inputs, expected outputs, learning rate, min error, epochs, and max time in seconds.
    net.learn(inputs, targets, epsilon=.005, learning_rate=.03, seconds=200)
    np.set_printoptions(suppress=True)
    print(net.forward_propagate(digits.data[177]))
    print(digits.target[177])
    plt.gray()
    plt.matshow(digits.images[177])
    plt.show()
