import copy
import numpy as np


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
                        for x, y in zip(self.network_layer_sizes[:-1], self.network_layer_sizes[1:])]
        self.activations = [np.zeros((x, 1)) for x in network_layer_sizes[:]]
        self.weight_derivatives = [np.zeros((y, x))
                                   for x, y in zip(self.network_layer_sizes[:-1], self.network_layer_sizes[1:])]
        self.bias_derivatives = [np.zeros((x, 1)) for x in network_layer_sizes[1:]]

    def forward_propagate(self, inputs):
        activations = inputs.reshape(inputs.shape[0], 1)
        self.activations[0] = activations
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            net_inputs = np.dot(weight, activations) + bias
            activations = sigmoid(net_inputs)
            self.activations[i+1] = activations
        return activations

    def backward_propagate(self, difference):
        for i in reversed(range(len(self.weight_derivatives))):
            a_i_plus_1 = self.activations[i + 1]
            delta = np.multiply(difference, sigmoid(a_i_plus_1, True))
            a_i = self.activations[i]
            # why I got to do this???
            self.weight_derivatives[i] = np.dot(delta, a_i.T)
            self.bias_derivatives[i] = delta
            difference = np.dot(self.weights[i].T, delta)

    def gradient_descent(self, learning_rate=1.0):
        for i in range(len(self.weights)):
            self.weights[i] -= self.weight_derivatives[i] * learning_rate
            self.biases[i] -= self.bias_derivatives[i] * learning_rate

    def learn(self, inputs, targets, epochs, learning_rate=1.0):
        for i in range(epochs):
            weight_derivatives = []
            bias_derivatives = []
            for input, target in zip(inputs, targets):
                output = self.forward_propagate(input)
                difference = output - target
                self.backward_propagate(difference)
                weight_derivatives.append(copy.deepcopy(self.weight_derivatives))
                bias_derivatives.append(copy.deepcopy(self.bias_derivatives))
            for (weight_der_list, bias_der_list) in zip(weight_derivatives[:-1], bias_derivatives[:-1]):
                for j, (weight_derivative, bias_derivative) in enumerate(zip(weight_der_list, bias_der_list)):
                    self.weight_derivatives[j] += weight_derivative
                    self.bias_derivatives[j] += bias_derivative
            for (weight_derivative, bias_derivative) in zip(self.weight_derivatives, self.bias_derivatives):
                weight_derivative /= len(weight_derivatives)
                bias_derivative /= len(bias_derivatives)
            self.gradient_descent(learning_rate)
