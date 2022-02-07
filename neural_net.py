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

    def learn(self, inputs, targets, epochs, learning_rate=1.0):
        for i in range(epochs):
            weight_derivatives = [np.zeros((y, x))
                                  for x, y in zip(self.network_layer_sizes[:-1], self.network_layer_sizes[1:])]
            bias_derivatives = [np.zeros((x, 1)) for x in self.network_layer_sizes[1:]]
            for input, target in zip(inputs, targets):
                output = self.forward_propagate(input)
                difference = output - target
                self.backward_propagate(difference)
                weight_derivatives = [np.add(x, y) for x, y in zip(weight_derivatives, self.weight_derivatives)]
                bias_derivatives = [np.add(x, y) for x, y in zip(bias_derivatives, self.bias_derivatives)]
            self.weight_derivatives = [np.divide(x, len(weight_derivatives)) for x in weight_derivatives]
            self.bias_derivatives = [np.divide(x, len(bias_derivatives)) for x in bias_derivatives]
            self.gradient_descent(learning_rate)


def forward_propagate(network):
    print(network.forward_propagate(np.array([0, 0])))
    print(network.forward_propagate(np.array([0, 1])))
    print(network.forward_propagate(np.array([1, 0])))
    print(network.forward_propagate(np.array([1, 1])))


if __name__ == '__main__':
    # Its easy to create an arbitrary neural net. Simply pass to the constructor a list which contains the number
    # of neurons in each layer. In out example we will have a net of 2 input, 2 hidden, and 1 output neuron.
    net = Network([2, 2, 1])
    print('Before training')
    forward_propagate(net)
    # Once created call the learn function. It takes a np 2d array where the rows are input elements
    # in a single training set. It then takes a np 2d array where the rows are the target output of
    # the net. Next it takes an epoch which is the number of times it will run the entire training set
    # and backpropagate. Finally it takes a learning rate which is multiplied by the gradient vector
    # when doing gradient descent. The larger it is the faster it learns**. There is no error checking
    # of any kind so make sure that the dimensions match. It is configured now for xor so you can simply
    # run it and see the results. Forward propagate returns the result vector. I configure the net then
    # before it is trained I forward prop on the inputs. The results are bad. Then I train then forward
    # propagate and the results are good.
    net.learn(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
              np.array([[0], [1], [1], [0]]), 1000, 10)
    print()
    print('After training')
    forward_propagate(net)
