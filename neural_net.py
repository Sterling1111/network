import math
import time
import numpy as np
import pygame
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


def compress_canvas(screen):
    small_canvas = pygame.surfarray.array3d(screen)
    small_canvas = np.mean(small_canvas, axis=2) / 255.0
    compressed_canvas = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            block = small_canvas[i*50:(i+1)*50, j*50:(j+1)*50]
            compressed_canvas[i, j] = np.mean(block)
    compressed_canvas = np.rot90(compressed_canvas, k=-1)
    compressed_canvas = np.flip(compressed_canvas, axis=1)
    return 1 - compressed_canvas  # Return 8x8 image


def pygame_canvas():
    pygame.init()
    screen = pygame.display.set_mode([400, 400])
    pygame.display.set_caption("Draw a Digit")
    screen.fill((255, 255, 255))

    running = True
    drawing = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                compressed_digit = compress_canvas(screen)
                pygame.quit()  # Quit Pygame after pressing Enter
                return compressed_digit

        if drawing:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, (0, 0, 0), mouse_pos, 20)
        pygame.display.flip()

    pygame.quit()


def show_compressed_image(compressed_image):
    """Function to show the compressed image using plt, similar to how training data is shown."""
    plt.gray()
    plt.matshow(compressed_image)
    plt.show()


if __name__ == '__main__':
    digits = load_digits()
    net = Network([64, 16, 16, 10])
    num_to_train = 777
    targets = np.empty((num_to_train, 10))
    for i, elem in enumerate(digits.target[0:num_to_train]):
        targets[i] = np.zeros(10)
        targets[i][elem] = 1
    inputs = digits.data[0:num_to_train]
    for i, elem in enumerate(inputs):
        inputs[i] = inputs[i] / 16

    net.learn(inputs, targets, learning_rate=.00213, seconds=1000)
    np.set_printoptions(suppress=True)
    num_wrong = 0
    for i, target in enumerate(digits.target):
        result = net.forward_propagate(digits.data[i])
        if target != np.argmax(result):
            num_wrong = num_wrong + 1
            # print(i, np.argmax(result), target)
            # print(result)
            # plt.gray()
            # plt.matshow(digits.images[i])
            # plt.show()
    print(str(round(100 * ((len(digits.target) - num_wrong) / len(digits.target)), 0)) + '% accuracy')

    while True:
        choice = input("Enter '1' to input a number or '2' to draw a digit (or 'q' to quit): ").strip()
        if choice == '1':
            num = int(input("Enter a number: "))
            if num == -1:
                break
            else:
                result = net.forward_propagate(digits.data[num] / 16)
                target = digits.target[num]
                print('Predicted:', np.argmax(result), 'Actual:', target)
                print('Output Layer Activations:')
                print(result)
                plt.gray()
                plt.matshow(digits.images[num])
                plt.show()

        elif choice == '2':
            compressed_digit = pygame_canvas()
            show_compressed_image(compressed_digit)  # Display the compressed image using plt
            compressed_digit_flattened = compressed_digit.reshape(64)  # Flatten for input into the network
            result = net.forward_propagate(compressed_digit_flattened)
            print("Neural Network Prediction:", np.argmax(result))
            print("Output Layer Activations:")
            print(result)

        elif choice.lower() == 'q':
            break
