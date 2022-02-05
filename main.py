from neural_net import *


def forward_propagate():
    print(net.forward_propagate(np.array([0, 0])))
    print(net.forward_propagate(np.array([0, 1])))
    print(net.forward_propagate(np.array([1, 0])))
    print(net.forward_propagate(np.array([1, 1])))


if __name__ == '__main__':
    net = Network([2, 2, 1])
    # forward_propagate()
    net.learn(np.array([[0, 0], [1, 1], [1, 0], [0, 1]]), np.array([[1], [0], [1], [1]]), 1000, 5)
    print()
    forward_propagate()
