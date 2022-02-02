from neural_net import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    net = Network([2, 2, 1])
    net.learn(np.array([[0, 0], [1, 1], [1, 0], [0, 1]]), np.array([[0], [1], [0], [0]]), 1000, 10)
    print(net.forward_propagate(np.array([0, 0])))
    print(net.forward_propagate(np.array([0, 1])))
    print(net.forward_propagate(np.array([1, 0])))
    print(net.forward_propagate(np.array([1, 1])))

