import os
import numpy as np
import re
INIT_EPSILON = 0.1
LAMBDA = 3
ALPHA = 1
input_layer_num = 128
hidden_layer_num = 80
output_layer_num = 54


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_gradient(x):
    return sigmoid(x)*(1-sigmoid(x))


def cost(x, y, theta):
    m = x.shape[0]

    theta1 = theta[:hidden_layer_num*(input_layer_num+1)]
    theta1.resize([hidden_layer_num, input_layer_num+1])
    theta2 = theta[hidden_layer_num*(input_layer_num+1):]
    theta2.resize([output_layer_num, hidden_layer_num+1])

    a1 = np.hstack((np.ones([m, 1]), x))
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)

    a2 = np.hstack((np.ones([m, 1]), a2))
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)

    J = np.sum(-y * np.log(a3) - (1 - y) * np.log(1 - a3))
    J += (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2))) * LAMBDA / 2
    J /= m

    delta3 = a3 - y
    delta2 = np.dot(delta3, theta2[:, 1:]) * sigmoid_gradient(z2)

    grad1 = np.dot(delta2.T, a1)/m
    grad2 = np.dot(delta3.T, a2)/m

    grad1[:, 1:] += np.power(grad1[:, 1:], 2) * LAMBDA / m
    grad2[:, 1:] += np.power(grad2[:, 1:], 2) * LAMBDA / m

    return [J, np.append(grad1, grad2)]


def check(x, y, theta):
    e = np.eye(theta.size) * 0.001
    esti = np.zeros(20)
    J, grad = cost(x, y, theta)
    for i in range(10):
        esti[i] = (cost(x, y, theta+e[i])[0] - cost(x, y, theta-e[i])[0]) / 2 / 0.001
    index = hidden_layer_num * (input_layer_num + 1)
    for i in range(10):
        esti[10+i] = (cost(x, y, theta+e[index+i])[0] - cost(x, y, theta-e[index+i])[0]) / 2 / 0.001
    print(grad[:10])
    print(esti[:10])
    print(grad[index:index+10])
    print(esti[10:])


def load_theta():
    files = os.listdir('./theta')
    max_iteration = 0
    if files:
        max_iteration = max(map(lambda x: int(re.findall('theta-(.*?).npy', x)[0]), files))
        print('load /theta-%d.npy' % max_iteration)
        theta = np.load('./theta/theta-%d.npy' % max_iteration)
    else:
        theta = np.random.rand(hidden_layer_num * (input_layer_num+1) + output_layer_num * (hidden_layer_num+1))\
            * 2 * INIT_EPSILON - INIT_EPSILON
    return theta, max_iteration


def main():
    theta, iteration = load_theta()
    print('theta size', theta.size)
    x = np.load('x.npy')
    print('x shape:', x.shape)
    y = np.load('y.npy')
    print('y shape:', y.shape)
    # check(x, y, theta)
    while True:
        for i in range(1, 50+1):
            J, grad = cost(x, y, theta)
            theta -= ALPHA * grad
            iteration += 1
            print('iteration: ', iteration, ',cost: ', J)
        np.save('./theta/theta-%d.npy' % iteration, theta)


if __name__ == '__main__':
    main()
