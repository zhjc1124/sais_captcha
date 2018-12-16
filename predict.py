import numpy as np
from operate import gif2vec, vec2char
from train import sigmoid, load_theta
input_layer_num = 128
hidden_layer_num = 80
output_layer_num = 54


def predict(x, theta):
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
    txt = vec2char(a3)
    return txt


def calc_error_rate(x_test, y_test, theta):
    pred = predict(x_test, theta)

    real = vec2char(y_test)

    num = len(pred)
    error = 0

    for i in range(len(pred)):
        # print(pred[i], real[i])
        if pred[i] != real[i]:
            error += 1
    print('error rate: ', error/num)
    return error/num


def ocr(url, theta):
    x = gif2vec(url)
    labels = predict(x, theta)
    count_dicts = [{}, {}, {}, {}]
    for index, l in enumerate(labels):
        i = index % 4
        if l in count_dicts[i]:
            count_dicts[i][l] += 1
        else:
            count_dicts[i][l] = 1
    return ''.join(map(lambda x: max(x, key=x.get), count_dicts))


def main():
    theta, iteration = load_theta()
    x_test = np.load('x.npy')
    y_test = np.load('y.npy')
    calc_error_rate(x_test[:1000], y_test[:1000], theta)
    error =0
    for i in range(1, 250+1):
        pred = ocr('./tests/gifs/%06d.gif' % i, theta)
        with open('./tests/labels/%06d.txt' % i) as f:
            real = f.read()
        if pred != real:
            print(pred, real)
            error += 1
    print('label error rate: ', error/250)


if __name__ == '__main__':
    main()
