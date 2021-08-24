from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt


def simple_linear_reg(_data):
    w1, w0 = 0.5, 0.5
    lr = 0.001
    for i in range(15000):
        loss = 0
        grad1 = 0
        grad0 = 0
        for x, y in _data:
            loss += (y - w1*x - w0)**2
            grad1 += 2*(y - w1*x - w0) * -1*x
            grad0 += 2*(y - w1*x - w0) * -1
        mse = np.mean(loss)
        w1 = w1 - lr * grad1
        w0 = w0 - lr * grad0
        print(
            f'Iteration: {i:>3} || loss: {mse:.5f} || w1: {w1:.5f} || w0: {w0:.5f}')

    return w1, w0


def linear_reg_v2(_data):
    w2, w1, w0 = 0.5, 0.5, 0.5
    lr = 0.001
    for i in range(15000):
        loss = 0
        grad2 = 0
        grad1 = 0
        grad0 = 0
        for x, y in _data:
            loss += (y - w2*x*x - w1*x - w0)**2
            grad2 += 2*(y - w2*x*x - w1*x - w0) * -1*x*x
            grad1 += 2*(y - w2*x*x - w1*x - w0) * -1*x
            grad0 += 2*(y - w2*x*x - w1*x - w0) * -1
        mse = np.mean(loss)
        w2 = w2 - lr * grad2
        w1 = w1 - lr * grad1
        w0 = w0 - lr * grad0
        print(
            f'Iteration: {i:>3} || loss: {mse:.5f} || w2: {w2:.5f} || w1: {w1:.5f} || w0: {w0:.5f}')

    return w2, w1, w0


if __name__ == '__main__':
    data = [(1, 1), (2, 3), (3, 4), (5, 5)]
    my_x, my_y = zip(*data)

    u1, u0 = simple_linear_reg(data)
    pred_y = [u1*k + u0 for k in my_x]

    u2, u1, u0 = linear_reg_v2(data)
    pred_y_v2 = [u2*k*k + u1*k + u0 for k in my_x]

    plt.plot(my_x, my_y, label='train', color='blue')
    plt.plot(my_x, pred_y, label='pred', color='red')
    plt.plot(my_x, pred_y_v2, label='pred_v2', color='orange')
    plt.legend()

    plt.savefig('graph/linear_reg.png')
