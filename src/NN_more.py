import math
from typing import List, Any

import numpy as np


def linear(inp, wei):
    return inp.dot(wei)


class AutoNN:
    
    def __init__(self, sizes, lr):
        self.weight = [np.zeros(s) + 0.5 for s in sizes]
        self.grads = [np.zeros(s) for s in sizes]
        self.lr = lr

    def forward(self, x, y):
        self.outs = [x]
        for i in range(len(self.weight)):
            self.outs.append(linear(self.outs[i], self.weight[i]))
        print('Weights:', self.weight)
        print('Outs:', self.outs)


if __name__ == '__main__':
    sizes = [(3, 3), (3, 2), (2, 1)]
    ann = AutoNN(sizes, 1e-2)
    x = np.array([[1, 1, 1]])
    y = np.array([1])
    ann.forward(x, y)
