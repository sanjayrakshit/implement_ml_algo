import math
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# np.random.seed(10
# )


def get_dataset():
    X, y = make_classification(
        100, 2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)
    X = MinMaxScaler().fit_transform(X)
    data = {f'feature_{i+1}': x for i, x in enumerate(zip(*X))}
    data['target'] = y
    df = pd.DataFrame(data)
    df = df[[c for c in sorted(df.columns)
             if c not in ['target']] + ['target']]
    return df


def px_(x2, x1, w2, w1, w0):
    h = (x2 * w2) + (x1 * w1) + w0
    p = 1 / (1 + math.exp(-h))
    return p


def logistic_regression(_data):
    w2, w1, w0 = 0.5, 0.5, 0.5
    lr = 0.01

    for iter in range(10000):
        loss = 0
        grad2, grad1, grad0 = 0, 0, 0

        for x1, x2, y in _data:
            _p = px_(x2, x1, w2, w1, w0)
            lle = y * math.log(_p) + \
                (1 - y) * math.log(1 - _p)
            loss += -1 * lle
            grad2 += -1 * (y - _p) * x2
            grad1 += -1 * (y - _p) * x1
            grad0 += -1 * (y - _p) * 1

        mean_loss = np.mean(loss)
        w2 = w2 - lr*grad2
        w1 = w1 - lr*grad1
        w0 = w0 - lr*grad0

        print(
            f'Iteration: {iter:>3} || loss: {mean_loss:.5f} || w2: {w2:.5f} || w1: {w1:.5f} || w0: {w0:.5f}')

    return w2, w1, w0


if __name__ == '__main__':
    dataf = get_dataset()
    u2, u1, u0 = logistic_regression(dataf.values.tolist())

    def surface(k):
        return - (u1/u2) * k - (u0/u2)

    mark = list(range(-100, 100))
    plt.plot(mark, list(map(surface, mark)),
             color='#e64980', label='LR')
    plt.legend()
    plt.xlim(-0.1, +1.1)
    plt.ylim(-0.1, +1.1)
    print(f'Columns: {dataf.columns}')
    sns.scatterplot(x='feature_1', y='feature_2', data=dataf, hue='target')

    plt.savefig('plots/logistic_regression.png')
