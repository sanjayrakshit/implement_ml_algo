import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler


# np.random.seed(10
# )


def get_dataset():
    X, y = make_classification(
        100, 2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)
    X = MinMaxScaler().fit_transform(X)
    data = {f'feature_{i + 1}': x for i, x in enumerate(zip(*X))}
    data['target'] = [1 if i == 1 else -1 for i in y]
    df = pd.DataFrame(data)
    df = df[[c for c in sorted(df.columns)
             if c not in ['target']] + ['target']]
    return df


def hinge_loss(y, x2, x1, w2, w1, w0):
    h = (x2 * w2) + (x1 * w1) + w0
    hinge = max(0, 1 - y * h)
    return hinge


def linear_svm(_data):
    w2, w1, w0 = 0.5, 0.5, 0.5
    lr = 0.005

    for iter in range(10000):
        loss = 0
        grad2, grad1, grad0 = 0, 0, 0
        for x1, x2, y in _data:
            inst_hinge_loss = hinge_loss(y, x2, x1, w2, w1, w0)
            loss += inst_hinge_loss
            grad2 += 0. if inst_hinge_loss == 0. else -y * x2
            grad1 += 0. if inst_hinge_loss == 0. else -y * x1
            grad0 += 0. if inst_hinge_loss == 0. else -y * 1
        w2 = w2 - lr * grad2
        w1 = w1 - lr * grad1
        w0 = w0 - lr * grad0

        print(
            f'Iteration: {iter:>7} || loss: {loss:.5f} || w2: {w2:.5f} || w1: {w1:.5f} || w0: {w0:.5f}')
    return w2, w1, w0


if __name__ == '__main__':
    dataf = get_dataset()
    u2, u1, u0 = linear_svm(dataf.values.tolist())

    def surface(k):
        return - (u1 / u2) * k - (u0 / u2)

    mark = list(range(-100, 100))
    plt.plot(mark, list(map(surface, mark)),
             color='green', label='lin-svm')
    plt.legend()
    plt.xlim(-0.1, +1.1)
    plt.ylim(-0.1, +1.1)
    print(f'Columns: {dataf.columns}')
    sns.scatterplot(x='feature_1', y='feature_2', data=dataf, hue='target')

    plt.savefig('plots/linear_svm.png')
