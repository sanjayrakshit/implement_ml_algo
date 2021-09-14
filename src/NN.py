import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def ddxsigmoid(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


class NeuralNetwork:
    def __init__(self, size, lr):
        # initializing everything with 0.5 since it converges faster
        self.w = np.zeros(size) + 0.5
        self.lr = lr
        print(self.w)

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self._forward(self.x, self.y)

    def _forward(self, x, y):
        self.first_stage = x.dot(self.w)
        self.y_hat = sigmoid(self.first_stage)
        self.se = (y - self.y_hat) ** 2
        return self.y_hat, self.se

    def backward(self):
        # Calculate gradients
        self.grad = 2 * (self.y_hat - self.y) * \
            np.dot(self.x.T, ddxsigmoid(self.first_stage))
        return self.grad

    def apply_gradients(self, accumulated_grad):
        # Apply gradients
        self.w = self.w - self.lr * accumulated_grad


if __name__ == '__main__':
    nn = NeuralNetwork((4, 1), 5e-2)
    data = [(1, 1), (2, 3), (3, 4), (5, 5)]
    losses = []
    for itern in range(300):
        loss = 0
        grad = 0
        for x, y in data:
            x = np.array([[x ** 0.5, x, x ** 2, x ** 3]])
            y_hat, se = nn.forward(x, y)
            loss += se
            grad += nn.backward()
        mse = loss / len(data)
        losses.append(mse)
        print(f'iter: {itern:<4} | loss: {mse:.5}')
        nn.apply_gradients(grad)

        sns.scatterplot(x=range(1, len(losses) + 1), y=losses, color='blue')
        plt.savefig('plots/nn_loss.png')
