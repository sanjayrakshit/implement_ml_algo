import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def ddxsigmoid(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


class NeuralNetwork:
    def __init__(self, size: Tuple[int, int], lr: float) -> None:
        """Basic init

        Args:
            size (Tuple[int, int]): The size of the matrix / layer
            lr (float): This is the learning rate for gradient descent
        """
        # initializing everything with 0.5 since it converges faster
        self.w = np.zeros(size) + 0.5
        self.lr = lr

    def forward(self, x: np, y: float) -> Tuple[float, float]:
        """Basically calls the _forward function
        """
        self.x = x
        self.y = y
        return self._forward(self.x, self.y)

    def _forward(self, x: np, y: float) -> Tuple[float, float]:
        """Main forward function for the forward pass

        Args:
            x (np): This is the feature
            y (float): This is the target

        Returns:
            Tuple[float, float]: Returns the prediction and the loss
        """
        # Do the matrix multiplication / linear layer
        self.first_stage = x.dot(self.w)
        # Apply activation
        self.y_hat = sigmoid(self.first_stage)
        # Calculate loss
        self.se = (y - self.y_hat) ** 2
        return self.y_hat, self.se

    def backward(self) -> np:
        """Backward function deos the backward pass and chain rule

        Returns:
            np: Returns the calculated gradients
        """
        # Calculate gradients
        self.grad = 2 * (self.y_hat - self.y) * \
            np.dot(self.x.T, ddxsigmoid(self.first_stage))
        return self.grad

    def apply_gradients(self, accumulated_grad: np) -> None:
        """Applies the gradient to the weight matrix

        Args:
            accumulated_grad (np): This is the acumulated gradient from the backward function
        """
        # Apply gradients
        self.w = self.w - self.lr * accumulated_grad


if __name__ == '__main__':
    nn = NeuralNetwork((3, 1), 5e-2)
    # data = [(1, 1), (2, 3), (3, 4), (5, 5)]
    features, targets = make_regression(10, 3, n_informative=3, n_targets=1, noise=10
                                        )
    # You're not supposed to scale the train and test data together, but we are going to do it for convenience
    features = MinMaxScaler().fit_transform(features)
    losses = []
    epochs = 1e4
    for itern in range(1, int(epochs)+1):
        loss = 0
        grad = 0
        for x, y in zip(features, targets):
            x = np.array([x])
            y_hat, se = nn.forward(x, y)
            loss += se
            grad += nn.backward()
        mse = loss / len(features)
        losses.append(mse)
        print(f'iter: {itern:<4} | loss: {mse:.5}')
        nn.apply_gradients(grad)

    sns.scatterplot(x=range(1, len(losses) + 1), y=losses, color='blue')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss vs Epoch')
    plt.savefig('plots/nn_loss.png')
