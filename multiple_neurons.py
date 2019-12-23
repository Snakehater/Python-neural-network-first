# https://www.youtube.com/watch?v=oJoPbwmVBlk
import numpy as np
class NeuralNetwork():

    def __init__(self):
        self.imputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    # sigmoid function to normalize inputs
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def constFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)

    def sigmoidPrime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def constFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        djdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3.selfw3.T)*self.sigmoidPrime(self.z2)
        djdW1 = np.dot(X.T, delta2)

    # sigmoid derivatives to adjust synaptic weights
    # def sigmoid_derivative(self, x):
    #     return x * (1 - x)


if __name__ == "__main__":
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)
    print(X.shape)
    print(y.shape)
