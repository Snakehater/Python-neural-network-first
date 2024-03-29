import numpy as np

# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1
print('synaptic_weights: ')
print(synaptic_weights)
print('\n')

for iteration in range(20000):

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights ))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print('synaptic_weights after training: ')
print(synaptic_weights)

print('Output after training: ')
print(outputs)

inputss = input('Input: ')
usrInput1 = int(inputss[0])
usrInput2 = int(inputss[1])
usrInput3 = int(inputss[2])
calculated = sigmoid(np.dot(np.array([usrInput1, usrInput2, usrInput3]), synaptic_weights))
print(int(round(calculated[0])))
