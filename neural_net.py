
# This program  demonstrates a simple feed-forward neural network with two input neurons,
# two hidden neurons, and one output neuron. The network is trained using backpropagation
# and the weights and biases are updated according to the computed gradients.
# Finally, we make predictions for each input and display the results.

import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for x, target in zip(X, y):
                x = np.array(x).reshape(-1, 1)
                target = np.array(target).reshape(-1, 1)

                # Forward pass
                activations = [x]
                zs = []

                for w, b in zip(self.weights, self.biases):
                    z = np.dot(w, activations[-1]) + b
                    zs.append(z)
                    activations.append(self.sigmoid(z))

                # Backward pass
                delta = (activations[-1] - target) * self.sigmoid_prime(zs[-1])

                nabla_b = [delta]
                nabla_w = [np.dot(delta, activations[-2].T)]

                for l in range(2, self.num_layers):
                    delta = np.dot(self.weights[-l + 1].T, delta) * self.sigmoid_prime(zs[-l])
                    nabla_b.append(delta)
                    nabla_w.append(np.dot(delta, activations[-l - 1].T))

                nabla_b = nabla_b[::-1]
                nabla_w = nabla_w[::-1]

                # Update weights and biases
                self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - learning_rate * db for b, db in zip(self.biases, nabla_b)]

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


# Example usage
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

nn = NeuralNetwork([2, 2, 1])  # 2 input neurons, 2 hidden neurons, 1 output neuron
nn.train(X, y, epochs=10000, learning_rate=0.1)

for x, target in zip(X, y):
    prediction = nn.feedforward(np.array(x).reshape(-1, 1))
    print(f"Input: {x} -> Prediction: {prediction}")

    
