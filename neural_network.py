import numpy as np
import math


class NeuralNetwork:

    class Layer:
        def __init__(self, n_inputs, n_neurons, mutation_rate_weights, mutation_rate_biases):
            self.sigma_weights = mutation_rate_weights
            self.sigma_biases = mutation_rate_biases
            self.outputs = None
            self.n_inputs = n_inputs
            self.n_neurons = n_neurons
            self.rng = np.random.default_rng()
            self.weights = np.array(mutation_rate_weights * self.rng.standard_normal(size=(n_inputs, n_neurons)))
            self.biases = np.array(mutation_rate_biases * self.rng.standard_normal(size=(1, n_neurons)))

        def forward(self, inputs):
            if not len(inputs) == self.n_inputs:
                raise ValueError("network layer got wrong number of inputs")
            self.outputs = np.maximum(0, np.squeeze(np.dot(inputs, self.weights) + self.biases))

        def print_weights(self):
            print(self.weights.shape)
            print(self.weights)

        def print_biases(self):
            print(self.biases.shape)
            print(self.biases)

        def variate_values(self):
            self.weights += np.array(
                self.sigma_weights * self.rng.standard_normal(size=(self.n_inputs, self.n_neurons)))
            self.biases += np.array(self.sigma_biases * self.rng.standard_normal(size=(1, self.n_neurons)))

    def __init__(self, shape, sigma_weights=0.1, sigma_biases=0.1):
        self.outputs = None
        self.shape = shape
        self.network = []
        for i, layer_size in enumerate(shape):
            if i == 0:
                continue
            self.network.append(self.Layer(shape[i - 1], shape[i], sigma_weights, sigma_biases))

    def mutate_values(self):
        for layer in self.network:
            layer.variate_values()

    def calculate(self, inputs):
        temp = None
        for layer in self.network:
            if layer == self.network[0]:
                layer.forward(inputs)
            else:
                layer.forward(temp)
            temp = layer.outputs
        self.outputs = temp
        return self.outputs

    def print_network(self):
        for i, x in enumerate(self.network):
            print(f"Layer {i+1}:")
            x.print_weights()
            x.print_biases()


if __name__ == "__main__":
    network = NeuralNetwork([4, 2, 3, 2, 2])
    network.calculate([-1, -2, -3, -4])
    print(network.outputs)
