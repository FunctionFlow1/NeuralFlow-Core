import numpy as np

class Layer:
    """Base class for all neural network layers."""
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """Performs the forward pass computation."""
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        """Performs the backward pass computation (gradient calculation)."""
        raise NotImplementedError

class Dense(Layer):
    """A fully connected (dense) layer."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient

class Activation(Layer):
    """Base class for activation functions."""
    def __init__(self, activation_func, activation_prime_func):
        super().__init__()
        self.activation_func = activation_func
        self.activation_prime_func = activation_prime_func

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation_func(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime_func(self.input)

class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0).astype(float)

        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    """Sigmoid activation function."""
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Tanh(Activation):
    """Hyperbolic Tangent activation function."""
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x)**2

        super().__init__(tanh, tanh_prime)

# Example Usage (conceptual, would integrate with C++ backend for performance)
if __name__ == "__main__":
    # Simulate input data
    X = np.random.randn(10, 5) # 10 samples, 5 features

    # Create a simple neural network
    network = [
        Dense(5, 10),
        ReLU(),
        Dense(10, 3),
        Sigmoid()
    ]

    # Forward pass
    output = X
    for layer in network:
        output = layer.forward(output)
    print("\nOutput of the network (first 3 samples):\n", output[:3])

    # Simulate backward pass (simplified)
    # Assuming a loss gradient of the same shape as output
    output_gradient = np.random.randn(10, 3)
    learning_rate = 0.01

    for layer in reversed(network):
        output_gradient = layer.backward(output_gradient, learning_rate)
    print("\nBackward pass completed.")

    # In a real scenario, this Python code would call into the C++ core_engine
    # for performance-critical operations like matrix multiplication.
    print("\n(Note: In a production NeuralFlow-Core, performance-critical ops would leverage the C++ backend.)")
