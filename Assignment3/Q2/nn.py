import numpy as np

np.random.seed(42)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_x):
    return sigmoid_x * (1 - sigmoid_x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # avoid log(0)
    return -np.sum(y_true * np.log(y_pred)) / m


class Layer:
    def __init__(self, input_dim, output_dim, activation='sigmoid'):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(1. / input_dim)
        self.biases = np.zeros((1, output_dim))
        self.activation = activation

    def __repr__(self):
        return f"input_output = {self.weights.shape[0]} --> {self.weights.shape[1]}, activation={self.activation}"
        
    def forward(self, X):
        self.input = X
        self.linear_output = np.dot(X, self.weights) + self.biases
        if self.activation == 'sigmoid':
            self.output = sigmoid(self.linear_output)
        elif self.activation == 'relu':
            self.output = relu(self.linear_output)
        elif self.activation == 'softmax':
            self.output = softmax(self.linear_output)
        return self.output

    def backward(self, grad_output, learning_rate):
        if self.activation == 'sigmoid':
            grad_activation = sigmoid_derivative(self.output) * grad_output
        elif self.activation == 'relu':
            grad_activation = relu_derivative(self.linear_output) * grad_output
        elif self.activation == 'softmax':
            grad_activation = grad_output  # because softmax + CE has simplified gradient

        grad_weights = np.dot(self.input.T, grad_activation)
        grad_biases = np.sum(grad_activation, axis=0, keepdims=True)
        grad_input = np.dot(grad_activation, self.weights.T)

        # update weights
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

class NeuralNetwork:
    def __init__(self, input_dim, hidden_layers, output_dim, activation=None, adaptive=False):
        self.layers = []
        self.adaptive = adaptive
        if activation is None:
            activation = ['sigmoid'] * len(hidden_layers)
        dims = [input_dim] + hidden_layers
        for i in range(len(hidden_layers)):
            self.layers.append(Layer(dims[i], dims[i + 1], activation=activation))
        self.layers.append(Layer(dims[-1], output_dim, activation='softmax'))

        self.epsilon = 1e-8

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred, learning_rate):
        grad = y_pred - y_true  # correct gradient for softmax + cross-entropy
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(self, X, y, epochs = None, batch_size = 32, learning_rate = 0.01, log = False):
        num_samples = X.shape[0]
        if epochs is None:
            epochs = 1e5
        oldest_loss = float('inf')
        for epoch in range(epochs):
            perm = np.random.permutation(num_samples)
            X_shuffled, y_shuffled = X[perm], y[perm]
            epoch_loss = 0

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred = self.forward(X_batch)
                loss = cross_entropy_loss(y_batch, y_pred)
                epoch_loss += loss * X_batch.shape[0]  # sum for averaging
                if self.adaptive:
                    learning_rate_special = learning_rate / np.sqrt(epoch + 1)
                    self.backward(y_batch, y_pred, learning_rate_special)
                else:
                    self.backward(y_batch, y_pred, learning_rate)

            avg_loss = epoch_loss / num_samples
            if avg_loss < oldest_loss - self.epsilon:
                oldest_loss = avg_loss
            else:
                break
            if log:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def __repr__(self):
        desc = f"NeuralNetwork Architecture:\n"
        for i, l in enumerate(self.layers):
            desc += f"  Layer {i + 1}: {l}\n"
        return desc

    def predict(self, X):
        probs = self.forward(X)
        preds = np.argmax(probs, axis=1)
        return preds