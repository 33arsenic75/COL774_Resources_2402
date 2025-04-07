import numpy as np
import copy
# import nnfs
# from nnfs.datasets import spiral_data

LOGGING = 1

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

# SGD optimizer
class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates


    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Common loss class
class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    
    def predict(self, inputs):
        # Numerically stable softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

class DeepNeuralNetwork:
    def __init__(self, *args):
        self.input_dim = args[0]["input_dim"]
        self.hidden_dims = args[0]["hidden_dims"]
        self.output_dim = args[0]["output_dim"]
        self.learning_rate = args[0]["learning_rate"]
        self.batch_size = args[0]["batch_size"]
        self.activation = args[0]["activation"]
        self.input_layer = Layer_Dense(self.input_dim, self.hidden_dims[0])
        self.layers = [self.input_layer]
        for i in range(len(self.hidden_dims) - 1):
            layer = Layer_Dense(self.hidden_dims[i], self.hidden_dims[i + 1])
            self.layers.append(layer)
        self.output_layer = Layer_Dense(self.hidden_dims[-1], self.output_dim)
        # self.layers.append(self.output_layer)
        # Initialize activation functions
        if self.activation == "relu":
            self.activation_layer = [Activation_ReLU() for _ in range(len(self.layers))]
        elif self.activation == "sigmoid":
            self.activation_layer = [Activation_Softmax() for _ in range(len(self.layers))]
        
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        self.optimizer = Optimizer_SGD(learning_rate=self.learning_rate)
    
    def fit(self, X, y, epochs=1):
        for epoch in range(epochs):
            batch_count = X.shape[0] // self.batch_size
            total_loss = 0
            for batch in range(batch_count):
                # Get batch data
                X_batch = X[batch * self.batch_size:(batch + 1) * self.batch_size]
                y_batch = y[batch * self.batch_size:(batch + 1) * self.batch_size]
                input_ = copy.deepcopy(X_batch)
                for i in range(len(self.layers)):
                    layer = self.layers[i]
                    layer.forward(input_)
                    self.activation_layer[i].forward(layer.output)
                    input_ = layer.output

                self.output_layer.forward(input_)
                # Calculate loss
                loss = self.loss_activation.forward(self.output_layer.output, y_batch)         
                total_loss += loss
                # Backward pass

                # self.loss_activation.backward(self.output_layer.output, y_batch)
                # self.output_layer.backward(self.loss_activation.dinputs)    
                # prev_input = self.output_layer.dinputs
                # for i in range(len(self.layers)):
                #     layer = self.layers[len(self.layers) - 1 - i]
                #     self.activation_layer[i].backward(prev_input)
                #     layer.backward(self.activation_layer[i].dinputs)
                #     prev_input = layer.dinputs

                # Backward pass
                self.loss_activation.backward(self.output_layer.output, y_batch)
                self.output_layer.backward(self.loss_activation.dinputs)
                prev_input = self.output_layer.dinputs
                # Process layers and activations in reverse order
                for i in range(len(self.layers)):
                    layer_idx = len(self.layers) - 1 - i
                    activation = self.activation_layer[layer_idx]
                    activation.backward(prev_input)
                    self.layers[layer_idx].backward(activation.dinputs)
                    prev_input = self.layers[layer_idx].dinputs

                self.optimizer.pre_update_params()
                for layer in self.layers:
                    self.optimizer.update_params(layer)
                self.optimizer.update_params(self.output_layer)
                self.optimizer.post_update_params()
            total_loss /= len(X)
            if (epoch % 100 == 0) and LOGGING:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    
    def predict(self, X):
        input_ = copy.deepcopy(X)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.forward(input_)
            self.activation_layer[i].forward(layer.output)
            input_ = layer.output
        self.output_layer.forward(input_)
        prob_pred = self.loss_activation.predict(self.output_layer.output)
        pred = np.argmax(prob_pred, axis=1)
        return pred



# X, y = spiral_data(samples=100, classes=3)
from dataset import *
X_train, y_train, y_train_raw = get_train_data(NUM_CLASSES=43)
X_test, y_test, y_test_raw = get_test_data(NUM_CLASSES=43)
params = {
    "input_dim": X_train.shape[1],
    "hidden_dims": [512, 256, 128, 64],
    # "hidden_dims": [100],
    "output_dim": y_train.shape[1],
    "learning_rate": 0.01,
    "batch_size": 32,
    "activation": "sigmoid"
}

if LOGGING:
    print("Data Loaded")
model = DeepNeuralNetwork(params)
# print(y_test_raw)
model.fit(X_train, y_train_raw, epochs=10)

predictions = model.predict(X_train)
accuracy = np.mean(predictions == y_train_raw)
print(f"Train Accuracy: {accuracy * 100:.2f}%")
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test_raw)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
