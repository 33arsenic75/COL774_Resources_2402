import numpy as np
import copy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from dataset import *
import sys
import time
LOGGING = 1
np.random.seed(0)
NUM_CLASSES = 43

question_wise_activation = {
    'b' : 'sigmoid',
    'c' : 'sigmoid',
    'd' : 'sigmoid',
    'e' : 'relu',
    
}

question_wise_epoch = {
    'b' : 330,
    'c' : 300,
    'd' : 300,
    'e' : 300,
}
# b, 100 , 20, 180
# b, 50 , 10, 180
# b, 10, 5, 180
# c, 512, 20, 200 , Epoch 174/1000, Loss: 0.0004, Accuracy: 73.44%, Part-wise Accuracy: 67.08%, Batch Size: 8192



# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons, activation=None):
        # Initialize weights and biases
        if activation == 'sigmoid':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1. / n_inputs)
            self.biases = np.random.randn(1, n_neurons) * np.sqrt(1. / n_inputs)
        elif activation == 'relu':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
            self.biases = np.random.randn(1, n_neurons) * np.sqrt(2. / n_inputs)
        else:
            # print("default")
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = 0.01 * np.random.randn(1, n_neurons)
        
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

class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Clip input to prevent overflow
        clipped_inputs = np.clip(inputs, -100, 100)
        self.output = 1 / (1 + np.exp(-clipped_inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculate gradient
        self.dinputs = dvalues * (1 - self.output) * self.output

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

# Adam optimizer
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

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
        self.initializer = args[0]["initializer"]

        self.batch_size_double = args[0]["batch_size_double"]

        self.input_layer = Layer_Dense(self.input_dim, self.hidden_dims[0], activation=self.initializer)
        self.layers = [self.input_layer]

        for i in range(len(self.hidden_dims) - 1):
            layer = Layer_Dense(self.hidden_dims[i], self.hidden_dims[i + 1], activation=self.initializer)
            self.layers.append(layer)
        self.output_layer = Layer_Dense(self.hidden_dims[-1], self.output_dim, activation=self.initializer)
        
        # Initialize activation functions
        if self.activation == "relu":
            self.activation_layer = [Activation_ReLU() for _ in range(len(self.layers))]
        elif self.activation == "sigmoid":
            self.activation_layer = [Activation_Softmax() for _ in range(len(self.layers))]
            # self.activation_layer = [Activation_Sigmoid() for _ in range(len(self.layers))]
        
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        
        self.optimzer_name = args[0]["optimizer"]
        if self.optimzer_name == "sgd":
            self.optimizer = Optimizer_SGD(learning_rate=self.learning_rate)
        else:
            self.optimizer = Optimizer_Adam(learning_rate=self.learning_rate)

    
    def fit(self, X, y, epochs=1, adaptive_lr = False, lst = []):

        if adaptive_lr:
            self.learning_rate = self.learning_rate * np.sqrt(2)

        for epoch in range(epochs):
            batch_count = X.shape[0] // self.batch_size

            total_loss = 0
            total_correct = 0
            if adaptive_lr:
                self.learning_rate = self.learning_rate * np.sqrt( (epoch + 1) / (epoch + 2))

            for batch in range(batch_count):
                X_batch = X[batch * self.batch_size:(batch + 1) * self.batch_size]
                y_batch = y[batch * self.batch_size:(batch + 1) * self.batch_size]
                input_ = copy.deepcopy(X_batch)
                for i in range(len(self.layers)):
                    layer = self.layers[i]
                    layer.forward(input_)
                    self.activation_layer[i].forward(layer.output)
                    input_ = layer.output
                self.output_layer.forward(input_)
                loss = self.loss_activation.forward(self.output_layer.output, y_batch)    

                total_loss += loss
                
                predictions = np.argmax(self.output_layer.output, axis=1)
                if len(y_batch.shape) == 2:
                    y_batch = np.argmax(y_batch, axis=1)
                total_correct += np.sum(predictions == y_batch)

                self.loss_activation.backward(self.output_layer.output, y_batch)
                self.output_layer.backward(self.loss_activation.dinputs)
                prev_input = self.output_layer.dinputs
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

        
            predictions = self.predict(X)
            accuracy = np.mean(predictions==y)
            total_loss /= len(X)
            part_wise_accuracy = total_correct / (len(X))
            if epoch % self.batch_size_double == 0 and epoch > 0:
                self.batch_size *= 2
                self.batch_size = min(self.batch_size, X.shape[0])
            
            if epoch in lst:
                self.batch_size *= 2
                self.batch_size = min(self.batch_size, X.shape[0])

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Part-wise Accuracy: {part_wise_accuracy * 100:.2f}%, Batch Size: {self.batch_size}")
            
    
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
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred, average='macro', zero_division=0)
        precision = precision_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        return accuracy, recall, precision, f1


def q2_b():
    start_time = time.time()
    X_train, y_train = get_train_data(NUM_CLASSES=NUM_CLASSES)
    X_test, y_test = get_test_data(NUM_CLASSES=NUM_CLASSES)
    params = {
        "input_dim": X_train.shape[1],
        "output_dim": NUM_CLASSES,
        "learning_rate": 0.05,
        "batch_size": 64,
        "optimizer": "sgd",
        "initializer": "sigmoid",
        "batch_size_double": 250,
    }

    params["activation"] = question_wise_activation[question_part]
    # EPOCHS = question_wise_epoch[question_part]
    EPOCHS = 200

    # hidden_layers = [[1], [5], [10], [50], [100]]
    # hidden_layers = [[100], [50], [10], [5], [1]]
    hidden_layers = [[512, 256]]
    
    
    lst = [25, 50, 75, 100, 125, 150, 155, 160, 165]
    print("Doing B")
    for hidden_layer in hidden_layers:
        params["hidden_dims"] = hidden_layer
        print(f"hidden_layers: {hidden_layer}")
        model = DeepNeuralNetwork(params)
        model.fit(X_train, y_train, epochs=EPOCHS, lst=lst)
        accuracy, recall, precision, f1 = model.score(X_train, y_train)
        y_pred_train = model.predict(X_train)
        print(f"hidden_layers: {hidden_layer}")
        print(f"Train Accuracy: {accuracy * 100:.2f}%")
        print(f"Train Recall: {recall * 100:.2f}%")
        print(f"Train Precision: {precision * 100:.2f}%")
        print(f"Train F1: {f1 * 100:.2f}%")
        accuracy, recall, precision, f1 = model.score(X_test, y_test)
        y_pred_test = model.predict(X_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Recall: {recall * 100:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}%")
        print(f"Test F1: {f1 * 100:.2f}%")
        precision_train_per_class = precision_score(y_train, y_pred_train, average=None)
        recall_train_per_class = recall_score(y_train, y_pred_train, average=None)
        f1_train_per_class = f1_score(y_train, y_pred_train, average=None)
        precision_test_per_class = precision_score(y_test, y_pred_test, average=None)
        recall_test_per_class = recall_score(y_test, y_pred_test, average=None)
        f1_test_per_class = f1_score(y_test, y_pred_test, average=None)

        for i, (p, r, f) in enumerate(zip(precision_train_per_class, recall_train_per_class, f1_train_per_class)):
            print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
        for i, (p, r, f) in enumerate(zip(precision_test_per_class, recall_test_per_class, f1_test_per_class)):
            print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
    end_time = time.time()
    time_taken = (end_time - start_time)/60
    print(f"Time taken: {time_taken:.2f} minutes")

def q2_c():
    X_train, y_train = get_train_data(NUM_CLASSES=NUM_CLASSES)
    X_test, y_test = get_test_data(NUM_CLASSES=NUM_CLASSES)
    params = {
        "input_dim": X_train.shape[1],
        "output_dim": NUM_CLASSES,
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "sgd",
        "initializer": "sigmoid",
        # "initializer": "default",
        "batch_size_double": 40,
    }
    params["activation"] = question_wise_activation[question_part]
    EPOCHS = question_wise_epoch[question_part]

    # hidden_layers = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
    # hidden_layers = [[512, 256, 128, 64], [512, 256, 128], [512, 256], [512]]
    # hidden_layers = [[512, 256, 128, 64],  [512]]
    # hidden_layers = [[512, 256, 128]]
    hidden_layers = [[512]]
    
    # lst = [40, 80, 150, 200, 250, 300, 350, 400, 425, 450]
    lst = []
    for hidden_layer in hidden_layers:
        params["hidden_dims"] = hidden_layer
        model = DeepNeuralNetwork(params)
        model.fit(X_train, y_train, epochs=EPOCHS, lst=lst)
        accuracy, recall, precision, f1 = model.score(X_train, y_train)
        print(f"hidden_layers: {hidden_layer}")
        print(f"Train Accuracy: {accuracy * 100:.2f}%")
        print(f"Train Recall: {recall * 100:.2f}%")
        print(f"Train Precision: {precision * 100:.2f}%")
        print(f"Train F1: {f1 * 100:.2f}%")
        accuracy, recall, precision, f1 = model.score(X_test, y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Recall: {recall * 100:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}%")
        print(f"Test F1: {f1 * 100:.2f}%")


def q2_d():
    X_train, y_train = get_train_data(NUM_CLASSES=NUM_CLASSES)
    X_test, y_test = get_test_data(NUM_CLASSES=NUM_CLASSES)
    params = {
        "input_dim": X_train.shape[1],
        "output_dim": NUM_CLASSES,
        "learning_rate": 10,
        "batch_size": 64,
        "optimizer": "sgd",
        "initializer": "sigmoid",
        # "initializer": "default",
        "batch_size_double": 40,
    }
    params["activation"] = question_wise_activation[question_part]
    EPOCHS = question_wise_epoch[question_part]

    # hidden_layers = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
    # hidden_layers = [[512, 256, 128, 64], [512, 256, 128], [512, 256], [512]]
    # hidden_layers = [[512, 256, 128, 64],  [512]]
    # hidden_layers = [[512, 256, 128]]
    hidden_layers = [[512]]
    
    # lst = [40, 80, 150, 200, 250, 300, 350, 400, 425, 450]
    lst = []
    for hidden_layer in hidden_layers:
        params["hidden_dims"] = hidden_layer
        model = DeepNeuralNetwork(params)
        model.fit(X_train, y_train, epochs=EPOCHS, lst=lst, adaptive_lr=True)
        accuracy, recall, precision, f1 = model.score(X_train, y_train)
        print(f"hidden_layers: {hidden_layer}")
        print(f"Train Accuracy: {accuracy * 100:.2f}%")
        print(f"Train Recall: {recall * 100:.2f}%")
        print(f"Train Precision: {precision * 100:.2f}%")
        print(f"Train F1: {f1 * 100:.2f}%")
        accuracy, recall, precision, f1 = model.score(X_test, y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Recall: {recall * 100:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}%")
        print(f"Test F1: {f1 * 100:.2f}%")



def q2_e():
    X_train, y_train = get_train_data(NUM_CLASSES=NUM_CLASSES)
    X_test, y_test = get_test_data(NUM_CLASSES=NUM_CLASSES)
    params = {
        "input_dim": X_train.shape[1],
        "output_dim": NUM_CLASSES,
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "sgd",
        "initializer": "relu",
        "batch_size_double": 1000,
    }
    params["activation"] = question_wise_activation[question_part]
    EPOCHS = question_wise_epoch[question_part]

    # hidden_layers = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
    # hidden_layers = [[512, 256, 128, 64], [512, 256, 128], [512, 256], [512]]
    # hidden_layers = [[512, 256, 128, 64],  [512]]
    hidden_layers = [[512, 256, 128]]
    
    # lst = [80, 160, 300, 400, 500, 600, 700, 800, 900, 950]
    # lst = [150, 200, 250, 300, 350, 400, 450, 500, 550]
    lst = [75, 100, 125, 150, 175, 200, 225, 250, 275]
    for hidden_layer in hidden_layers:
        params["hidden_dims"] = hidden_layer
        model = DeepNeuralNetwork(params)
        model.fit(X_train, y_train, epochs=EPOCHS, lst=lst)
        accuracy, recall, precision, f1 = model.score(X_train, y_train)
        print(f"hidden_layers: {hidden_layer}")
        print(f"Train Accuracy: {accuracy * 100:.2f}%")
        print(f"Train Recall: {recall * 100:.2f}%")
        print(f"Train Precision: {precision * 100:.2f}%")
        print(f"Train F1: {f1 * 100:.2f}%")
        accuracy, recall, precision, f1 = model.score(X_test, y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Recall: {recall * 100:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}%")
        print(f"Test F1: {f1 * 100:.2f}%")


def q2_f():
    X_train, y_train = get_train_data(NUM_CLASSES=NUM_CLASSES)
    X_test, y_test = get_test_data(NUM_CLASSES=NUM_CLASSES)
    from sklearn.neural_network import MLPClassifier
    # Replace with architecture from part (c), e.g., one hidden layer of 100 units
    # hidden_layers = [ (512,), (512, 256,), (512, 256, 128), (512, 256, 128, 64) ]
    hidden_layers = [ (512, 256, 128, 64) ]

    for hidden_layer in hidden_layers:
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer,
            activation='relu',
            solver='sgd',
            alpha=0.0,
            batch_size=32,
            learning_rate='invscaling',
            max_iter=200,  # You can change this or set early_stopping=True
            verbose=True,
            random_state=42
        )
        mlp.fit(X_train, y_train)
        y_pred_train = mlp.predict(X_train)
        print(f"hidden_layers: {hidden_layer}")
        print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train) * 100:.2f}%")
        print(f"Train Recall: {recall_score(y_train, y_pred_train, average='macro', zero_division=0) * 100:.2f}%")
        print(f"Train Precision: {precision_score(y_train, y_pred_train, average='macro', zero_division=0) * 100:.2f}%")
        print(f"Train F1: {f1_score(y_train, y_pred_train, average='macro',zero_division=0) * 100:.2f}%")
        y_pred_test = mlp.predict(X_test)
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test) * 100:.2f}%")
        print(f"Test Recall: {recall_score(y_test, y_pred_test, average='macro',zero_division=0) * 100:.2f}%")
        print(f"Test Precision: {precision_score(y_test, y_pred_test, average='macro', zero_division=0) * 100:.2f}%")
        print(f"Test F1: {f1_score(y_test, y_pred_test, average='macro',zero_division=0) * 100:.2f}%")
    

    
    


if __name__ == '__main__':
    # if len(sys.argv) != 6:
    #     print("Usage: python decision_tree.py <train_data_path> <validation_data_path> <test_data_path> <output_folder_path> <question_part>")
    #     sys.exit(1)
    
    question_part = sys.argv[1]
    if question_part not in ['b', 'c', 'd', 'e', 'f']:
        print("Invalid question part. Please provide a number between 1 and 5.")
        sys.exit(1)

    if question_part == 'b':
        q2_b()
    
    elif question_part == 'c':
        q2_c()
    
    elif question_part == 'd':
        q2_d()
    
    elif question_part == 'e':
        q2_e()
    
    elif question_part == 'f':
        q2_f()

    # y_pred = None

    # if y_pred is not None:
    #     df = pd.DataFrame(y_pred, columns=['prediction'])
    #     # df['prediction'] = df['prediction'].map({0: "<=50K", 1: ">50K"})
    #     # output_file_path = f"{output_folder_path}/prediction_{question_part}.csv"
    #     # df.to_csv(output_file_path, index=False)
    #     # print(f"Output saved to {output_file_path}")
