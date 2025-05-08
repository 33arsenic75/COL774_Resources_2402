import numpy as np
import copy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import sys
import time
LOGGING = 1
np.random.seed(0)
NUM_CLASSES = 43
import numpy as np
import pandas as pd
import os
from PIL import Image
import numpy as np

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


def one_hot_encode(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1
    return np.eye(num_classes)[y]

def load_gtsrb_train_data(root_dir, img_size=None):
    X = []
    y = []

    for class_id in sorted(os.listdir(root_dir)):
        # print(class_id)
        class_path = os.path.join(root_dir, class_id)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            if img_file.endswith((".ppm", ".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert("RGB")  # Ensure 3 channels
                if img_size:
                    img = img.resize(img_size)
                img_array = np.array(img).flatten()  # Flatten to 1D
                X.append(img_array)
                y.append(int(class_id))
    

    X = np.array(X)
    return X, y

def load_gtsrb_test_from_csv(csv_path, images_dir, img_size=None):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    name = []
    for _, row in df.iterrows():
        img_filename = row['image']
        img_path = os.path.join(images_dir, img_filename)
        img_label = row['label']
        try:
            img = Image.open(img_path).convert("RGB")
            if img_size:
                img = img.resize(img_size)
            img_array = np.array(img).flatten()  # shape: (2352,)
            name.append(img_path)
            X.append(img_array)
            y.append(int(img_label))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    X = np.array(X)
    return X, y

def get_train_data(DATA_PATH, NUM_CLASSES=43):
    # Load the training data
    X_train, y_train_raw = load_gtsrb_train_data(DATA_PATH + "/train/")
    X_train = X_train.astype(np.float32) / 255.0
    y_train = np.array(y_train_raw)
    # y_train = one_hot_encode(y_train_raw, num_classes=NUM_CLASSES)
    return X_train, y_train

def get_test_data(DATA_PATH, NUM_CLASSES=43):
    csv_path = DATA_PATH + "/test_labels.csv"
    images_dir = DATA_PATH + "/test"
    X_test, y_test_raw = load_gtsrb_test_from_csv(csv_path, images_dir)
    X_test = X_test.astype(np.float32) / 255.0

    y_test = np.array(y_test_raw)
    # y_test = one_hot_encode(y_test_raw, num_classes=NUM_CLASSES)
    return X_test, y_test

def load_gtsrb_test_from_csv_new(images_dir, img_size=None):
    X = []
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".ppm", ".png", ".jpg", ".jpeg"))
    ])

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        if img_size:
            img = img.resize(img_size)
        img_array = np.array(img).flatten()
        X.append(img_array)

    X = np.array(X)
    return X
    
def get_train_data_new(DATA_PATH, NUM_CLASSES=43):
    X_train, y_train_raw = load_gtsrb_train_data(DATA_PATH)
    X_train = X_train.astype(np.float32) / 255.0
    y_train = np.array(y_train_raw)
    return X_train, y_train

def get_test_data_new(DATA_PATH, NUM_CLASSES=43):
    X_test = load_gtsrb_test_from_csv_new(DATA_PATH)
    X_test = X_test.astype(np.float32) / 255.0
    return X_test


"""Dense layer"""
class Layer_Dense:
    """Layer initialization"""
    def __init__(self, n_inputs, n_neurons, activation=None):
        """Initialize weights and biases"""
        if activation == 'sigmoid':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1. / n_inputs)
            self.biases = np.zeros((1, n_neurons))
        elif activation == 'relu':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
            self.biases = np.zeros((1, n_neurons))
        else:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        
    """Forward pass"""
    def forward(self, inputs):
        """Remember input values"""
        self.inputs = inputs
        """Calculate output values from inputs, weights and biases"""
        self.output = np.dot(inputs, self.weights) + self.biases

    """Backward pass"""
    def backward(self, dvalues):
        """Gradients on parameters"""
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        """Gradient on values"""
        self.dinputs = np.dot(dvalues, self.weights.T)

"""ReLU activation"""
class Activation_ReLU:

    """Forward pass"""
    def forward(self, inputs):
        """Remember input values"""
        self.inputs = inputs
        """Calculate output values from inputs"""
        self.output = np.maximum(0, inputs)

    """Backward pass"""
    def backward(self, dvalues):
        """Gradient on values"""
        self.dinputs = dvalues.copy()
        """Zero gradient where input was negative"""
        self.dinputs[self.inputs <= 0] = 0

"""Softmax activation"""
class Activation_Softmax:
    """Forward pass"""
    def forward(self, inputs):
        """Remember input values"""
        self.inputs = inputs
        """Get unnormalized probabilities"""
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        """Normalize them for each sample"""
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    """Backward pass"""
    def backward(self, dvalues):
        """Create uninitialized array"""
        self.dinputs = np.empty_like(dvalues)
        """Enumerate outputs and gradients"""
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            """Flatten output array"""
            single_output = single_output.reshape(-1, 1)
            """Calculate Jacobian matrix of the output"""
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            """Calculate sample-wise gradient"""
            """and add it to the array of sample gradients"""
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        """Clip input to prevent overflow"""
        clipped_inputs = np.clip(inputs, -100, 100)
        self.output = 1 / (1 + np.exp(-clipped_inputs))

    def backward(self, dvalues):
        """Derivative - calculate gradient"""
        self.dinputs = dvalues * (1 - self.output) * self.output

"""Optimizer base class"""
class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        """Initialize current learning rate"""
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.decay = decay

    """Call once before any parameter updates"""
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        else:
            pass

    """Update parameters"""
    def update_params(self, layer):
        weight_updates = -self.current_learning_rate * layer.dweights
        bias_updates = -self.current_learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates


    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

"""Common loss class"""
class Loss:
    """Calculates the data and regularization losses"""
    """given model output and ground truth values"""
    def calculate(self, output, y):
        """Calculate sample losses"""
        sample_losses = self.forward(output, y)
        """Calculate mean loss"""
        data_loss = np.mean(sample_losses)
        """Return loss"""
        return data_loss

"""Cross-entropy loss"""
class Loss_CategoricalCrossentropy(Loss):
    """Forward pass"""
    def forward(self, y_pred, y_true):
        """Number of samples in a batch"""
        samples = len(y_pred)
        """Clip both sides to not drag mean towards any value"""
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        """Probabilities for target values 
        only if categorical labels"""
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            """Mask values - only for one-hot encoded labels"""
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        """Losses"""
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    """Backward pass"""
    def backward(self, dvalues, y_true):

        """Number of samples"""
        samples = len(dvalues)
        """Number of labels in every sample
        We'll use the first sample to count them"""
        labels = len(dvalues[0])
        """ If labels are sparse, turn them into one-hot vector"""
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        """Gradient dA = -y_true / y_pred"""
        self.dinputs = -y_true / dvalues
        """Normalize gradient"""
        self.dinputs = self.dinputs / samples


"""Softmax activation + categorical cross-entropy loss"""
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    """Forward pass"""
    def forward(self, inputs, y_true):
        """Output layer's activation function"""
        self.activation.forward(inputs)
        """Set the output"""
        self.output = self.activation.output
        """Calculate and return loss value"""
        return self.loss.calculate(self.output, y_true)

    """Backward pass"""
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        """If labels are one-hot encoded,
        turn them into discrete values"""
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        """Copy so we can safely modify"""
        self.dinputs = dvalues.copy()
        """Calculate gradient"""
        self.dinputs[range(samples), y_true] -= 1
        """Normalize gradient"""
        self.dinputs = self.dinputs / samples
    
    def predict(self, inputs):
        """Numerically stable softmax"""
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        """Normalize them for each sample"""
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
        
        self.optimizer = Optimizer_SGD(learning_rate=self.learning_rate)
        

    
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


def q2_b_analysis():
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
    end_time = time.time()
    time_taken = (end_time - start_time)/60
    print(f"Time taken: {time_taken:.2f} minutes")

def q2_c_analysis():
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


def q2_d_analysis():
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



def q2_e_analysis():
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


def q2_f_analysis():
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
    
def q2_b(train_data_path, test_data_path):
    X_train, y_train = get_train_data_new(train_data_path,NUM_CLASSES=NUM_CLASSES)
    X_test = get_test_data_new(test_data_path,NUM_CLASSES=NUM_CLASSES)
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
    EPOCHS = 200
    hidden_layer = [512, 256]
    lst = [25, 50, 75, 100, 125, 150, 155, 160, 165]
    params["hidden_dims"] = hidden_layer
    model = DeepNeuralNetwork(params)
    model.fit(X_train, y_train, epochs=EPOCHS, lst=lst)
    y_pred = model.predict(X_test)
    return y_pred

def q2_c(train_data_path, test_data_path):
    X_train, y_train = get_train_data_new(train_data_path,NUM_CLASSES=NUM_CLASSES)
    X_test = get_test_data_new(test_data_path,NUM_CLASSES=NUM_CLASSES)
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
    EPOCHS = 200
    hidden_layer = [512, 256]
    lst = [25, 50, 75, 100, 125, 150, 155, 160, 165]
    params["hidden_dims"] = hidden_layer
    model = DeepNeuralNetwork(params)
    model.fit(X_train, y_train, epochs=EPOCHS, lst=lst)
    y_pred = model.predict(X_test)
    return y_pred


def q2_d(train_data_path, test_data_path):
    X_train, y_train = get_train_data_new(train_data_path,NUM_CLASSES=NUM_CLASSES)
    X_test = get_test_data_new(test_data_path,NUM_CLASSES=NUM_CLASSES)
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
    EPOCHS = 200
    hidden_layer = [512, 256]
    lst = [25, 50, 75, 100, 125, 150, 155, 160, 165]
    params["hidden_dims"] = hidden_layer
    model = DeepNeuralNetwork(params)
    model.fit(X_train, y_train, epochs=EPOCHS, lst=lst)
    y_pred = model.predict(X_test)
    return y_pred


def q2_e(train_data_path, test_data_path):
    X_train, y_train = get_train_data_new(train_data_path,NUM_CLASSES=NUM_CLASSES)
    X_test = get_test_data_new(test_data_path,NUM_CLASSES=NUM_CLASSES)
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
    EPOCHS = 200
    hidden_layer = [512, 256]
    lst = [25, 50, 75, 100, 125, 150, 155, 160, 165]
    params["hidden_dims"] = hidden_layer
    model = DeepNeuralNetwork(params)
    model.fit(X_train, y_train, epochs=EPOCHS, lst=lst)
    y_pred = model.predict(X_test)
    return y_pred


def q2_f(train_data_path, test_data_path):
    X_train, y_train = get_train_data_new(train_data_path,NUM_CLASSES=NUM_CLASSES)
    X_test = get_test_data_new(test_data_path,NUM_CLASSES=NUM_CLASSES)
    from sklearn.neural_network import MLPClassifier
    # Replace with architecture from part (c), e.g., one hidden layer of 100 units
    # hidden_layers = [ (512,), (512, 256,), (512, 256, 128), (512, 256, 128, 64) ]
    hidden_layer = (512, 256, 128, 64) 

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
    y_pred_test = mlp.predict(X_test)
    return y_pred_test

    
    
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python neural_network.py <train_data_path> <test_data_path> <output_folder_path> <question_part>")
        sys.exit(1)

    train_data_path = str(sys.argv[1])
    test_data_path = str(sys.argv[2])
    output_folder_path = str(sys.argv[3])
    question_part = str(sys.argv[4])

    # print(f"train_data_path: {train_data_path}")
    # print(f"test_data_path: {test_data_path}")
    # print(f"output_folder_path: {output_folder_path}")
    # print(f"question_part: {question_part}")

    if question_part not in ['b', 'c', 'd', 'e', 'f']:
        print("Invalid question part. Please provide a number between 1 and 5.")
        sys.exit(1)
    
    question_part = question_part.lower()
    
    y_pred = None
    if question_part == 'b':
        y_pred = q2_b(train_data_path, test_data_path)
    elif question_part == 'c':
        y_pred = q2_c(train_data_path, test_data_path)
    elif question_part == 'd':
        y_pred = q2_d(train_data_path, test_data_path)
    elif question_part == 'e':
        y_pred = q2_e(train_data_path, test_data_path)
    elif question_part == 'f':
        y_pred = q2_f(train_data_path, test_data_path)

    if y_pred is not None:
        df = pd.DataFrame(y_pred, columns=['prediction'])
        output_file_path = f"{output_folder_path}/prediction_{question_part}.csv"
        df.to_csv(output_file_path, index=False)
        print(f"Output saved to {output_file_path}")

    
    

    