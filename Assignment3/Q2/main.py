from nn import *
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def q2_2():
    n = 2352  
    r = 43               
    M = 32        
    hidden_archs = [[1], [5], [10], [50], [100]]
    results = {}
    activation = 'sigmoid'
    for hidden_arch in hidden_archs:
        X_train, y_train = get_train_data(NUM_CLASSES=r)
        X_test, y_test, y_test_raw = get_test_data(NUM_CLASSES=r)
        nn = NeuralNetwork(input_dim=n, hidden_layers=hidden_arch, output_dim=r, activation=activation)
        nn.train(X_train, y_train, epochs=200, batch_size=M, learning_rate=0.01)
        y_pred_labels = nn.predict(X_test)
        accuracy = accuracy_score(y_test_raw, y_pred_labels)
        precision = precision_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        # print(f"Hidden Architecture: {hidden_arch}")
        # print(f"Test Accuracy: {accuracy * 100:.2f}%")
        # print(f"Test Precision: {precision * 100:.2f}%")
        # print(f"Test Recall: {recall * 100:.2f}%")
        # print(f"Test F1 Score: {f1 * 100:.2f}%")
        results[str(hidden_arch)] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    print("Architecture, Accuracy, Precision, Recall, F1 Score")
    for arch, metrics in results.items():
        print(f"{arch}, {metrics['accuracy']:.4f}, {metrics['precision']:.4f}, {metrics['recall']:.4f}, {metrics['f1_score']:.4f}")


def q2_3():
    n = 2352  
    r = 43               
    M = 32        
    hidden_archs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
    results = {}
    activation = 'sigmoid'
    for hidden_arch in hidden_archs:
        X_train, y_train = get_train_data(NUM_CLASSES=r)
        X_test, y_test, y_test_raw = get_test_data(NUM_CLASSES=r)
        nn = NeuralNetwork(input_dim=n, hidden_layers=hidden_arch, output_dim=r, activation=activation)
        nn.train(X_train, y_train, epochs=200, batch_size=M, learning_rate=0.01)
        y_pred_labels = nn.predict(X_test)
        accuracy = accuracy_score(y_test_raw, y_pred_labels)
        precision = precision_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        print(f"Hidden Architecture: {hidden_arch}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}%")
        print(f"Test Recall: {recall * 100:.2f}%")
        print(f"Test F1 Score: {f1 * 100:.2f}%")
        results[str(hidden_arch)] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    print("Architecture, Accuracy, Precision, Recall, F1 Score")
    for arch, metrics in results.items():
        print(f"{arch}, {metrics['accuracy']:.4f}, {metrics['precision']:.4f}, {metrics['recall']:.4f}, {metrics['f1_score']:.4f}")

def q2_4():
    n = 2352  
    r = 43               
    M = 32        
    hidden_archs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
    results = {}
    activation = 'sigmoid'
    for hidden_arch in hidden_archs:
        X_train, y_train = get_train_data(NUM_CLASSES=r)
        X_test, y_test, y_test_raw = get_test_data(NUM_CLASSES=r)
        nn = NeuralNetwork(input_dim=n, hidden_layers=hidden_arch, output_dim=r, activation=activation, adaptive=True)
        nn.train(X_train, y_train, epochs=200, batch_size=M, learning_rate=0.01)
        y_pred_labels = nn.predict(X_test)
        accuracy = accuracy_score(y_test_raw, y_pred_labels)
        precision = precision_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        print(f"Hidden Architecture: {hidden_arch}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}%")
        print(f"Test Recall: {recall * 100:.2f}%")
        print(f"Test F1 Score: {f1 * 100:.2f}%")
        results[str(hidden_arch)] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    print("Architecture, Accuracy, Precision, Recall, F1 Score")
    for arch, metrics in results.items():
        print(f"{arch}, {metrics['accuracy']:.4f}, {metrics['precision']:.4f}, {metrics['recall']:.4f}, {metrics['f1_score']:.4f}")

def q2_5():
    n = 2352  
    r = 43               
    M = 32        
    hidden_archs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
    results = {}
    activation = 'relu'
    for hidden_arch in hidden_archs:
        X_train, y_train = get_train_data(NUM_CLASSES=r)
        X_test, y_test, y_test_raw = get_test_data(NUM_CLASSES=r)
        nn = NeuralNetwork(input_dim=n, hidden_layers=hidden_arch, output_dim=r, activation=activation, adaptive=True)
        nn.train(X_train, y_train, epochs=200, batch_size=M, learning_rate=0.01)
        y_pred_labels = nn.predict(X_test)
        accuracy = accuracy_score(y_test_raw, y_pred_labels)
        precision = precision_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        print(f"Hidden Architecture: {hidden_arch}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}%")
        print(f"Test Recall: {recall * 100:.2f}%")
        print(f"Test F1 Score: {f1 * 100:.2f}%")
        results[str(hidden_arch)] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    print("Architecture, Accuracy, Precision, Recall, F1 Score")
    for arch, metrics in results.items():
        print(f"{arch}, {metrics['accuracy']:.4f}, {metrics['precision']:.4f}, {metrics['recall']:.4f}, {metrics['f1_score']:.4f}")

def q2_6():
    n = 2352  
    r = 43               
    M = 32        
    hidden_archs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
    results = {}
    activation = 'relu'

    for hidden_arch in hidden_archs:
        X_train, y_train = get_train_data(NUM_CLASSES=r)
        X_test, y_test, y_test_raw = get_test_data(NUM_CLASSES=r)

        nn = MLPClassifier(
            hidden_layer_sizes=hidden_arch,
            activation='relu',
            solver='sgd',
            alpha=0,
            batch_size=32,
            learning_rate='invscaling',
            max_iter=200,
            random_state=42,
            verbose=True
        )

        # Fit the model before predicting
        nn.fit(X_train, y_train)

        y_pred_labels = nn.predict(X_test)

        accuracy = accuracy_score(y_test_raw, y_pred_labels)
        precision = precision_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test_raw, y_pred_labels, average='weighted', zero_division=0)

        print(f"Hidden Architecture: {hidden_arch}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}%")
        print(f"Test Recall: {recall * 100:.2f}%")
        print(f"Test F1 Score: {f1 * 100:.2f}%")

        results[str(hidden_arch)] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    print("Architecture, Accuracy, Precision, Recall, F1 Score")
    for arch, metrics in results.items():
        print(f"{arch}, {metrics['accuracy']:.4f}, {metrics['precision']:.4f}, {metrics['recall']:.4f}, {metrics['f1_score']:.4f}")


# q2_2()
# q2_3()
# q2_4()
# q2_5()
q2_6()