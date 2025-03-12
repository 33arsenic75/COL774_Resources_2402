import cv2
import numpy as np
import os
from svm import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from sklearn.linear_model import SGDClassifier 
import pandas as pd
import seaborn as sns

ENTRYNUMBER = 11596
DIRECTORY_PATH = "../data/Q2/dataset/"
TOTAL_CLASSES = 3
cv2.setLogLevel(0)

def preprocess_images(images):
    cv2.setLogLevel(0)
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, (100, 100))  # Resize to 100x100
        img_cropped = img_resized  # Center cropping is redundant here as resizing achieves uniformity
        img_flattened = img_cropped.flatten() / 255.0  # Normalize to [0,1] and flatten
        processed_images.append(img_flattened)
    return np.array(processed_images)

def plot_images(images, titles, cmap=None, q_num = "q2_1"):
    """
    Helper function to plot multiple images in a row.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap=cmap)
        ax.set_title(titles[i])
        ax.axis('off')
    plt.draw()
    plt.savefig(f"support_vectors_{q_num}.png")

def get_folder_images(path, lst, valid_extensions=(".jpg", ".png", ".jpeg", ".bmp")):
    cv2.setLogLevel(0)
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folders.sort()
    folder_images = {}
    for n in lst:
        try:
            folder_index = n % len(folders)
            folder_name = folders[folder_index]
            folder_path = os.path.join(path, folder_name)
            images = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(valid_extensions):  # Check for valid image extensions
                    img_path = os.path.join(folder_path, file)
                    img = cv2.imread(img_path)  # Read image
                    if img is not None:
                        images.append(img)
            
            folder_images[folder_name] = images
        
        except IndexError:
            raise IndexError(f"Index {n} is out of bounds. There are only {len(folders)} folders.")
    
    return folder_images

def convert_to_X_y(flattened_images, name_to_key):
    X = []
    y = []
    keys = list(name_to_key.keys())  # Extract the two keys
    key1, key2 = keys[0], keys[1]  # Assign them to variables

    name_to_key_2 = {
        key1: 1 if name_to_key[key1] > name_to_key[key2] else -1,
        key2: 1 if name_to_key[key2] > name_to_key[key1] else -1
    }
    for key in flattened_images:
        X.extend(flattened_images[key])
        y.extend([name_to_key_2[key]]*len(flattened_images[key]))
    return np.array(X), np.array(y)


def train(lst):
    lst = [ (x%100)%11 for x in lst]
    # print(lst)
    dict = get_folder_images(DIRECTORY_PATH, lst)
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    X, y = convert_to_X_y(flattened_images, name_to_key)
    # print(X.shape, y.shape)
    model = SupportVectorMachine()
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)    
    print(accuracy)
    return model

def q2_1():
    start_time = time.time()
    lst = [ENTRYNUMBER, ENTRYNUMBER + 1]
    dict = get_folder_images(DIRECTORY_PATH, lst)
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    X, y = convert_to_X_y(flattened_images, name_to_key)
    # Train the model
    model = SupportVectorMachine()
    model.fit(X, y)  # Assuming X_train and y_train are defined
    # print(y[0:10])
    # Get the number of support vectors
    num_support_vectors = len(model.support_vectors)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)  
    # Get the percentage of training samples that are support vectors
    total_samples = len(y)
    percentage_support_vectors = (num_support_vectors / total_samples) * 100
    end_time = time.time()
    print("--"*20)
    print(f"Number of support vectors: {num_support_vectors}")
    print(f"Percentage of training samples that are support vectors: {percentage_support_vectors:.2f}%")
    print(f"Training accuracy: {accuracy:.2f}")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    # print(f"Weights: {model.w}")
    # print(f"Bias: {model.b}")
    print("--"*20)
    
    top_5_indices = np.argsort(-model.alphas)[:5]  # Get indices of top-5 largest alphas
    top_5_support_vectors = model.support_vectors[top_5_indices]  # Extract corresponding support vectors

    top_5_images = [sv.reshape(100, 100, 3) for sv in top_5_support_vectors]
    plot_images(top_5_images, [f"Support Vector q2_1 {i+1}" for i in range(5)], q_num="q2_1")
    del model

def q2_2():
    start_time = time.time()
    lst = [ENTRYNUMBER, ENTRYNUMBER + 1]
    dict = get_folder_images(DIRECTORY_PATH, lst)
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    X, y = convert_to_X_y(flattened_images, name_to_key)
    # Train the model
    model = SupportVectorMachine()
    model.fit(X, y, kernel = 'gaussian', C = 1.0, gamma=0.001)  # Assuming X_train and y_train are defined
    # print(y[0:10])
    # Get the number of support vectors
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)  
    total_samples = len(y)
    num_support_vectors = len(model.support_vectors)
    percentage_support_vectors = (num_support_vectors / total_samples) * 100
    # Get the percentage of training samples that are support vectors
    total_samples = len(y)
    end_time = time.time()
    print("--"*20)
    print(f"Percentage of training samples that are support vectors: {percentage_support_vectors:.2f}%")
    print(f"Training accuracy: {accuracy:.2f}")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("--"*20)
    top_5_indices = np.argsort(-model.alphas)[:5]  # Get indices of top-5 largest alphas
    top_5_support_vectors = model.support_vectors[top_5_indices]  # Extract corresponding support vectors

    top_5_images = [sv.reshape(100, 100, 3) for sv in top_5_support_vectors]
    plot_images(top_5_images, [f"Support Vector q2_2 {i+1}" for i in range(5)], q_num = "q2_2")
    del model

def q2_3acd():
    lst = [ENTRYNUMBER, ENTRYNUMBER + 1]

    start_time = time.time()
    dict = get_folder_images(DIRECTORY_PATH, lst)
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    X, y = convert_to_X_y(flattened_images, name_to_key)
    model = SVC(kernel='linear', C = 1.0)   
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    num_support_vectors = sum(model.n_support_)
    support_vectors = model.support_vectors_
    percentage_support_vectors = (np.sum(num_support_vectors) / len(y)) * 100
    end_time = time.time()
    print("--"*20)
    print("Linear Kernel")  
    print(f"Number of support vectors: {num_support_vectors}")
    print(f"Percentage of training samples that are support vectors: {percentage_support_vectors:.2f}%")
    print(f"Training accuracy: {accuracy:.2f}") 
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("--"*20)

    del model

    start_time = time.time()
    model = SVC(kernel='rbf', C = 1.0, gamma=0.001)   
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    num_support_vectors = sum(model.n_support_)
    support_vectors = model.support_vectors_
    percentage_support_vectors = (np.sum(num_support_vectors) / len(y)) * 100
    end_time = time.time()
    print("--"*20)
    print("Gaussian Kernel")  
    print(f"Number of support vectors: {num_support_vectors}")
    print(f"Percentage of training samples that are support vectors: {percentage_support_vectors:.2f}%")
    print(f"Training accuracy: {accuracy:.2f}") 
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("--"*20)
    del model

def q2_3b():
    lst = [ENTRYNUMBER, ENTRYNUMBER + 1]

    dict = get_folder_images(DIRECTORY_PATH, lst)
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    X, y = convert_to_X_y(flattened_images, name_to_key)
    model = SVC(kernel='linear', C = 1.0)   
    model.fit(X, y)

    weights_1 = model.coef_
    bias_1 = model.intercept_
    # print(weights_1, bias_1)
    del model

    model = SupportVectorMachine()
    model.fit(X, y)  # Assuming X_train and y_train are defined
    weights_2 = [model.w]
    bias_2 = [model.b]
    # print(weights_2, bias_2)
    del model
    weight_diff = np.linalg.norm(weights_1 - weights_2)
    bias_diff = np.linalg.norm(bias_1 - bias_2)
    total_l2_norm = np.sqrt(weight_diff**2 + bias_diff**2)
    print("L2 norm of weights difference:", weight_diff)
    print("L2 norm of bias difference:", bias_diff)
    print("L2 norm of the difference:", total_l2_norm)

def q2_4():
    lst = [ENTRYNUMBER, ENTRYNUMBER + 1]
    dict = get_folder_images(DIRECTORY_PATH, lst)
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    X, y = convert_to_X_y(flattened_images, name_to_key)
    libsvm = SVC(kernel='linear')

    start_time = time.time()
    libsvm.fit(X, y)
    libsvm_train_time = time.time() - start_time
    libsvm_acc = accuracy_score(y, libsvm.predict(X))
    sgd_svm = SGDClassifier(loss='hinge', learning_rate='optimal', max_iter=1000, random_state=42)

    start_time = time.time()
    sgd_svm.fit(X, y)
    sgd_train_time = time.time() - start_time

    sgd_acc = accuracy_score(y, sgd_svm.predict(X))

    print(f"LIBLINEAR: Accuracy = {libsvm_acc:.4f}, Training Time = {libsvm_train_time:.4f} sec")
    print(f"SGD SVM: Accuracy = {sgd_acc:.4f}, Training Time = {sgd_train_time:.4f} sec")

def q2_5():
    model = {}
    for i in range(TOTAL_CLASSES):
        for j in range(i+1, TOTAL_CLASSES):
            lst = [i , j]
            dict = get_folder_images(DIRECTORY_PATH, lst)
            flattened_images = {key: preprocess_images(dict[key]) for key in dict}
            name_to_key = {key: i for i, key in enumerate(flattened_images)}
            key_to_name = {i: key for i, key in enumerate(flattened_images)}

            X, y = convert_to_X_y(flattened_images, name_to_key)
            model[i, j] = SupportVectorMachine()
            model[i, j].fit(X, y, kernel = 'gaussian', C = 1.0, gamma=0.001)
            print(f"Trained model for {i} vs {j}")
    
    lst = [i for i in range(TOTAL_CLASSES)]
    dict = get_folder_images(DIRECTORY_PATH, lst)
    X, y = [], []
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    for key in flattened_images:
        X.extend(flattened_images[key])
        y.extend([name_to_key[key]]*len(flattened_images[key]))

    X = np.array(X)
    y = np.array(y)
    predictions = {}
    for i in range(TOTAL_CLASSES):
        for j in range(i+1, TOTAL_CLASSES):
            predictions[i, j] = model[i, j].predict(X)
            print(f"Predicted for {i} vs {j}")


    votes = np.zeros((X.shape[0], TOTAL_CLASSES), dtype=int)
    # Corrected vectorized voting process for -1 and 1 predictions
    for j in range(TOTAL_CLASSES):
        for k in range(j + 1, TOTAL_CLASSES):
            mask = predictions[j, k] == 1  # Boolean mask where class k is predicted
            votes[:, k] += mask  # Increase votes for class k when predicted
            votes[:, j] += ~mask  # Increase votes for class j when not predicted (bitwise NOT is incorrect for -1, so we use direct logic)

    # Get final predictions
    final_predictions = np.argmax(votes, axis=1)
    accuracy_score = np.mean(final_predictions == y)
    print(f"Accuracy: {accuracy_score}")
    map_labels = np.vectorize(lambda k: key_to_name[k])
    df = pd.DataFrame({
        'Ground Truth': map_labels(y),
        'Predictions': map_labels(final_predictions),
        'Misclassified': (final_predictions != y).astype(int)
    })
    df.to_csv("q2_5_multiclass_predictions.csv", index=False)

    cm = confusion_matrix(y, final_predictions)
    labels = [key_to_name[i] for i in range(TOTAL_CLASSES)]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6,4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("q2_5_confusion_matrix.png")

def q2_6():
    model = {}
    for i in range(TOTAL_CLASSES):
        for j in range(i+1, TOTAL_CLASSES):
            lst = [i , j]
            dict = get_folder_images(DIRECTORY_PATH, lst)
            flattened_images = {key: preprocess_images(dict[key]) for key in dict}
            name_to_key = {key: i for i, key in enumerate(flattened_images)}
            key_to_name = {i: key for i, key in enumerate(flattened_images)}

            X, y = convert_to_X_y(flattened_images, name_to_key)
            model[i, j] = SVC(kernel='rbf', C = 1.0, gamma=0.001)   
            model[i, j].fit(X, y)
            print(f"Trained model for {i} vs {j}")
    
    lst = [i for i in range(TOTAL_CLASSES)]
    dict = get_folder_images(DIRECTORY_PATH, lst)
    X, y = [], []
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    for key in flattened_images:
        X.extend(flattened_images[key])
        y.extend([name_to_key[key]]*len(flattened_images[key]))

    X = np.array(X)
    y = np.array(y)
    predictions = {}
    for i in range(TOTAL_CLASSES):
        for j in range(i+1, TOTAL_CLASSES):
            predictions[i, j] = model[i, j].predict(X)
            print(f"Predicted for {i} vs {j}")


    votes = np.zeros((X.shape[0], TOTAL_CLASSES), dtype=int)
    # Corrected vectorized voting process for -1 and 1 predictions
    for j in range(TOTAL_CLASSES):
        for k in range(j + 1, TOTAL_CLASSES):
            mask = predictions[j, k] == 1  # Boolean mask where class k is predicted
            votes[:, k] += mask  # Increase votes for class k when predicted
            votes[:, j] += ~mask  # Increase votes for class j when not predicted (bitwise NOT is incorrect for -1, so we use direct logic)

    # Get final predictions
    final_predictions = np.argmax(votes, axis=1)
    accuracy_score = np.mean(final_predictions == y)
    print(f"Accuracy: {accuracy_score}")
    map_labels = np.vectorize(lambda k: key_to_name[k])
    df = pd.DataFrame({
        'Ground Truth': map_labels(y),
        'Predictions': map_labels(final_predictions),
        'Misclassified': (final_predictions != y).astype(int)
    })
    df.to_csv("q2_6_multiclass_predictions.csv", index=False)

    cm = confusion_matrix(y, final_predictions)
    labels = [key_to_name[i] for i in range(TOTAL_CLASSES)]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6,4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("q2_6_confusion_matrix.png")



    
    



# dict = get_folder_images(DIRECTORY_PATH, [i for i in range(11)] )
# flattened_images = {key: preprocess_images(dict[key]) for key in dict}
# name_to_key = {key: i for i, key in enumerate(flattened_images)}
# key_to_name = {i: key for i, key in enumerate(flattened_images)}

# X, y = convert_to_X_y(flattened_images, name_to_key)
# print(X.shape, y.shape)

# q2_1()
# q2_2()
# q2_3acd()
# q2_3b()
# q2_4()
# q2_5()
# q2_6()