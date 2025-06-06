import cv2
import numpy as np
import os
from svm import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,  GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
import time
from sklearn.linear_model import SGDClassifier 
import pandas as pd
import seaborn as sns
import shutil

ENTRYNUMBER = 11596
TEST_DIRECTORY_PATH = "../data/Q2/test/"
TRAIN_DIRECTORY_PATH = "../data/Q2/train/"
TOTAL_CLASSES = 11
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

def get_folder_images(path, lst, valid_extensions=(".jpg", ".png", ".jpeg", ".bmp"), names = False):
    cv2.setLogLevel(0)
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folders.sort()
    folder_images = {}
    folder_filenames = {}
    for n in lst:
        try:
            folder_index = n % len(folders)
            folder_name = folders[folder_index]
            folder_path = os.path.join(path, folder_name)
            images = []
            filenames = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(valid_extensions):  # Check for valid image extensions
                    img_path = os.path.join(folder_path, file)
                    img = cv2.imread(img_path)  # Read image
                    if img is not None:
                        images.append(img)
                        filenames.append(file)
            
            folder_images[folder_name] = images
            folder_filenames[folder_name] = filenames
        
        except IndexError:
            raise IndexError(f"Index {n} is out of bounds. There are only {len(folders)} folders.")
    
    if names:
        return folder_images, folder_filenames
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
    dict = get_folder_images(TRAIN_DIRECTORY_PATH, lst)
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    X, y = convert_to_X_y(flattened_images, name_to_key)
    # print(X.shape, y.shape)
    model = SupportVectorMachine()
    model.fit(X, y, autograder=False)

    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)    
    print(accuracy)
    return model

def q2_1():
    start_time = time.time()
    lst = [ENTRYNUMBER, ENTRYNUMBER + 1]
    dict_train = get_folder_images(TRAIN_DIRECTORY_PATH, lst)
    dict_test = get_folder_images(TEST_DIRECTORY_PATH, lst)
    flattened_images_train = {key: preprocess_images(dict_train[key]) for key in dict_train}
    flattened_images_test = {key: preprocess_images(dict_test[key]) for key in dict_test}
    name_to_key = {key: i for i, key in enumerate(flattened_images_train)}
    key_to_name = {i: key for i, key in enumerate(flattened_images_train)}
    print(name_to_key.keys())

    X_train, y_train = convert_to_X_y(flattened_images_train, name_to_key)
    X_test, y_test = convert_to_X_y(flattened_images_test, name_to_key)
    # Train the model
    model = SupportVectorMachine()
    model.fit(X_train, y_train, autograder=False)  # Assuming X_train and y_train are defined
    # Get the number of support vectors
    num_support_vectors = len(model.support_vectors)
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)  
    # Get the percentage of training samples that are support vectors
    total_samples = len(y_train)
    percentage_support_vectors = (num_support_vectors / total_samples) * 100
    end_time = time.time()
    weight_norm = np.linalg.norm(model.w)
    recall = np.sum((predictions == 1) & (y_test == 1)) / np.sum(y_test == 1)
    precision = np.sum((predictions == 1) & (y_test == 1)) / np.sum(predictions == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print("--"*20)
    print(f"Number of support vectors: {num_support_vectors}")
    print(f"Percentage of training samples that are support vectors: {percentage_support_vectors:.2f}%")
    print(f"Training accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Weights: {model.w}")
    print(f"Weight norm: {weight_norm}")
    print(f"Bias: {model.b}")
    print("--"*20)
    
    top_5_indices = np.argsort(-model.alphas)[:5]  # Get indices of top-5 largest alphas
    top_5_support_vectors = model.support_vectors[top_5_indices]  # Extract corresponding support vectors

    top_5_images = [sv.reshape(100, 100, 3) for sv in top_5_support_vectors]
    plot_images(top_5_images, [f"Support Vector q2_1 {i+1}" for i in range(5)], q_num="q2_1")

    weight_vector_image = model.w.reshape(100, 100, 3)

    weight_min, weight_max = weight_vector_image.min(), weight_vector_image.max()
    weight_vector_scaled = (weight_vector_image - weight_min) / (weight_max - weight_min) * 255
    weight_vector_scaled = weight_vector_scaled.astype(np.uint8)  # Convert to 8-bit image

    # Plot the scaled weight vector
    plt.figure(figsize=(5, 5))
    plt.imshow(weight_vector_scaled, cmap="gray")
    plt.axis("off")
    plt.title("Weight Vector Visualization (Scaled 0-255)")

    del model

def q2_2():
    start_time = time.time()
    lst = [ENTRYNUMBER, ENTRYNUMBER + 1]
    dict_train = get_folder_images(TRAIN_DIRECTORY_PATH, lst)
    dict_test = get_folder_images(TEST_DIRECTORY_PATH, lst)
    flattened_images_train = {key: preprocess_images(dict_train[key]) for key in dict_train}
    flattened_images_test = {key: preprocess_images(dict_test[key]) for key in dict_test}
    name_to_key = {key: i for i, key in enumerate(flattened_images_train)}
    key_to_name = {i: key for i, key in enumerate(flattened_images_train)}

    X_train, y_train = convert_to_X_y(flattened_images_train, name_to_key)
    X_test, y_test = convert_to_X_y(flattened_images_test, name_to_key)
    # Train the model
    model = SupportVectorMachine()
    model.fit(X_train, y_train, kernel = 'gaussian', C = 1.0, gamma=0.001, autograder=False)  # Assuming X_train and y_train are defined
    # print(y[0:10])
    # Get the number of support vectors
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)  
    total_samples = len(y_train)
    num_support_vectors = len(model.support_vectors)
    percentage_support_vectors = (num_support_vectors / total_samples) * 100
    recall = np.sum((predictions == 1) & (y_test == 1)) / np.sum(y_test == 1)
    precision = np.sum((predictions == 1) & (y_test == 1)) / np.sum(predictions == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    end_time = time.time()
    print("--"*20)
    print(f"Number of support vectors: {num_support_vectors}")
    print(f"Percentage of training samples that are support vectors: {percentage_support_vectors:.2f}%")
    print(f"Training accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Bias: {model.b}")
    print("--"*20)
    top_5_indices = np.argsort(-model.alphas)[:5]  # Get indices of top-5 largest alphas
    top_5_support_vectors = model.support_vectors[top_5_indices]  # Extract corresponding support vectors

    top_5_images = [sv.reshape(100, 100, 3) for sv in top_5_support_vectors]
    plot_images(top_5_images, [f"Support Vector q2_2 {i+1}" for i in range(5)], q_num = "q2_2")
    del model

def q2_3acd():
    lst = [ENTRYNUMBER, ENTRYNUMBER + 1]

    start_time = time.time()
    dict_train = get_folder_images(TRAIN_DIRECTORY_PATH, lst)
    dict_test = get_folder_images(TEST_DIRECTORY_PATH, lst)
    flattened_images_train = {key: preprocess_images(dict_train[key]) for key in dict_train}
    flattened_images_test = {key: preprocess_images(dict_test[key]) for key in dict_test}
    name_to_key = {key: i for i, key in enumerate(flattened_images_train)}
    key_to_name = {i: key for i, key in enumerate(flattened_images_train)}

    X_train, y_train = convert_to_X_y(flattened_images_train, name_to_key)
    X_test, y_test = convert_to_X_y(flattened_images_test, name_to_key)
    print(name_to_key.keys())
    model = SVC(kernel='linear', C = 1.0)   
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    num_support_vectors = sum(model.n_support_)
    percentage_support_vectors = (np.sum(num_support_vectors) / len(y_train)) * 100
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
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    num_support_vectors = sum(model.n_support_)
    support_vectors = model.support_vectors_
    percentage_support_vectors = (np.sum(num_support_vectors) / len(y_train)) * 100
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

    dict_train = get_folder_images(TRAIN_DIRECTORY_PATH, lst)
    dict_test = get_folder_images(TEST_DIRECTORY_PATH, lst)
    flattened_images_train = {key: preprocess_images(dict_train[key]) for key in dict_train}
    flattened_images_test = {key: preprocess_images(dict_test[key]) for key in dict_test}
    name_to_key = {key: i for i, key in enumerate(flattened_images_train)}
    key_to_name = {i: key for i, key in enumerate(flattened_images_train)}

    X_train, y_train = convert_to_X_y(flattened_images_train, name_to_key)
    X_test, y_test = convert_to_X_y(flattened_images_test, name_to_key)

    model = SVC(kernel='linear', C = 1.0)   
    model.fit(X_train, y_train)

    weights_1 = model.coef_
    bias_1 = model.intercept_
    # print(weights_1, bias_1)
    del model

    model = SupportVectorMachine()
    model.fit(X_test, y_test, autograder=False)  # Assuming X_train and y_train are defined
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
    dict_train = get_folder_images(TRAIN_DIRECTORY_PATH, lst)
    dict_test = get_folder_images(TEST_DIRECTORY_PATH, lst)
    flattened_images_train = {key: preprocess_images(dict_train[key]) for key in dict_train}
    flattened_images_test = {key: preprocess_images(dict_test[key]) for key in dict_test}
    name_to_key = {key: i for i, key in enumerate(flattened_images_train)}
    key_to_name = {i: key for i, key in enumerate(flattened_images_train)}

    X_train, y_train = convert_to_X_y(flattened_images_train, name_to_key)
    X_test, y_test = convert_to_X_y(flattened_images_test, name_to_key)
    
    libsvm = SVC(kernel='linear')

    start_time = time.time()
    libsvm.fit(X_train, y_train)
    libsvm_train_time = time.time() - start_time
    libsvm_acc = accuracy_score(y_test, libsvm.predict(X_test))
    sgd_svm = SGDClassifier(loss='hinge', learning_rate='optimal', max_iter=1000, random_state=42)

    start_time = time.time()
    sgd_svm.fit(X_train, y_train)
    sgd_train_time = time.time() - start_time

    sgd_acc = accuracy_score(y_test, sgd_svm.predict(X_test))

    print(f"LIBLINEAR: Accuracy = {libsvm_acc:.4f}, Training Time = {libsvm_train_time:.4f} sec")
    print(f"SGD SVM: Accuracy = {sgd_acc:.4f}, Training Time = {sgd_train_time:.4f} sec")

def q2_5_7():
    model = {}
    lst_total = [i for i in range(TOTAL_CLASSES)]
    dict_total = get_folder_images(TRAIN_DIRECTORY_PATH, lst_total)
    flattened_images_total = {key: preprocess_images(dict_total[key]) for key in dict_total}
    for i in range(TOTAL_CLASSES):
        for j in range(i+1, TOTAL_CLASSES):
            lst = [i , j]
            # dict = get_folder_images(TRAIN_DIRECTORY_PATH, lst)
            # flattened_images = {key: preprocess_images(dict[key]) for key in dict}
            flattened_images = {key: flattened_images_total[key] for key in lst}
            name_to_key = {key: i for i, key in enumerate(flattened_images)}
            key_to_name = {i: key for i, key in enumerate(flattened_images)}

            X, y = convert_to_X_y(flattened_images, name_to_key)
            model[i, j] = SupportVectorMachine()
            model[i, j].fit(X, y, kernel = 'gaussian', C = 1.0, gamma=0.001, autograder = False)
            print(f"Trained model for {i} vs {j}")
    
    lst = [i for i in range(TOTAL_CLASSES)]
    dict, dict_filenames = get_folder_images(TEST_DIRECTORY_PATH, lst, names=True)
    X, y = [], []
    filenames = []
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    for key in flattened_images:
        X.extend(flattened_images[key])
        y.extend([name_to_key[key]]*len(flattened_images[key]))
        filenames.extend(dict_filenames[key])

    X = np.array(X)
    y = np.array(y)
    predictions = {}
    for i in range(TOTAL_CLASSES):
        for j in range(i+1, TOTAL_CLASSES):
            predictions[i, j] = model[i, j].predict(X)
            print(f"Predicted for {i} vs {j}")

    del model

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
    recall = np.sum((final_predictions == 1) & (y == 1)) / np.sum(y == 1)
    precision = np.sum((final_predictions == 1) & (y == 1)) / np.sum(final_predictions == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy: {accuracy_score:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    map_labels = np.vectorize(lambda k: key_to_name[k])
    df = pd.DataFrame({
        'Filename': filenames,
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

def q2_6_7():
    model = {}
    lst_total = [i for i in range(TOTAL_CLASSES)]
    dict_total = get_folder_images(TRAIN_DIRECTORY_PATH, lst_total)
    flattened_images_total = {key: preprocess_images(dict_total[key]) for key in dict_total}
    for i in range(TOTAL_CLASSES):
        for j in range(i+1, TOTAL_CLASSES):
            lst = [i , j]
            # dict = get_folder_images(TRAIN_DIRECTORY_PATH, lst)
            # flattened_images = {key: preprocess_images(dict[key]) for key in dict}
            flattened_images = {key: flattened_images_total[key] for key in lst}
            name_to_key = {key: i for i, key in enumerate(flattened_images)}
            key_to_name = {i: key for i, key in enumerate(flattened_images)}

            X, y = convert_to_X_y(flattened_images, name_to_key)
            model[i, j] = SVC(kernel='rbf', C = 1.0, gamma=0.001)   
            model[i, j].fit(X, y)
            print(f"Trained model for {i} vs {j}")
    
    lst = [i for i in range(TOTAL_CLASSES)]
    dict, dict_filenames = get_folder_images(TEST_DIRECTORY_PATH, lst, names=True)
    X, y = [], []
    filenames = []
    flattened_images = {key: preprocess_images(dict[key]) for key in dict}
    name_to_key = {key: i for i, key in enumerate(flattened_images)}
    key_to_name = {i: key for i, key in enumerate(flattened_images)}

    for key in flattened_images:
        X.extend(flattened_images[key])
        y.extend([name_to_key[key]]*len(flattened_images[key]))
        filenames.extend(dict_filenames[key])

    X = np.array(X)
    y = np.array(y)
    predictions = {}
    for i in range(TOTAL_CLASSES):
        for j in range(i+1, TOTAL_CLASSES):
            predictions[i, j] = model[i, j].predict(X)
            print(f"Predicted for {i} vs {j}")

    del model

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
    recall = np.sum((final_predictions == 1) & (y == 1)) / np.sum(y == 1)
    precision = np.sum((final_predictions == 1) & (y == 1)) / np.sum(final_predictions == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy: {accuracy_score:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Accuracy: {accuracy_score}")
    map_labels = np.vectorize(lambda k: key_to_name[k])
    df = pd.DataFrame({
        'Filename': filenames,
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

def q2_7():
    qtype = ["q2_5", "q2_6"]
    for q in qtype:
        CSV_PATH = f"{q}_multiclass_predictions.csv"
        IMAGE_FOLDER = TEST_DIRECTORY_PATH
        OUTPUT_FOLDER = f"misclassified_samples_{q}"

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Read CSV file
        df = pd.read_csv(CSV_PATH)

        # Filter misclassified images
        misclassified_df = df[df["Misclassified"] == 1]

        # Dictionary to store one example per (Ground Truth, Prediction) pair
        selected_images = {}

        for _, row in misclassified_df.iterrows():
            true_label = row["Ground Truth"]
            predicted_label = row["Predictions"]
            filename = row["Filename"]

            key = (true_label, predicted_label)

            # Save only one image per misclassification type
            if key not in selected_images:
                selected_images[key] = filename
                src_path = os.path.join(f"{IMAGE_FOLDER}{true_label}/", filename)
                dst_path = os.path.join(OUTPUT_FOLDER, f"{q}_{true_label}_{predicted_label}_{filename}")

                # Copy image to output folder
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                    print(f"Saved misclassified image: {filename} ({true_label} → {predicted_label})")
                else:
                    print(f"Image not found: {src_path}")

def q2_8():
    C_values = [1e-5, 1e-3, 1, 5, 10]
    KFOLD = 5
    # Prepare data for training

    lst_total = [i for i in range(TOTAL_CLASSES)]
    dict_total = get_folder_images(TRAIN_DIRECTORY_PATH, lst_total)
    flattened_images_train = {key: preprocess_images(dict_total[key]) for key in dict_total}

    name_to_key = {key: i for i, key in enumerate(flattened_images_train)}
    key_to_name = {i: key for i, key in enumerate(flattened_images_train)}

    X_train = []
    y_train = []

    for key in flattened_images_train:
        X_train.extend(flattened_images_train[key])
        y_train.extend([name_to_key[key]]*len(flattened_images_train[key]))

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Perform cross-validation for each C value
    cv_accuracies = {}
    for C in C_values:
        model = SVC(kernel='rbf', C=C, gamma=0.001)
        scores = cross_val_score(model, X_train, y_train, cv=KFOLD, scoring='accuracy')
        cv_accuracies[C] = np.mean(scores)
        print(f"Cross-validation accuracy for C={C}: {cv_accuracies[C]:.4f}")

    # Select best C value
    best_C = max(cv_accuracies, key=cv_accuracies.get)
    print(f"Best C value: {best_C}")

    # Train final model using best C
    final_model = SVC(kernel='rbf', C=best_C, gamma=0.001)
    final_model.fit(X_train, y_train)

    # Load test data
    dict_test, dict_filenames = get_folder_images(TEST_DIRECTORY_PATH, lst_total, names=True)
    flattened_images_test = {key: preprocess_images(dict_test[key]) for key in dict_test}

    X_test = []
    y_test = []

    for key in flattened_images_test:
        X_test.extend(flattened_images_test[key])
        y_test.extend([name_to_key[key]]*len(flattened_images_test[key]))

    X_test, y_test = np.array(X_test), np.array(y_test)

    # Make predictions
    y_pred = final_model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    test_accuracies = [accuracy_score(y_test, SVC(kernel='rbf', C=C, gamma=0.001).fit(X_train, y_train).predict(X_test)) for C in C_values]
    
    # Print results
    print("--" * 20)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("--" * 20)

    # Print cross-validation results
    for C in C_values:
        print(f"Mean cross-validation accuracy for C={C}: {cv_accuracies[C]:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(C_values, list(cv_accuracies.values()), marker='o', label='5-Fold CV Accuracy')
    plt.plot(C_values, test_accuracies, marker='s', label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel('C values (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Cross-validation vs Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("q2_8_accuracy.png")





# q2_1()
# q2_2()
# q2_3acd()
# q2_3b()
# q2_4()
# q2_5_7()
# q2_6_7()
# q2_7()
q2_8()
