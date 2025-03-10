import cv2
import numpy as np
import os
from svm import *
import matplotlib.pyplot as plt

ENTRYNUMBER = 11596
DIRECTORY_PATH = "../data/Q2/dataset/"

def preprocess_images(images):
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
        # ax.imshow(images[i], cmap=cmap)
        ax.set_title(titles[i])
        ax.axis('off')
    plt.savefig(f"support_vectors_{q_num}.png")

def get_folder_images(path, lst, valid_extensions=(".jpg", ".png", ".jpeg", ".bmp")):
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

def convert_to_X_y(dict, name_to_key):
    X = []
    y = []
    keys = list(name_to_key.keys())  # Extract the two keys
    key1, key2 = keys[0], keys[1]  # Assign them to variables

    name_to_key_2 = {
        key1: 1 if name_to_key[key1] > name_to_key[key2] else -1,
        key2: 1 if name_to_key[key2] > name_to_key[key1] else -1
    }
    for key in dict:
        X.extend(dict[key])
        y.extend([name_to_key_2[key]]*len(dict[key]))
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
    print("--"*20)
    print(f"Number of support vectors: {num_support_vectors}")
    print(f"Percentage of training samples that are support vectors: {percentage_support_vectors:.2f}%")
    print(f"Training accuracy: {accuracy:.2f}")
    print(f"Weights: {model.w}")
    print(f"Bias: {model.b}")
    print("--"*20)
    
    top_5_indices = np.argsort(-model.alphas)[:5]  # Get indices of top-5 largest alphas
    top_5_support_vectors = model.support_vectors[top_5_indices]  # Extract corresponding support vectors

    top_5_images = [sv.reshape(100, 100, 3) for sv in top_5_support_vectors]
    plot_images(top_5_images, [f"Support Vector q2_1 {i+1}" for i in range(5)], q_num="q2_1")


def q2_2():
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
    num_support_vectors = len(model.support_vectors)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)  
    # Get the percentage of training samples that are support vectors
    total_samples = len(y)
    percentage_support_vectors = (num_support_vectors / total_samples) * 100
    print("--"*20)
    print(f"Number of support vectors: {num_support_vectors}")
    print(f"Percentage of training samples that are support vectors: {percentage_support_vectors:.2f}%")
    print(f"Training accuracy: {accuracy:.2f}")
    print(f"Weights: {model.w}")
    print(f"Bias: {model.b}")
    print("--"*20)
    
    top_5_indices = np.argsort(-model.alphas)[:5]  # Get indices of top-5 largest alphas
    top_5_support_vectors = model.support_vectors[top_5_indices]  # Extract corresponding support vectors

    top_5_images = [sv.reshape(100, 100, 3) for sv in top_5_support_vectors]
    plot_images(top_5_images, [f"Support Vector q2_2 {i+1}" for i in range(5)], q_num = "q2_2")

# dict = get_folder_images(DIRECTORY_PATH, [i for i in range(11)] )
# flattened_images = {key: preprocess_images(dict[key]) for key in dict}
# name_to_key = {key: i for i, key in enumerate(flattened_images)}
# key_to_name = {i: key for i, key in enumerate(flattened_images)}

# X, y = convert_to_X_y(flattened_images, name_to_key)
# print(X.shape, y.shape)

q2_1()
# q2_2()