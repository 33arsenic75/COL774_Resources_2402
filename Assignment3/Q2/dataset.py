import numpy as np
import pandas as pd
import os
from PIL import Image
import numpy as np

DATA_PATH = '../data/Q2/Traffic sign board'

def one_hot_encode(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1
    return np.eye(num_classes)[y]

def load_gtsrb_train_data(root_dir, img_size=None):
    X = []
    y = []

    for class_id in os.listdir(root_dir):
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

    for _, row in df.iterrows():
        img_filename = row['image']
        img_path = os.path.join(images_dir, img_filename)
        img_label = row['label']
        try:
            img = Image.open(img_path).convert("RGB")
            if img_size:
                img = img.resize(img_size)
            img_array = np.array(img).flatten()  # shape: (2352,)
            X.append(img_array)
            y.append(int(img_label))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    X = np.array(X)
    return X, y


def get_train_data(NUM_CLASSES=43):
    # Load the training data
    X_train, y_train_raw = load_gtsrb_train_data(DATA_PATH + "/train/")
    y_train = one_hot_encode(y_train_raw, num_classes=NUM_CLASSES)
    return X_train, y_train

def get_test_data(NUM_CLASSES=43):
    csv_path = DATA_PATH + "/test_labels.csv"
    images_dir = DATA_PATH + "/test"
    X_test, y_test_raw = load_gtsrb_test_from_csv(csv_path, images_dir)
    y_test_raw = np.array(y_test_raw)
    y_test = one_hot_encode(y_test_raw, num_classes=NUM_CLASSES)
    return X_test, y_test, y_test_raw
    