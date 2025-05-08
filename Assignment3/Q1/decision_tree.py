from dtree import *
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from itertools import product
import sys


CONTINOUS = 0
CATEGORICAL = 1

columns = {
    'age': CONTINOUS,
    'workclass': CATEGORICAL,
    'fnlwgt': CONTINOUS,
    'education': CATEGORICAL,
    'education.num': CONTINOUS,
    'marital.status': CATEGORICAL,
    'occupation': CATEGORICAL,
    'relationship': CATEGORICAL,
    'race' : CATEGORICAL,
    'sex': CATEGORICAL,
    'capital.gain': CONTINOUS,
    'capital.loss': CONTINOUS,
    'hours.per.week': CONTINOUS,
    'native.country': CATEGORICAL, 
}

question_part_depth = {
    'a': 20,
    'b': 55,
    'c': 55,
    'd': 10
}

def encode_labels(y):
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        return np.vectorize(label_map.get)(y).astype(int)



def q1_data_old():
    categorical_indices = [i for i, col in enumerate(columns) if columns[col] == CATEGORICAL]

    def label_encode(data, indices):
        data = data.astype(str)  # Ensure all values are treated as strings before encoding
        unique_values_map = {i: {val: idx for idx, val in enumerate(np.unique(data[:, i]))} for i in indices}
        for i in indices:
            data[:, i] = np.vectorize(unique_values_map[i].get)(data[:, i])
        return data.astype(int)

    
    X_train_final = label_encode(X_train_raw.copy(), categorical_indices)
    X_val_final = label_encode(X_val_raw.copy(), categorical_indices)
    X_test_final = label_encode(X_test_raw.copy(), categorical_indices)
    return X_train_final, y_train_raw, X_val_final, y_val_raw, X_test_final, y_test_raw
    
def q2_data_old():
     # Identify indices for categorical features
    categorical_indices = [i for i, col in enumerate(columns) if columns[col] == CATEGORICAL]
    def one_hot_encode(data, indices, unique_values_dict=None):
        data = data.astype(str)
        if unique_values_dict is None:
            unique_values_dict = {i: np.unique(data[:, i]) for i in indices}
        encoded_columns = []
        for i in indices:
            unique_values = unique_values_dict[i]
            one_hot_vectors = np.array([np.eye(len(unique_values))[np.where(unique_values == val)[0][0]] for val in data[:, i]])
            encoded_columns.append(one_hot_vectors)
        
        non_categorical_data = np.delete(data, indices, axis=1)
        return np.hstack([non_categorical_data.astype(float)] + encoded_columns), unique_values_dict
    
    # Apply one-hot encoding to categorical features in X
    X_train_final, unique_values_dict = one_hot_encode(X_train_raw.copy(), categorical_indices)
    X_val_final, _ = one_hot_encode(X_val_raw.copy(), categorical_indices, unique_values_dict)
    X_test_final, _ = one_hot_encode(X_test_raw.copy(), categorical_indices, unique_values_dict)
    
    return X_train_final, y_train_raw, X_val_final, y_val_raw, X_test_final, y_test_raw


def q1_data():
    categorical_indices = [i for i, col in enumerate(columns) if columns[col] == CATEGORICAL]

    def label_encode(data, indices):
        data = data.astype(str)  # Ensure all values are treated as strings before encoding
        unique_values_map = {i: {val: idx for idx, val in enumerate(np.unique(data[:, i]))} for i in indices}
        for i in indices:
            data[:, i] = np.vectorize(unique_values_map[i].get)(data[:, i])
        return data.astype(int)

    
    X_train_final = label_encode(X_train_raw.copy(), categorical_indices)
    X_val_final = label_encode(X_val_raw.copy(), categorical_indices)
    X_test_final = label_encode(X_test_raw.copy(), categorical_indices)
    return X_train_final, y_train_raw, X_val_final, y_val_raw, X_test_final
    
def q2_data():
     # Identify indices for categorical features
    categorical_indices = [i for i, col in enumerate(columns) if columns[col] == CATEGORICAL]
    def one_hot_encode(data, indices, unique_values_dict=None):
        data = data.astype(str)
        if unique_values_dict is None:
            unique_values_dict = {i: np.unique(data[:, i]) for i in indices}
        encoded_columns = []
        for i in indices:
            unique_values = unique_values_dict[i]
            one_hot_vectors = np.array([np.eye(len(unique_values))[np.where(unique_values == val)[0][0]] for val in data[:, i]])
            encoded_columns.append(one_hot_vectors)
        
        non_categorical_data = np.delete(data, indices, axis=1)
        return np.hstack([non_categorical_data.astype(float)] + encoded_columns), unique_values_dict
    
    # Apply one-hot encoding to categorical features in X
    X_train_final, unique_values_dict = one_hot_encode(X_train_raw.copy(), categorical_indices)
    X_val_final, _ = one_hot_encode(X_val_raw.copy(), categorical_indices, unique_values_dict)
    X_test_final, _ = one_hot_encode(X_test_raw.copy(), categorical_indices, unique_values_dict)
    
    return X_train_final, y_train_raw, X_val_final, y_val_raw, X_test_final

def q3_data():
    return q2_data()

def q4_data():
    return q2_data()

def q5_data():
    return q2_data()



def print_tree(node):
    print(node)
    if not node.is_leaf:    
        for child in node.children:
            print_tree(child)


def q1_a_old():
    D = [5, 10, 15, 20]
    # D = [5, 15]
    print(D)
    X_train, y_train, X_val, y_val, X_test, y_test = q1_data_old()
    result = {}
    for d in D:
        tree = DecisionTree()
        tree.fit(X_train, y_train, max_depth=d)
        # tree.prune(X_val, y_val)
        y_pred = tree.predict(X_test)
        test_accuracy = np.sum(y_pred == y_test) / len(y_test)
        y_pred = tree.predict(X_train)
        train_accuracy = np.sum(y_pred == y_train) / len(y_train)
        print(f"Done for {d}")
        result[d] = (train_accuracy, test_accuracy)
    for d, (train_accuracy, test_accuracy) in result.items():
        print(d, train_accuracy, test_accuracy)

def q1_b_old():
    # D = [5, 25, 35, 45, 55]
    # D = [5, 10, 15, 20]
    # D = [25, 35]
    D = [25, 35, 45, 55]
    result = {}
    X_train, y_train, X_val, y_val, X_test, y_test = q2_data_old()
    print(D)
    print("d, train_accuracy, test_accuracy")
    for d in D:
        tree = DecisionTree()
        tree.fit(X_train, y_train, max_depth=d)
        y_pred = tree.predict(X_train)
        train_accuracy = np.sum(y_pred == y_train) / len(y_train)
        y_pred = tree.predict(X_test)
        test_accuracy = np.sum(y_pred == y_test) / len(y_test)
        print(f"Done for {d}")
        result[d] = (train_accuracy, test_accuracy)

    for d, (train_accuracy, test_accuracy) in result.items():
        print(f"{d}, {train_accuracy}, {test_accuracy}")

def q1_c_old():
    D = [55]
    # D = [5]
    X_train, y_train, X_val, y_val, X_test, y_test = q2_data_old()
    for d in D:
        tree = DecisionTree()
        tree.fit(X_train, y_train, max_depth=d)
        tree.prune(X_val, y_val)

        y_pred = tree.predict(X_val)
        val_accuracy = np.sum(y_pred == y_val) / len(y_val)
        print(f"Validation Accuracy for Depth {d} : {val_accuracy * 100:.2f}%")
        # y_pred = tree.predict(X_test)
        # accuracy = np.sum(y_pred == y_test) / len(y_test)
        # print(f"Accuracy for Depth {d} : {accuracy * 100:.2f}%")

def q1_d1_old():
    X_train, y_train, X_val, y_val, X_test, y_test = q2_data_old()
    # D = [5, 10, 15, 20]
    # D = [i*5 for i in range(1, 13)]
    D = [25, 35, 45, 55]
    best_model = None
    best_accuracy = 0
    result = {}
    for d in D:
        clf = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=42)
        clf.fit(X_train, y_train)
        # print(f"Training done for {d}")
        y_pred = clf.predict(X_train)
        accuracy_on_train = np.sum(y_pred == y_train) / len(y_train)
        y_pred = clf.predict(X_test)
        # accuracy_on_test = np.sum(y_pred == y_test) / len(y_test)
        y_pred = clf.predict(X_val)
        accuracy_on_val = np.sum(y_pred == y_val) / len(y_val)
        if accuracy_on_val > best_accuracy:
            best_accuracy = accuracy_on_val
            best_model = clf
        result[d] = (accuracy_on_train, accuracy_on_val)

    print("d, acc_train, acc_val")
    for d, (acc_train, acc_val) in result.items():
        print(f"{d}, {acc_train}, {acc_val}")
    

    print(f"Best Depth: {best_model.get_depth()}")
    print(f"Best Number of Nodes: {best_model.tree_.node_count}")
    y_pred = best_model.predict(X_test)
    accuracy_on_test = np.sum(y_pred == y_test) / len(y_test)
    print(f"Accuracy on test set: {accuracy_on_test * 100:.2f}%")
    

        
def q1_d2_old():
    X_train, y_train, X_val, y_val, X_test, y_test = q2_data_old()
    # D = [5, 10, 15, 20]
    final_model = None
    best_accuracy = 0
    result = {}
    ccp_alpha_values = [0.0, 0.001, 0.01, 0.1, 0.2]
    for ccp_alpha in ccp_alpha_values:
        clf = DecisionTreeClassifier(
                                    criterion='entropy',
                                    random_state=42,
                                    max_depth=None,
                                    ccp_alpha=0.01,)
        clf.fit(X_train, y_train)
        # print(f"Training done for {d}")
        y_pred = clf.predict(X_train)
        accuracy_on_train = np.sum(y_pred == y_train) / len(y_train)
        print(f"{ccp_alpha}, {accuracy_on_train}")
        y_pred = clf.predict(X_test)
        accuracy_on_test = np.sum(y_pred == y_test) / len(y_test)
        y_pred = clf.predict(X_val)
        accuracy_on_val = np.sum(y_pred == y_val) / len(y_val)
        result[ccp_alpha] = (accuracy_on_val, clf.tree_.node_count, clf.tree_.max_depth)
        if accuracy_on_val > best_accuracy:
            best_accuracy = accuracy_on_val
            final_model = clf
    
    print("ccp_alpha, acc_val, nodes, height")
    for ccp_alpha, (acc_val, nodes, height) in result.items():
        print(f"{ccp_alpha}, {acc_val}, {nodes}, {height}")
    print(f"Best ccp_alpha: {final_model.ccp_alpha}")
    print(f"Best Depth: {final_model.get_depth()}")
    print(f"Best Number of Nodes: {final_model.tree_.node_count}")
    y_pred = final_model.predict(X_test)
    accuracy_on_test = np.sum(y_pred == y_test) / len(y_test)
    print(f"Accuracy on test set: {accuracy_on_test * 100:.2f}%")



def q1_e_old():
    X_train, y_train, X_val, y_val, X_test, y_test = q2_data_old()
    
    n_estimators = [50, 150, 250, 350]
    max_features = [0.1, 0.3, 0.5, 0.7, 1.0]
    min_samples_split = [2, 4, 6, 8, 10]
   
    best_oob_score = 0.0
    best_params = None
    best_model = None
    result = {}

    for (n, m, s) in product(n_estimators, max_features, min_samples_split):
        rf = RandomForestClassifier(
            n_estimators=n,
            max_features=m,
            min_samples_split=s,
            criterion='entropy',
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        oob_score = rf.oob_score_
        y_pred_val = rf.predict(X_val)
        accuracy_val = np.mean(y_pred_val == y_val)

        if oob_score > best_oob_score:
            best_oob_score = oob_score
            best_params = (n, m, s)
            best_model = rf
        print(f"Training done for n_estimators={n}, max_features={m}, min_samples_split={s}")
        result[(n, m, s)] = (accuracy_val, oob_score)
    
    print("n_estimators, max_features, min_samples_split, acc_val, oob_score")
    for (n, m, s), (acc_val, oob_score) in result.items():
        print(f"{n}, {m}, {s}, {acc_val:.2f}, {oob_score:.2f}")
    print(f"Best Parameters: {best_params}")
    print(f"Best OOB Score: {best_oob_score:.2f}")
    y_pred = best_model.predict(X_test)
    accuracy_on_test = np.mean(y_pred == y_test)
    print(f"Accuracy on test set: {accuracy_on_test * 100:.2f}%")


def q1_a():
    X_train, y_train, _, _ , X_test = q1_data()
    d = question_part_depth[question_part]
    
    tree = DecisionTree()
    tree.fit(X_train, y_train, max_depth=d)
    y_pred = tree.predict(X_test)
    return y_pred

def q1_b():
    X_train, y_train, _, _ , X_test = q2_data()
    d = question_part_depth[question_part]
    
    tree = DecisionTree()
    tree.fit(X_train, y_train, max_depth=d)
    y_pred = tree.predict(X_test)
    return y_pred
    


def q1_c():
    X_train, y_train, X_val, y_val, X_test = q3_data()
    d = question_part_depth[question_part]

    tree = DecisionTree()
    tree.fit(X_train, y_train, max_depth=d)
    tree.prune(X_val, y_val)
    y_pred = tree.predict(X_test)
    return y_pred

def q1_d():
    X_train, y_train, X_val, y_val, X_test = q4_data()
    d = question_part_depth[question_part]
    y_train = np.concatenate((y_train, y_val))
    X_train = np.concatenate((X_train, X_val))
    clf = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred
    
    




def q1_e():
    X_train, y_train, X_val, y_val, X_test = q5_data()
    
    n_estimators = [50, 150, 250, 350]
    max_features = [0.1, 0.3, 0.5, 0.7, 1.0]
    min_samples_split = [2, 4, 6, 8, 10]
    n, m, s= 350, 0.3, 10
    rf = RandomForestClassifier(
        n_estimators=n,
        max_features=m,
        min_samples_split=s,
        criterion='entropy',
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    return y_pred
    




if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python decision_tree.py <train_data_path> <validation_data_path> <test_data_path> <output_folder_path> <question_part>")
        sys.exit(1)
    
    train_data_path = str(sys.argv[1])
    val_data_path = str(sys.argv[2])
    test_data_path = str(sys.argv[3])
    output_folder_path = str(sys.argv[4])
    question_part = str(sys.argv[5])
    if question_part not in ['a', 'b', 'c', 'd', 'e']:
        print("Invalid question part. Please provide a number between 1 and 5.")
        sys.exit(1)

    raw_train_data = pd.read_csv(train_data_path, header=0).values
    raw_val_data = pd.read_csv(val_data_path, header=0).values
    raw_test_data = pd.read_csv(test_data_path, header=0).values

    X_train_raw = raw_train_data[:, :-1]
    y_train_raw = encode_labels(raw_train_data[:, -1])
    X_val_raw = raw_val_data[:, :-1]
    y_val_raw = encode_labels(raw_val_data[:, -1])

    # X_test_raw = raw_test_data[:, :-1]
    # y_test_raw = encode_labels(raw_test_data[:, -1])
    
    X_test_raw = raw_test_data

    y_pred = None
    if question_part == 'a':
       y_pred = q1_a()
    elif question_part == 'b':
        y_pred = q1_b()
    elif question_part == 'c':
        y_pred = q1_c()
    elif question_part == 'd':
        y_pred = q1_d()
    elif question_part == 'e':
        y_pred = q1_e()


    if y_pred is not None:
        df = pd.DataFrame(y_pred, columns=['prediction'])
        df['prediction'] = df['prediction'].map({0: "<=50K", 1: ">50K"})
        output_file_path = f"{output_folder_path}/prediction_{question_part}.csv"
        df.to_csv(output_file_path, index=False)
        print(f"Output saved to {output_file_path}")
