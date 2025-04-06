import pandas as pd
import numpy as np
import copy


next_ = 1

class Node:
    def __init__(self):
        self.name = None
        self.parent: Node = None
        self.children: list = []

        self.split_attribute: str = None
        self.split_value: float = None

        self.is_leaf: bool = False
        self.class_label: str = None

        self.entropy: float = 1
        self.val_accuracy: float = 0

        self.depth: int = 0
        self.child_to_choose: function = None

        self.tree_size: int = 0
        self.tree_height: int = 0
        self.max_depth: int = int(1e10)

        self.tree_entropy = 0
        self.val_tree_accuracy: float = 0
        self.child_probabilities : list = []
        self.val_child_probabilities : list = []

        self.fraction_of_child: float = 0
        self.val_fraction_of_total: float = 0


    def __repr__(self):
        if self.is_leaf:
            return (f"[*] Leaf Node ({self.name})\n"
                    f"    Depth: {self.depth}\n"
                    f"    Label: {self.class_label}\n"
                    f"    Entropy: {self.entropy:.2f}\n"
                    f"    Fraction of child: {self.fraction_of_child:.2f}\n"
                    f"    Validation Entropy: {self.val_entropy:.2f}\n"
                    f"    Validation Tree Entropy: {self.val_tree_entropy:.2f}\n")
        else:
            return (f"[*] Node ({self.name})\n"
                    f"    Depth: {self.depth}\n"
                    f"    Split: {self.split_attribute}\n"
                    f"    Entropy: {self.entropy:.2f}\n"
                    f"    Fraction of child: {self.fraction_of_child:.2f}\n"
                    f"    Validation Entropy: {self.val_entropy:.2f}\n"
                    f"    Validation Tree Entropy: {self.val_tree_entropy:.2f}\n")

    def has_children(self):
        return len(self.children) > 0
    
    def majority_class(self, y):
        if self.class_label is None:
            self.class_label = pd.Series(y).mode()[0]
        return self.class_label
        
    def initialize(self, X, y, parent = None, name = " ", depth = 0, fraction_of_child = 1, max_depth = int(1e10)):
        self.parent = parent
        self.name = name

        self.fraction_of_child = fraction_of_child

        self.depth = depth
        self.max_depth = max_depth
        self.tree_size = 0
        self.tree_height = 0
        self.majority_class(y)
        self.entropy = max(0, self.calculate_entropy(X, y))

        if self.max_depth == self.depth:
            self.is_leaf = True
            self.class_label = self.majority_class(y)
            return
        
        self.split(X, y)
        self.whole_tree_entropy()
        

    def whole_tree_entropy(self):
        if self.tree_entropy is not None:
            return self.tree_entropy
        
        elif self.is_leaf:
            self.tree_entropy = self.entropy

        else:
            self.tree_entropy = sum(self.child_probabilities[i] * self.children[i].whole_tree_entropy() for i in range(len(self.children)))
        
        return self.tree_accuracy

    def validation(self, X, y, total_samples = np.inf):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        self.val_accuracy = accuracy
        self.val_fraction_of_total = len(X)/total_samples
        # self.val_child_probabilities = np.zeros(len(self.children))
        if self.is_leaf:
            self.val_tree_accuracy = self.val_accuracy
            self.val_child_probabilities = []
            return
        
        child_indices = np.array([self.child_to_choose(val) for val in X[:, self.split_attribute]])
        self.val_child_probabilities = np.zeros(len(self.children))
        for i, child in enumerate(self.children):
            mask = (child_indices == i)  # Boolean mask for selecting relevant rows
            self.val_child_probabilities[i] = len(X[mask]) / len(X)
            if np.any(mask):  # Process only if there are samples for this child
                child.validation(X[mask], y[mask], total_samples)
        
        self.val_tree_accuracy = sum(self.val_child_probabilities[i] * self.children[i].val_tree_accuracy for i in range(len(self.children)))
        

    def get_tree_size(self):
        if self.is_leaf:
            self.tree_size = 1
        else:
            self.tree_size = 1 + sum(x.get_tree_size() for x in self.children)
        
        return self.tree_size
    
    def get_tree_height(self):
        if self.is_leaf:
            self.tree_height = 0
        else:
            self.tree_height = 1 + max(x.get_tree_height() for x in self.children)
        
        return self.tree_height

    def calculate_entropy(self, X, y):
        # If y is empty, return entropy as 0
        if len(y) == 0:
            return 0
        
        # Convert to pandas Series if needed
        p = pd.Series(y).value_counts(normalize=True)

        # Compute entropy
        entropy = -np.sum(p * np.log2(p.clip(1e-10, 1)))  # Clip values to avoid log(0)
        return entropy

 
    def calculate_information_gain(self, X, y, attribute):
        
        unique_values = np.unique(X[:, attribute])
        child_to_choose = None
        value = None
        if (X[:, attribute].dtype == np.float64 or X[:, attribute].dtype == np.int64) and len(unique_values) > 2:
            value = np.median(unique_values)
            mask = X[:, attribute] <= value
            child_data_X = [X[mask], X[~mask]]
            child_data_y = [y[mask], y[~mask]]
            child_to_choose = lambda x: 0 if x <= value else 1
        else:
            unique_values = np.unique(X[:, attribute])
            child_data_X = [X[X[:, attribute] == val] for val in unique_values]
            child_data_y = [y[X[:, attribute] == val] for val in unique_values]
            child_to_choose = lambda x: next((i for i, val in enumerate(unique_values) if (val == x)), None)


        if self.entropy is None:
            self.entropy = self.calculate_entropy(X, y)
        
        parent_entropy = self.entropy
        total_samples = len(y)
        child_entropy = sum(
            (len(child_data_y[i]) / total_samples) * self.calculate_entropy(child_data_X[i], child_data_y[i])
            for i in range(len(child_data_X))
        )
        # Calculate the information gain
        information_gain = parent_entropy - child_entropy
        return child_data_X, child_data_y, information_gain, child_to_choose, value
    
    def split(self, X, y):
        global next_

        best_info_gain = -1
        best_split_attribute = None
        best_split_value = None
        best_child_data_X = None
        best_child_data_y = None
        best_child_to_choose = None
    
        for attribute in range(X.shape[1]):
            child_data_X, child_data_y, info_gain, child_to_choose, split_value  = self.calculate_information_gain(X, y, attribute)
            # print(f"Attribute: {attribute}, Info Gain: {info_gain}")
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_attribute = attribute
                best_split_value = split_value
                best_child_data_X = child_data_X
                best_child_data_y = child_data_y
                best_child_to_choose = child_to_choose
        
        # print(f"Best Info Gain: {best_info_gain}")
        if best_info_gain == 0:
            self.is_leaf = True
            self.class_label = self.majority_class(y)
            return
        
        self.split_attribute = best_split_attribute
        self.split_value = best_split_value
        self.child_to_choose = best_child_to_choose

        self.children = [Node() for _ in range(len(best_child_data_X))]
        self.child_probabilities = [len(best_child_data_y[i]) / len(y) for i in range(len(best_child_data_y))]
        self.is_leaf = False
        old = next_
        next_ += len(self.children)
        for i in range(len(self.children)):
            self.children[i].initialize(best_child_data_X[i], 
                                        best_child_data_y[i], 
                                        self, name = old + i + 1, 
                                        depth = self.depth + 1, 
                                        fraction_of_child = self.child_probabilities[i],
                                        max_depth = self.max_depth)

    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def predict_one(self, x):
        if self.is_leaf:
            return self.class_label
        else:
            child_index = self.child_to_choose(x[self.split_attribute])
            if child_index is not None and 0 <= child_index < len(self.children):
                return self.children[child_index].predict_one(x)  # Fixed recursive call
            else:
                # Handle cases where the value was not seen during training
                return self.class_label
                


        
        
class DecisionTree:
    def __init__(self):
        self.root = Node()
        self.height = None
        self.num_nodes = None
        self.is_pruned = False
        self.all_nodes = []
        self.max_depth = None

    def __repr__(self):
        return f"{self._repr_tree(self.root, level=0)}"

    def _repr_tree(self, node, level):
        """ Recursively represent the tree structure with indentation. """
        indent = "    " * level  # Indentation for hierarchy
        node_repr = "\n".join(indent + line for line in str(node).split("\n"))  # Indent every line
        
        if not node.is_leaf:  # Recursively process children
            for child in node.children:
                node_repr += "\n" + self._repr_tree(child, level + 1)
        
        return node_repr
    
    def fit(self, X, y, max_depth = int(1e10)):
        self.max_depth = max_depth
        self.root.initialize(X, y, None, name = 1, depth = 0, max_depth = self.max_depth)
    
    def predict(self, X):
        return self.root.predict(X)
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        if len(y_pred) == 0:
            return 0
        # Calculate accuracy
        return np.sum(y_pred == y) / len(y)

    def get_tree_size(self):
        if self.is_pruned:
            self.is_pruned = False
            self.num_nodes = None
            self.height = None
            self.get_tree_size()
            self.get_tree_height()
        if self.num_nodes is None:
            self.num_nodes = self.root.get_tree_size()
        return self.num_nodes

    def get_tree_height(self):
        if self.height is None:
            self.height = self.root.get_tree_height()
        return self.height

    def prune(self, X, y):
        # Perform initial validation
        self.root.validation(X, y)
        best_accuracy = self.root.val_tree_accuracy
        best_node = None

        all_nodes = []
        stack = [self.root]  # Use stack instead of queue for efficiency

        # Collect all nodes in DFS order (avoids slow queue operations)
        while stack:
            node = stack.pop()
            if not node.is_leaf:
                all_nodes.append(node)
                stack.extend(node.children)

        # Store the original state of each node for quick rollback
        node_states = {node: (node.is_leaf, node.children[:]) for node in all_nodes}

        # Get original predictions for faster recomputation
        y_pred = self.root.predict(X)

        for node in all_nodes:
            if node.is_leaf:
                continue

            # Temporarily prune the node
            node.is_leaf = True
            node.children = []

            # Compute new predictions efficiently
            new_y_pred = self.root.predict(X)
            accuracy = np.mean(new_y_pred == y)  # Faster than np.sum()/len()

            # Select the best pruning step
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_node = node
            else:
                # Rollback if pruning is not beneficial
                node.is_leaf, node.children = node_states[node]

        # Apply the best pruning found and continue pruning iteratively
        if best_node is not None:
            best_node.is_leaf = True
            best_node.children = []
            self.prune(X, y)  # Recur for further pruning
        else:
            self.is_pruned = True

                    

class RandomForest:
    def __init__(self):
        self.trees = []
        self.num_trees = 0
        self.max_depth = None
        self.features_per_tree = None
        self.is_pruned = False

    def fit(self, X, y, max_depth = int(1e10), num_trees = 100, features_per_tree = None):
        self.trees = []
        self.num_trees = num_trees
        self.max_depth = max_depth
        n_samples, n_features = X.shape
        if features_per_tree is None:
            features_per_tree = int(np.sqrt(n_features))
        self.features_per_tree = features_per_tree

        for _ in range(self.num_trees):
            tree = DecisionTree()

            # Bootstrap sampling: Sample with replacement
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Feature Subset Sampling
            num_features = int(features_per_tree * n_features)  # Convert fraction to integer
            num_features = max(1, num_features)  # Ensure at least 1 feature is selected
            feature_indices = np.random.choice(n_features, size=num_features, replace=False)

            X_sample = X_sample[:, feature_indices]  

            # Train decision tree
            tree.fit(X_sample, y_sample, max_depth=self.max_depth)

            # Store tree and feature indices
            self.trees.append((tree, feature_indices))
            # print(f"Tree {_ + 1} trained")


    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.num_trees), dtype=int)
        
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset = X[:, feature_indices]  # Use the same features as during training
            predictions[:, i] = tree.predict(X_subset)

        return pd.DataFrame(predictions).mode(axis=1).iloc[:, 0].values

class Boosting:
    def __init__(self, num_estimators=50):
        self.num_estimators = num_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples  # Initialize sample weights

        for _ in range(self.num_estimators):
            model = DecisionTree()  # Weak learner
            model.fit(X, y, max_depth=10)
            predictions = model.predict(X)
            
            # Compute error
            error = np.sum(weights * (predictions != y)) / np.sum(weights)
            if error > 0.5:
                continue  # Skip weak learners worse than random guessing

            # Compute alpha (model weight)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))  # Avoid division by zero
            
            # Update sample weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalize

            self.models.append(model)
            self.alphas.append(alpha)
            print(f"Model {_ + 1} trained with alpha: {alpha:.4f}, error: {error:.4f}")

    def predict(self, X):
        # Weighted majority vote of weak learners
        final_predictions = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final_predictions += alpha * model.predict(X)

        return np.sign(final_predictions)