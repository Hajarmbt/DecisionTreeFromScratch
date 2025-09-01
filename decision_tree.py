import numpy as np
from sklearn.datasets import load_iris
from collections import Counter

# --- Helper functions ---
def entropy(y):
    """Calculate the entropy of a label array y."""
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def information_gain(y, left_y, right_y):
    """Calculate information gain from a potential split."""
    p = len(left_y) / len(y)
    return entropy(y) - (p * entropy(left_y) + (1 - p) * entropy(right_y))

# --- Node class ---
class Node:
    """A node in the Decision Tree.

    Attributes:
        feature: The feature index used for splitting
        threshold: The threshold value for the split
        left: Left child node
        right: Right child node
        value: Class label if this is a leaf node
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# --- Decision Tree class ---
class DecisionTree:
    """Decision Tree classifier implemented from scratch using NumPy.

    Hyperparameters:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        min_impurity: Minimum information gain required to make a split
    """
    def __init__(self, max_depth=3, min_samples_split=2, min_impurity=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.root = None

    def fit(self, X, y):
        """Fit the decision tree to the data."""
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            # Pure node or max depth reached
            return Node(value=self._most_common_label(y))

        # Find the best split
        best_feature, best_threshold, best_gain = self._best_split(X, y, n_features)

        # Stop if no good split
        if best_gain < self.min_impurity or best_feature is None:
            return Node(value=self._most_common_label(y))

        # Split dataset
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold

        # Recursively build left and right subtrees
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, n_features):
        """Find the best feature and threshold to split the data."""
        best_gain = -1
        split_idx, split_thresh = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left_y = y[X[:, feature] <= thresh]
                right_y = y[X[:, feature] > thresh]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gain = information_gain(y, left_y, right_y)
                # Check if this split is better than previous best
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = thresh
        return split_idx, split_thresh, best_gain

    def _most_common_label(self, y):
        """Return the most common class label in y."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse the tree to predict the class for a single sample."""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def print_tree(self, node=None, depth=0):
        """Print a text-based visualization of the decision tree."""
        if node is None:
            node = self.root
        prefix = "  " * depth
        if node.value is not None:
            print(f"{prefix}Leaf: Class={node.value}")
        else:
            print(f"{prefix}Feature {node.feature} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)


# --- Load Iris dataset ---
iris = load_iris()
X, y = iris.data, iris.target  # All three classes

# --- Train and evaluate ---
tree = DecisionTree(max_depth=4, min_samples_split=2, min_impurity=1e-7)
tree.fit(X, y)
y_pred = tree.predict(X)
accuracy = np.sum(y_pred == y) / len(y)
print("Accuracy:", accuracy)

# --- Print tree structure ---
print("\nDecision Tree Structure:")
tree.print_tree()

