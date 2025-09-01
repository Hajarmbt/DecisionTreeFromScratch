# Decision Tree From Scratch

This repository contains a **custom implementation of a Decision Tree classifier** in Python, built from scratch.

## Features
- Implements a `DecisionTree` class and `Node` structure
- Supports tree visualization using `networkx` and `matplotlib`
- Predicts class labels on new data

## Example Usage
```python
from decision_tree import DecisionTree
from plot_tree import plot_tree_graph
import numpy as np

X = np.array([[0,1],[1,1],[0,0],[1,0]])
y = np.array([0,1,0,1])

tree = DecisionTree(max_depth=2)
tree.fit(X, y)

plot_tree_graph(tree.root)
print(tree.predict(np.array([[1,1]])))
