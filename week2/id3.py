import numpy as np
import pandas as pd
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def information_gain(X, y, threshold):
    parent_entropy = entropy(y)
    left_indices = X <= threshold
    right_indices = X > threshold
    if sum(left_indices) == 0 or sum(right_indices) == 0:
        return 0
    n = len(y)
    n_left, n_right = sum(left_indices), sum(right_indices)
    e_left, e_right = entropy(y[left_indices]), entropy(y[right_indices])
    child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
    ig = parent_entropy - child_entropy
    return ig

def best_split(X, y):
    best_feature, best_threshold, best_ig = None, None, 0
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            ig = information_gain(X[:, feature], y, threshold)
            if ig > best_ig:
                best_ig = ig
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=None):
    if len(np.unique(y)) == 1 or (max_depth is not None and depth >= max_depth):
        most_common_label = Counter(y).most_common(1)[0][0]
        return Node(value=most_common_label)

    feature, threshold = best_split(X, y)
    if feature is None:
        return Node(value=Counter(y).most_common(1)[0][0])

    left_indices = X[:, feature] <= threshold
    right_indices = X[:, feature] > threshold
    left_subtree = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
    right_subtree = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)
    return Node(feature, threshold, left_subtree, right_subtree)

def predict_sample(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_sample(node.left, x)
    else:
        return predict_sample(node.right, x)

def predict(node, X):
    return np.array([predict_sample(node, x) for x in X])
