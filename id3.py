import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, value=None, result=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.children = {}

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(data, feature_name, labels):
    total_entropy = entropy(labels)
    values, counts = np.unique(data[feature_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(labels[data[feature_name] == values[i]]) for i in range(len(values))])
    information_gain = total_entropy - weighted_entropy
    return information_gain

def build_tree(data, labels, features):
    if len(np.unique(labels)) == 1:
        return Node(result=labels.iloc[0])
    
    if len(features) == 0:
        return Node(result=labels.mode()[0])
    
    max_gain = -1
    best_feature = None
    for feature in features:
        gain = information_gain(data, feature, labels)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature
    
    root = Node(feature=best_feature)
    values = np.unique(data[best_feature])
    for value in values:
        sub_data = data[data[best_feature] == value]
        sub_labels = labels[data[best_feature] == value]
        if len(sub_data) == 0:
            root.children[value] = Node(result=labels.mode()[0])
        else:
            root.children[value] = build_tree(sub_data, sub_labels, [f for f in features if f != best_feature])
    return root

def classify(root, sample):
    if root.result is not None:
        return root.result
    value = sample[root.feature]
    if value not in root.children:
        return None
    return classify(root.children[value], sample)

# Sample data
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

labels = data['PlayTennis']
features = data.columns[:-1]

# Build the decision tree
root = build_tree(data, labels, features)

# Classify a new sample
sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
classification = classify(root, sample)
print("Predicted class:", classification)
