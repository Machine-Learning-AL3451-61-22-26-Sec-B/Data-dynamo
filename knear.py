import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Synthetic Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris_data['data'], iris_data['target']],
                       columns= iris_data['feature_names'] + ['target'])

# Streamlit app
st.title('k-Nearest Neighbors (k-NN) Classifier for Iris Dataset')

# Sidebar for user input
test_size = st.sidebar.slider('Test Size', min_value=0.1, max_value=0.5, step=0.05, value=0.2)
n_neighbors = st.sidebar.slider('Number of Neighbors (k)', min_value=1, max_value=10, value=3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('target', axis=1), iris_df['target'], test_size=test_size, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
knn_classifier.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display results
st.write(f'Accuracy: {accuracy:.2f}')
st.write('Confusion Matrix:')
st.write(cm)

# Print correct and wrong predictions
st.subheader('Correct Predictions:')
correct_predictions = X_test[y_test == y_pred]
st.write(correct_predictions)

st.subheader('Wrong Predictions:')
wrong_predictions = X_test[y_test != y_pred]
st.write(wrong_predictions)
