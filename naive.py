import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV
@st.cache
def load_data():
    return pd.read_csv('tennisdata.csv')

data = load_data()

# Preprocessing
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

le = LabelEncoder()
X = X.apply(le.fit_transform)
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Evaluating the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Displaying accuracy
st.write("Accuracy:", accuracy)
