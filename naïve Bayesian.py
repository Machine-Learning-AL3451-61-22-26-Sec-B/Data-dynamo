import streamlit as st
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
@st.cache_data
def load_data():
    newsgroups = fetch_20newsgroups(subset='all')
    X, y = newsgroups.data, newsgroups.target
    return X, y, newsgroups.target_names

# Split and preprocess data
@st.cache_data
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# Train model
def train_model(_X_train_tfidf, y_train):
    model = MultinomialNB()
    model.fit(_X_train_tfidf, y_train)
    return model

# Evaluate model
def evaluate_model(model, _X_test_tfidf, y_test):
    y_pred = model.predict(_X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall

# Main application
def main():
    st.title('Naïve Bayesian Classifier for Document Classification')

    # Load data
    X, y, target_names = load_data()

    # Display target names
    st.write("Target Names:")
    st.write(target_names)

    # Preprocess data
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(X, y)

    # Train model
    model = train_model(X_train_tfidf, y_train)

    # Evaluate model
    accuracy, precision, recall = evaluate_model(model, X_test_tfidf, y_test)

    # Display metrics
    st.write(f'**Accuracy:** {accuracy:.4f}')
    st.write(f'**Precision:** {precision:.4f}')
    st.write(f'**Recall:** {recall:.4f}')

    # Allow user to input text for classification
    st.write("### Classify a New Document")
    user_input = st.text_area("Enter the document text here:")
    if st.button("Classify"):
        if user_input:
            user_input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(user_input_tfidf)
            predicted_category = target_names[prediction[0]]
            st.write(f'This document is classified as: **{predicted_category}**')

if __name__ == '__main__':
    main()
