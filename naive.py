import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data
def load_data():
    # Example data - replace with your own data source
    data = {
        'text': [
            'I love this movie, it is amazing!',
            'The movie was terrible and boring.',
            'Fantastic film with a great plot.',
            'Not my cup of tea, quite dull.',
            'An outstanding performance by the lead actor.',
            'I did not like the movie at all.'
        ],
        'label': [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
    }
    return pd.DataFrame(data)

# Train Naive Bayes Classifier
def train_model(data):
    X = data['text']
    y = data['label']

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)

    return model, vectorizer, accuracy, precision, recall

# Main function to run the Streamlit app
def main():
    st.title('Naive Bayesian Classifier for Document Classification')

    st.write("### Step 1: Load Data")
    data = load_data()
    st.write(data)

    st.write("### Step 2: Train Model")
    model, vectorizer, accuracy, precision, recall = train_model(data)
    
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")

    st.write("### Step 3: Make Predictions")
    user_input = st.text_area("Enter a document to classify", "")
    if user_input:
        user_input_transformed = vectorizer.transform([user_input])
        prediction = model.predict(user_input_transformed)[0]
        prediction_label = "Positive" if prediction == 1 else "Negative"
        st.write(f"Prediction: {prediction_label}")

if __name__ == '__main__':
    main()
