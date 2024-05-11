# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Load data from CSV
    data = pd.read_csv('tennisdata.csv')
    st.write("The first 5 values of data are:")
    st.write(data.head())

    # Obtain Train data and Train output
    X = data.iloc[:, :-1]
    st.write("The First 5 values of train data are:")
    st.write(X.head())

    y = data.iloc[:, -1]
    st.write("The first 5 values of Train output are:")
    st.write(y.head())

    # Convert them to numbers
    le_outlook = LabelEncoder()
    X.Outlook = le_outlook.fit_transform(X.Outlook)

    le_Temperature = LabelEncoder()
    X.Temperature = le_Temperature.fit_transform(X.Temperature)

    le_Humidity = LabelEncoder()
    X.Humidity = le_Humidity.fit_transform(X.Humidity)

    le_Windy = LabelEncoder()
    X.Windy = le_Windy.fit_transform(X.Windy)

    st.write("Now the Train data is:")
    st.write(X.head())

    le_PlayTennis = LabelEncoder()
    y = le_PlayTennis.fit_transform(y)
    st.write("Now the Train output is:")
    st.write(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    st.write("Accuracy is:", accuracy_score(classifier.predict(X_test), y_test))

if __name__ == "__main__":
    main()
