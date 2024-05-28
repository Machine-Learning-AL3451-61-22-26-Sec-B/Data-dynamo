import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Preprocess the data (convert categorical to numerical)
for column in ['Outlook', 'Temperature', 'Humidity']:
    data[column] = data[column].astype('category').cat.codes

# Separate features and target variable
X = data.drop('PlayTennis', axis=1)
y = data['PlayTennis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naïve Bayesian classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Streamlit app
st.title('Play Tennis Predictor')

st.write('Enter the weather conditions to predict whether you can play tennis:')

outlook = st.selectbox('Outlook', ['Sunny', 'Overcast', 'Rainy'])
temperature = st.selectbox('Temperature', ['Hot', 'Mild', 'Cool'])
humidity = st.selectbox('Humidity', ['High', 'Normal'])
windy = st.selectbox('Windy', ['False', 'True'])

# Map input to numerical values
outlook_code = ['Sunny', 'Overcast', 'Rainy'].index(outlook)
temperature_code = ['Hot', 'Mild', 'Cool'].index(temperature)
humidity_code = ['High', 'Normal'].index(humidity)
windy_code = ['False', 'True'].index(windy)

# Predict
if st.button('Predict'):
    input_data = [[outlook_code, temperature_code, humidity_code, windy_code]]
    prediction = model.predict(input_data)
    st.write(f"You can {'play' if prediction[0] == 'Yes' else 'not play'} tennis.")
