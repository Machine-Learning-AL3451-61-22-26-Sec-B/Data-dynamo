import pandas as pd
import numpy as np
import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Sample data for COVID-19 diagnosis
data = pd.DataFrame({
    'Fever': [1, 1, 1, 0, 0, 0],  # 1: Yes, 0: No
    'Cough': [1, 1, 0, 1, 0, 0],  # 1: Yes, 0: No
    'Fatigue': [1, 0, 1, 1, 0, 1],  # 1: Yes, 0: No
    'COVID': [1, 1, 1, 0, 0, 1]  # 1: Positive, 0: Negative
})

# Define the structure of the Bayesian Network
model = BayesianNetwork([('Fever', 'COVID'),
                         ('Cough', 'COVID'),
                         ('Fatigue', 'COVID')])

# Fit the model using Maximum Likelihood Estimator
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inference
inference = VariableElimination(model)

# Streamlit app
st.title('COVID-19 Diagnosis using Bayesian Network')

st.write('Enter the symptoms to diagnose the possibility of COVID-19:')

fever = st.selectbox('Fever', ['Yes', 'No'])
cough = st.selectbox('Cough', ['Yes', 'No'])
fatigue = st.selectbox('Fatigue', ['Yes', 'No'])

# Map input to numerical values
fever_code = 1 if fever == 'Yes' else 0
cough_code = 1 if cough == 'Yes' else 0
fatigue_code = 1 if fatigue == 'Yes' else 0

# Perform inference
if st.button('Diagnose'):
    evidence = {'Fever': fever_code, 'Cough': cough_code, 'Fatigue': fatigue_code}
    result = inference.map_query(variables=['COVID'], evidence=evidence)
    diagnosis = 'Positive' if result['COVID'] == 1 else 'Negative'
    st.write(f'The diagnosis for COVID-19 is: **{diagnosis}**')
