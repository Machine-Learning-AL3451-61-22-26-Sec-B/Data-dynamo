import streamlit as st

# Function to initialize G and S
def initialize_hypotheses(attributes):
    G = [['?' for _ in range(len(attributes))]]
    S = [['0' for _ in range(len(attributes))]]
    return G, S

# Function to check if hypothesis is consistent with an example
def is_consistent(hypothesis, example):
    return all(h == e or h == '?' for h, e in zip(hypothesis, example))

# Function to update the specific hypothesis S
def update_S(S, example):
    for i, s in enumerate(S):
        if s == '0':
            S[i] = example[i]
        elif s != example[i]:
            S[i] = '?'
    return S

# Function to update the general hypotheses G
def update_G(G, S, example, attributes):
    G_new = []
    for g in G:
        if not is_consistent(g, example):
            for i, attr in enumerate(attributes):
                if g[i] == '?':
                    for val in attr:
                        if S[i] == '?' or S[i] == val:
                            g_new = g[:i] + [val] + g[i+1:]
                            if is_consistent(g_new, example):
                                G_new.append(g_new)
    return [g for g in G_new if any(s != g[i] and s != '?' for i, s in enumerate(S))]

# Function to remove more specific hypotheses from G
def remove_more_specific_hypotheses(G):
    G_new = []
    for g1 in G:
        if not any(all(g1[i] == g2[i] or g2[i] == '?' for i in range(len(g1))) for g2 in G if g1 != g2):
            G_new.append(g1)
    return G_new

# Function to run candidate elimination algorithm
def candidate_elimination(attributes, dataset):
    G, S = initialize_hypotheses(attributes)
    hypotheses = [(G.copy(), S.copy())]
    for example in dataset:
        if example[-1] == 'yes':  # Positive example
            S = update_S(S, example[:-1])
            G = [g for g in G if is_consistent(g, example[:-1])]
        else:  # Negative example
            G = update_G(G, S, example[:-1], attributes)
            G = remove_more_specific_hypotheses(G)
        hypotheses.append((G.copy(), S.copy()))
    return hypotheses

# Define the attributes and the dataset
attributes = [['Sunny', 'Rainy'], ['Warm', 'Cold'], ['Normal', 'High'], ['Strong', 'Weak'], ['Warm', 'Cool'], ['Same', 'Change']]
dataset = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'no'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'yes']
]

# Run the candidate elimination algorithm
hypotheses = candidate_elimination(attributes, dataset)

# Streamlit App
st.title('Candidate Elimination Algorithm')

for i, (G, S) in enumerate(hypotheses):
    st.header(f'After example {i}:')
    st.write('General Hypotheses (G):')
    st.write(G)
    st.write('Specific Hypothesis (S):')
    st.write(S)
