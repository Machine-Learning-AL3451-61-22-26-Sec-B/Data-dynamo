import streamlit as st
import numpy as np

# Sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

def train_neural_network(X, y, epoch, lr, inputlayer_neurons, hiddenlayer_neurons, output_neurons):
    # Weight and bias initialization
    wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
    bh = np.random.uniform(size=(1, hiddenlayer_neurons))
    wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    bout = np.random.uniform(size=(1, output_neurons))

    # Draws a random range of numbers uniformly of dim x*y
    for i in range(epoch):
        hinp1 = np.dot(X, wh)
        hinp = hinp1 + bh
        hlayer_act = sigmoid(hinp)
        outinp1 = np.dot(hlayer_act, wout)
        outinp = outinp1 + bout
        output = sigmoid(outinp)

        # Backpropagation
        EO = y - output
        outgrad = derivatives_sigmoid(output)
        d_output = EO * outgrad
        EH = d_output.dot(wout.T)
        hiddengrad = derivatives_sigmoid(hlayer_act)  # How much hidden layer weights
        d_hiddenlayer = EH * hiddengrad

        wout += hlayer_act.T.dot(d_output) * lr  # Dot product of next layer error and
        wh += X.T.dot(d_hiddenlayer) * lr

    return output

def main():
    st.title("Backpropagation with Streamlit")

    st.sidebar.title("Parameters")
    epoch = st.sidebar.slider("Epoch", min_value=1000, max_value=10000, step=1000, value=7000)
    lr = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
    inputlayer_neurons = st.sidebar.number_input("Input Layer Neurons", min_value=1, max_value=10, value=2)
    hiddenlayer_neurons = st.sidebar.number_input("Hidden Layer Neurons", min_value=1, max_value=10, value=3)
    output_neurons = st.sidebar.number_input("Output Neurons", min_value=1, max_value=10, value=1)

    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
    y = np.array(([92], [86], [89]), dtype=float)

    X_normalized = X / np.amax(X, axis=0)  # maximum of X array longitudinally
    y_normalized = y / 100

    if st.button("Train Neural Network"):
        predicted_output = train_neural_network(X_normalized, y_normalized, epoch, lr, inputlayer_neurons, hiddenlayer_neurons, output_neurons)
        st.write("Predicted Output: \n", predicted_output)

if __name__ == "__main__":
    main()
