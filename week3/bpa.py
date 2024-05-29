import streamlit as st
import numpy as np

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backpropagate(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backpropagate(X, y, output)

# Initialize the neural network
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

# Create Streamlit interface
st.title('Neural Network Training with Backpropagation')

# Input dataset
st.header('Training Data')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the neural network
if st.button('Train Neural Network'):
    epochs = st.number_input('Number of Epochs', min_value=1, value=10000)
    nn.train(X, y, epochs)
    st.success('Training complete!')

# Test the neural network
st.header('Test the Neural Network')
test_input = st.text_input('Test Input (comma-separated)', '0,0')
test_input = np.array([float(i) for i in test_input.split(',')]).reshape(1, -1)

if st.button('Test Neural Network'):
    output = nn.feedforward(test_input)
    st.write('Output:', output[0][0])

# Run the Streamlit app
if __name__ == '__main__':
    st.run()
