import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.linalg import solve

def local_weighted_regression(X, y, tau):
    def kernel(x0, X, tau):
        return np.exp(-np.sum((X - x0)**2, axis=1) / (2 * tau**2))

    def predict(x0, X, y, tau):
        W = np.diag(kernel(x0, X, tau))
        XTWX = X.T @ W @ X
        XTWy = X.T @ W @ y
        theta = solve(XTWX, XTWy)
        return x0 @ theta

    return predict

def generate_data():
    np.random.seed(0)
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 1 * (0.5 - np.random.rand(16))
    return X, y

def plot_data(X, y, X_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='red', label='Training Data')
    ax.plot(X_test, y_pred, color='blue', label='LWR Fit')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Locally Weighted Regression")
    st.write("This is an implementation of Locally Weighted Regression using Streamlit.")
    
    tau = st.slider("Bandwidth parameter (tau)", 0.1, 1.0, 0.5)
    
    X, y = generate_data()
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])  # add intercept term
    X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])  # add intercept term

    predict = local_weighted_regression(X_b, y, tau)
    y_pred = np.array([predict(x0, X_b, y, tau) for x0 in X_test_b])

    plot_data(X, y, X_test, y_pred)

if __name__ == '__main__':
    main()
