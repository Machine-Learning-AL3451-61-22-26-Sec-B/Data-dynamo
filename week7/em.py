import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate synthetic dataset
np.random.seed(0)
data = np.random.randn(300, 2)

# Load the dataset
@st.cache
def load_data():
    return pd.DataFrame(data, columns=['Feature 1', 'Feature 2'])

# Perform K-Means clustering
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

# Perform EM clustering
def em_clustering(data, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(data)
    labels = gmm.predict(data)
    return labels

# Plot clusters
def plot_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    st.pyplot()

# Main function
def main():
    st.title('Clustering Comparison: EM vs K-Means')

    # Load data
    data = load_data()

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Sidebar for user input
    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3)

    # Perform clustering
    kmeans_labels = kmeans_clustering(scaled_data, n_clusters)
    em_labels = em_clustering(scaled_data, n_clusters)

    # Plot clusters
    st.subheader('K-Means Clustering')
    plot_clusters(scaled_data, kmeans_labels, 'K-Means Clustering')

    st.subheader('EM Clustering')
    plot_clusters(scaled_data, em_labels, 'EM Clustering')

    # Evaluate clustering using silhouette score
    kmeans_score = silhouette_score(scaled_data, kmeans_labels)
    em_score = silhouette_score(scaled_data, em_labels)

    st.write('Silhouette Score:')
    st.write(f'- K-Means: {kmeans_score}')
    st.write(f'- EM: {em_score}')

# Run the main function
if __name__ == '__main__':
    main()
