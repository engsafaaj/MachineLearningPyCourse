# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate sample data
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=42)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], s=30, c='gray')
plt.title("Sample Data")
plt.show()

# Perform Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = agglo.fit_predict(X)

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.title("Agglomerative Clustering")
plt.show()

# Create a dendrogram
linked = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=5, show_leaf_counts=True)
plt.title("Dendrogram")
plt.show()
