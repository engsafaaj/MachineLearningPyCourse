import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data points
data = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply DBSCAN
db = DBSCAN(eps=1.5, min_samples=2).fit(data_scaled)

# Get cluster labels
labels = db.labels_

# Print results
print("Cluster labels:", labels)

# Visualize clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
plt.title('DBSCAN Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
