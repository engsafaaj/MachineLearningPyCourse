import pandas as pd # To work with data
import matplotlib.pyplot as plt # For Plot Data
from sklearn.cluster import KMeans # For K-Means Algorithm
import joblib


# Set Data
data={
     'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
     'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
}

# Convert data to DataFrame
df=pd.DataFrame(data)

# Plot Data
plt.scatter(df['x'],df['y'],c='blue',s=50,alpha=0.6)
plt.title('Data Plot Before Apply K-Means')
plt.xlabel('X-Data')
plt.ylabel('Y-Data')
plt.show()


# Apply K-Means
kmeans=KMeans(n_clusters=3,random_state=0)
kmeans.fit(df) # Train
labels=kmeans.predict(df)
centroids=kmeans.cluster_centers_

# Plot
plt.scatter(df['x'],df['y'],c=labels,cmap='viridis',s=50)
plt.scatter(centroids[:,0],centroids[:,1],c='red',s=200,label='Centroids')
plt.title('K-Means')
plt.xlabel('XData')
plt.ylabel('YData')
plt.show()

print("Clusters Labels:")
print(centroids)

print("/nLabels:")
print(labels)

joblib.dump(kmeans,'kmeans.pkl')
print("The Model Saved !!")