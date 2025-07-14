# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Display first few rows
print("Dataset Preview:\n", df.head())

# Select features for clustering
data = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the data (important for K-means)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Elbow Method to find optimal number of clusters
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Find Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()

# From elbow plot, suppose we choose k = 5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Add cluster labels to the original data
df['Cluster'] = clusters

# Display with cluster labels
print("\nClustered Data:\n", df.head())

# Visualization: 2D scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='tab10', data=df, s=100)
plt.title("Customer Segments (Based on Income vs Spending Score)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.legend(title='Cluster')
plt.show()
