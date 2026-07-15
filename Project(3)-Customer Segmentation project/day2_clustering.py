import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Select features for clustering
X = data[['Age', 'Spending Score (1-100)']]
wcss = []  # Within-Cluster Sum of Squares

# Test k from 1 to 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.show()
# Apply K-Means with chosen k (e.g., 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to dataset
data['Cluster'] = y_kmeans
print(data.head())
# ✅ Summarize clusters for bar chart
cluster_summary = data.groupby('Cluster')[['Age', 'Spending Score (1-100)']].mean()

# Plot bar chart instead of scatter
cluster_summary.plot(kind='bar', figsize=(8,6))
plt.title("Age and Spending by Cluster")
plt.ylabel("Value")
plt.show()
# Save to new CSV for Day 3
data.to_csv("Clustered_Customers.csv", index=False)





