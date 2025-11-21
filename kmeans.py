import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_kmeans(data, n_clusters=3):
    """
    Runs K-Means clustering on the data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

if __name__ == "__main__":
    # Generate random data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    
    labels, centers = run_kmeans(X, n_clusters=3)
    print("Cluster centers:\n", centers)
