import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from dbscan import run_dbscan

def plot_clusters(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title("DBSCAN Clustering")
    plt.show()

if __name__ == "__main__":
    print("Running DBSCAN Demo...")
    # Generate moon-shaped data (hard for K-Means, easy for DBSCAN)
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    
    labels = run_dbscan(X, eps=0.3, min_samples=5)
    plot_clusters(X, labels)
    print("Done.")
