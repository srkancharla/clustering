import numpy as np
import matplotlib.pyplot as plt
from kmeans import run_kmeans

def plot_clusters(data, labels, centers):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title("K-Means Clustering")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Running K-Means Demo...")
    # Generate synthetic data
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(0, 1, (50, 2)),
        np.random.normal(5, 1, (50, 2)),
        np.random.normal(10, 1, (50, 2))
    ])
    
    labels, centers = run_kmeans(X, n_clusters=3)
    plot_clusters(X, labels, centers)
    print("Done.")
