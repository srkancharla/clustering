import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def run_dbscan(data, eps=0.5, min_samples=5):
    """
    Runs DBSCAN clustering on the data.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels

if __name__ == "__main__":
    # Generate random data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    
    labels = run_dbscan(X, eps=0.1, min_samples=5)
    print("Labels:\n", labels)
