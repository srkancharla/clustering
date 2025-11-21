import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from kmeans import run_kmeans

st.title("K-Means Clustering Visualization")
st.write("Paste your 2D data below (comma-separated values) to visualize K-Means clustering.")

# Input Data
default_data = "1.0, 2.0\n1.5, 1.8\n5.0, 8.0\n8.0, 8.0\n1.0, 0.6\n9.0, 11.0"
data_input = st.text_area("Input Data (x, y)", default_data, height=200)

# Parameters
k = st.slider("Number of Clusters (k)", min_value=1, max_value=10, value=3)

if st.button("Run K-Means"):
    try:
        # Parse data
        data_lines = data_input.strip().split('\n')
        data = []
        for line in data_lines:
            parts = line.split(',')
            if len(parts) == 2:
                data.append([float(parts[0]), float(parts[1])])
        
        X = np.array(data)
        
        if len(X) < k:
            st.error(f"Not enough data points ({len(X)}) for {k} clusters.")
        else:
            # Run K-Means
            labels, centers = run_kmeans(X, n_clusters=k)
            
            # Plot
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')
            ax.legend()
            ax.set_title(f"K-Means with k={k}")
            
            st.pyplot(fig)
            
    except ValueError:
        st.error("Invalid data format. Please ensure lines are 'x, y' numbers.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
