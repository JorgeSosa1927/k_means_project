import numpy as np
import pandas as pd
from time import time

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[labels == i]
        if cluster_points.shape[0] > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids

def kmeans(data, k, max_iter=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return centroids, labels

if __name__ == "__main__":
    data = pd.read_csv('../benchmarks/dataset.csv').values
    k = 5
    start_time = time()
    centroids, labels = kmeans(data, k)
    end_time = time()
    print(f"Execution time (Sequential): {end_time - start_time:.2f} seconds")
