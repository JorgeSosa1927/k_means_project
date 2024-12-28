import numpy as np
import pandas as pd
from time import time
from joblib import Parallel, delayed

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def compute_distances(point, centroids):
    return np.linalg.norm(point - centroids, axis=1)

def assign_clusters_parallel(data, centroids, n_jobs=-1):
    distances = Parallel(n_jobs=n_jobs)(delayed(compute_distances)(point, centroids) for point in data)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[labels == i]
        if cluster_points.shape[0] > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids

def kmeans_parallel(data, k, max_iter=100, tol=1e-4, n_jobs=-1):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        labels = assign_clusters_parallel(data, centroids, n_jobs)
        new_centroids = update_centroids(data, labels, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return centroids, labels

if __name__ == "__main__":
    data = pd.read_csv('../benchmarks/dataset.csv').values
    k = 5
    start_time = time()
    centroids, labels = kmeans_parallel(data, k)
    end_time = time()
    print(f"Execution time (Parallel): {end_time - start_time:.2f} seconds")
