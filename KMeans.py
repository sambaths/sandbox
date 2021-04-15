import numpy as np 
from metrics import euclidean_distance
import matplotlib.pyplot as plt

np.random.seed(42)

class KMeans:
    def __init__(self, k=5, max_iters=100, plot_steps=True):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self._N, self._features = X.shape

        random_sample_idxs = np.random.choice(self._N, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            self.clusters = self._create_cluster(self.centroids)
            if self.plot_steps:
                self.plot()
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self._is_converged(centroids_old, self.centroids):
                break
            if self.plot_steps:
                self.plot()
        
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self._N)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _is_converged(self, centroids_old, centroids_new):
        distances = [euclidean_distance(centroids_old[i], centroids_new[i]) for i in range(self.k)]
        return sum(distances) == 0

    def _create_cluster(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, centroid) for centroid in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self._features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker='x', color='black', linewidth=2)
        plt.show()

    
if __name__ == "__main__":
    from sklearn import datasets

    X, y = datasets.make_blobs(centers=2, n_samples=500, n_features=2, shuffle=True, random_state=42)
    clusters = len(np.unique(y))
    print(clusters)

    kmeans = KMeans(k=clusters, max_iters=150, plot_steps=False)
    y_pred = kmeans.predict(X)

    kmeans.plot()


