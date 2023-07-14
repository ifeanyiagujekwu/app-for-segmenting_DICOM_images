import numpy as np

class KMeans:
    def __init__(self, n_clusters, random_state=0, max_iters=100):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(self.random_state)

        # Randomly initialize centroids
        centroids = self._initialize_centroids(X)

        for _ in range(self.max_iters):
            # Assign data points to the nearest centroid
            labels = self._assign_labels(X, centroids)

            # Update centroids based on the mean of assigned data points
            new_centroids = self._update_centroids(X, labels)

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        self.labels_ = labels
        self.cluster_centers_ = centroids

    def _initialize_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[indices]
        return centroids

    def _assign_labels(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            centroids[i] = np.mean(X[labels == i], axis=0)
        return centroids
