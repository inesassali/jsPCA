# clustering_utils.py

from sklearn.mixture import GaussianMixture

def perform_gmm_clustering(data, num_clusters, random_state=None):
    """
    Performs Gaussian Mixture Model (GMM) clustering on the given data.

    Parameters:
    - data: The input data for clustering (NumPy array or similar structure).
    - num_clusters: The number of clusters to form.
    - random_state: Seed for random number generator (default is None for stochastic results).

    Returns:
    - A NumPy array of cluster labels for each data point.
    """
    # Initialize the Gaussian Mixture Model with a random state
    gmm = GaussianMixture(n_components=num_clusters, random_state=random_state)

    # Fit the GMM to the data
    gmm.fit(data)

    # Predict cluster labels
    return gmm.predict(data)
