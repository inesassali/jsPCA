import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors

def estimate_optimal_k(coordinates, min_k=3, max_k=30):
    """
    Estimate optimal k using distance statistics and connectivity analysis.
    
    Parameters:
    - coordinates: Array of spatial coordinates
    - min_k: Minimum number of neighbors to consider
    - max_k: Maximum number of neighbors to consider
    
    Returns:
    - optimal_k: Estimated optimal number of neighbors
    """
    n_samples = coordinates.shape[0]
    
    # Rule of thumb initial estimate: k ≈ log(n)
    k_initial = int(np.log(n_samples))
    
    # Find optimal k using distance statistics
    nbrs = NearestNeighbors(n_neighbors=max_k).fit(coordinates)
    distances, _ = nbrs.kneighbors(coordinates)
    
    # Calculate average distance to k-th neighbor for different k
    avg_distances = np.mean(distances, axis=0)
    
    # Calculate the rate of change in average distance
    distance_gradient = np.gradient(avg_distances)
    
    # Find the "elbow" point where distance increase starts to stabilize
    elbow_idx = np.argmax(distance_gradient[k_initial:]) + k_initial
    
    # Ensure the selected k maintains connectivity
    optimal_k = min(max(elbow_idx, k_initial), max_k)
    
    # Verify graph connectivity
    knn_graph = kneighbors_graph(coordinates, optimal_k, mode='distance')
    G = nx.from_scipy_sparse_array(knn_graph)
    
    # If graph is not connected, increment k until it is
    while not nx.is_connected(G) and optimal_k < max_k:
        optimal_k += 1
        knn_graph = kneighbors_graph(coordinates, optimal_k, mode='distance')
        G = nx.from_scipy_sparse_array(knn_graph)
    
    return optimal_k

def compute_knn_matrices(adata, k=None):
    """
    Construct a KNN graph on 3D spatial coordinates with automatic k selection.
    
    Parameters:
    - adata: AnnData object containing the spatial coordinates in adata.obsm['spatial']
    - k: Number of nearest neighbors (if None, automatically determined)
    
    Returns:
    - dist_matrix: A distance matrix (numpy array)
    - connectivity_matrix: An adjacency matrix (scipy sparse matrix)
    """
    coordinates = adata.obsm['spatial']
    
    # Automatically determine k if not provided
    if k is None:
        k = estimate_optimal_k(coordinates)
        print(f"Automatically selected k = {k}")
    
    # Create KNN graph
    knn_graph = kneighbors_graph(coordinates, k, mode="distance", include_self=False)
    
    # Convert KNN graph to NetworkX for easy edge access
    G = nx.from_scipy_sparse_array(knn_graph)
    
    # Create distance matrix
    dist_matrix = np.full((len(coordinates), len(coordinates)), np.inf)
    
    # Calculate Euclidean distances for KNN edges
    for edge in G.edges():
        dist_matrix[edge[0], edge[1]] = np.linalg.norm(coordinates[edge[0]] - coordinates[edge[1]])
        dist_matrix[edge[1], edge[0]] = dist_matrix[edge[0], edge[1]]
    
    connectivity_matrix = knn_graph
    print("KNN graph construction successful!")
    
    return dist_matrix, connectivity_matrix