import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph #  construit un graphe KNN sous forme de matrice sparse
from sklearn.neighbors import NearestNeighbors # Trouver les voisins les plus proches de chaque point

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
    
    # Règle empirique : une bonne valeur initiale pour k est log(n)
    #Si k est trop petit → le graphe est trop fragmenté (beaucoup de petits groupes, voire des points isolés).Si k est trop grand → le graphe devient trop     dense, et perd sa structure locale. log(n) augmente lentement avec n, ce qui donne un bon compromis : ni trop local, ni trop global.
    k_initial = int(np.log(n_samples))
    
    # Recherche des plus proches voisins jusqu'à max_k
    nbrs = NearestNeighbors(n_neighbors=max_k).fit(coordinates)
    distances, _ = nbrs.kneighbors(coordinates)
    
    # Moyenne des distances pour chaque rang de voisin 
    avg_distances = np.mean(distances, axis=0)
    
    # Calcule la variation (pente) entre les moyennes successives → pour détecter où la croissance ralentit(Variation des distances moyennes).
    distance_gradient = np.gradient(avg_distances)
    
    # Trouver le "coude" :On cherche où le changement de pente est le plus grand après k_initial (le point où augmenter k n'améliore plus beaucoup les distances (la courbe se stabilise))
    elbow_idx = np.argmax(distance_gradient[k_initial:]) + k_initial # np.argmax: donne l’indice local du plus grand gradient dans ce sous-tableau
    
    # Fixer un k raisonnable basé sur le coude (la courbe se stabilise) et les bornes (On garde k entre k_initial et max_k)
    optimal_k = min(max(elbow_idx, k_initial), max_k)
    
    #Créer le graphe KNN et tester la connectivité
    knn_graph = kneighbors_graph(coordinates, optimal_k, mode='distance')
    G = nx.from_scipy_sparse_array(knn_graph)
    
    # Assurer que le graphe est connexe (If graph is not connected, increment k until it is)
    while not nx.is_connected(G) and optimal_k < max_k:
        optimal_k += 1
        knn_graph = kneighbors_graph(coordinates, optimal_k, mode='distance')
        G = nx.from_scipy_sparse_array(knn_graph)
    
    return optimal_k

def compute_knn_matrices_3d(adata, k=None):
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


def compute_mutual_knn_matrices(adata, k=None):
    """
    Construct a mutual KNN graph from 3D spatial coordinates.
    Only mutual nearest neighbor pairs are connected (symmetric).
    
    Parameters:
    - adata: AnnData object containing spatial coordinates in adata.obsm['spatial']
    - k: Number of neighbors (auto-detected if None)
    
    Returns:
    - dist_matrix: Full distance matrix (numpy array)
    - connectivity_matrix: Symmetric sparse adjacency matrix (CSR)
    """
    coordinates = adata.obsm['spatial']
    
    # Auto-select k if not given
    if k is None:
        k = estimate_optimal_k(coordinates)
        print(f"Automatically selected k = {k}")
    
    # Trouver les k plus proches voisins
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(coordinates)  
    distances, indices = nbrs.kneighbors(coordinates)

    # Initialisation des matrices
    n = coordinates.shape[0]
    connectivity = np.zeros((n, n))
    dist_matrix = np.full((n, n), np.inf)
    
    #création du graphe mutuel
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            if i in indices[j]:  # Mutual neighbors only
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                connectivity[i, j] = 1
                connectivity[j, i] = 1
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    connectivity_matrix = csr_matrix(connectivity)
    
    print("Mutual KNN graph construction successful!")
    
    return dist_matrix, connectivity_matrix
