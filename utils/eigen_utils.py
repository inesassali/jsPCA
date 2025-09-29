# eigen_utils.py

import numpy as np
from scipy.sparse.linalg import eigsh

def compute_sorted_eigen(matrix, num_eigenvalues=50):
    """
    Computes the eigenvalues and eigenvectors of a given matrix,
    sorts them in descending order, and calculates the relative error.

    Parameters:
    - matrix: The input matrix (e.g., sparse matrix or LinearOperator).
    - num_eigenvalues: Number of eigenvalues and eigenvectors to compute.

    Returns:
    - sorted_eigenvalues: Sorted eigenvalues in descending order.
    - sorted_eigenvectors: Corresponding eigenvectors, aligned with the sorted eigenvalues.
    - rounded_relative_error: The relative error between the smallest and largest eigenvalues, rounded to 5 decimal places.
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(matrix, k=num_eigenvalues)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate relative error
    #relative_error = sorted_eigenvalues[-1] / sorted_eigenvalues[0]
    #rounded_relative_error = round(relative_error, 5)

    return sorted_eigenvalues, sorted_eigenvectors
