import numpy as np
import torch
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import SteepestDescent
import scipy.sparse.linalg

# Function: joint_spca
def joint_spca(A_spca1, A_spca2, k):
    """
    Performs joint SPCA and optimization over 4 datasets' eigenvectors.

    Args:
        A_spca1, A_spca2, A_spca3, A_spca4: SPCA matrices (e.g., adjacency matrix after SPCA).
        k: The number of eigenvectors to compute (default is 50).

    Returns:
        Optimized U matrix (joint basis for all datasets),
        SVD components (U, S, VT),
        Eigenvalues for each dataset (lambda1, lambda2, lambda3, lambda4),
        Normalized error (error_normalized).
    """
    # Compute eigenvectors and eigenvalues for each A_spca matrix
    S1, U1 = scipy.sparse.linalg.eigsh(A_spca1, k=k)
    S2, U2 = scipy.sparse.linalg.eigsh(A_spca2, k=k)
   

    # Concatenate the weighted eigenvectors
    Uis = np.concatenate([U1 * S1, U2 * S2], axis=1)

    # SVD of the concatenated matrix
    U, S, VT = scipy.sparse.linalg.svds(Uis, k=k)
    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("VT shape:", VT.shape)

    # Calculate the eigenvalues in the joint space
    lambda1 = S**2 * np.diag((VT[:,:k]/S1) @ VT[:,:k].T) # the eigenvalues of A1 in U
    lambda2 = S**2 * np.diag((VT[:,k:]/S2) @ VT[:,k:].T) # the eigenvalues of A2 in U

    # Print the results of the eigenvalues for each dataset
    print("Eigenvalues of A1 in U:")
    print(lambda1)
    print("Eigenvalues of A2 in U:")
    print(lambda2)
   

    # Define the diff_pmv function for optimization
    diff_pmv = lambda x: (A_spca1(x) - U @ (lambda1[:, None] * (U.T @ x))) if x.ndim == 2 else (A_spca1(x) - U @ (lambda1 * (U.T @ x)))

    # Create the linear operator for the optimization problem
    diff_op = scipy.sparse.linalg.LinearOperator(
        shape=A_spca1.shape,
        matvec=diff_pmv,
        rmatvec=diff_pmv
    )

    # Compute error
    error = scipy.sparse.linalg.svds(diff_op, k=1, return_singular_vectors=False)
    
    # Normalize the error
    error_normalized = error / S1.max()
    print("Optimization error:", error_normalized)

    # Convert eigenvectors and eigenvalues to PyTorch tensors
    U1tensor = torch.tensor(U1.copy())
    U2tensor = torch.tensor(U2.copy())
    S1tensor = torch.tensor(S1.copy())
    S2tensor = torch.tensor(S2.copy())
    Utensor = torch.tensor(U.copy())

    # Define the manifold for Stiefel optimization
    manifold = Stiefel(U1.shape[0], k)

    # Define the cost function for Pymanopt optimization
    @pymanopt.function.pytorch(manifold)
    def cost(point):
        UTU1 = point.T @ U1tensor
        UTU2 = point.T @ U2tensor
        

        return (
            -torch.sum(torch.diag((UTU1 * S1tensor) @ UTU1.T) ** 2)
            -torch.sum(torch.diag((UTU2 * S2tensor) @ UTU2.T) ** 2)
        )

    # Create the problem for Pymanopt optimization
    problem = pymanopt.Problem(manifold, cost)
    optimizer = SteepestDescent(max_iterations=100)

    # Perform optimization
    result = optimizer.run(problem, initial_point=np.array(Utensor))

    # Return the optimized joint basis matrix and the required components
    return {
        'optimized_matrix': result.point,
        'error_normalized': error_normalized
    }


# Example usage:
# Assuming that A_spca1, A_spca2, A_spca3, A_spca4 are already computed
# result = joint_spca(A_spca1, A_spca2, A_spca3, A_spca4)

# Accessing the results
# optimized_matrix = result['optimized_matrix']
# U = result['U']
# S = result['S']
# VT = result['VT']
# lambda1 = result['lambda1']
# error_normalized = result['error_normalized']
