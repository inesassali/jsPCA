import numpy as np
import torch
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import SteepestDescent
import scipy.sparse.linalg

def joint_spca(datasets, k):
    """
    Performs joint SPCA and optimization over multiple datasets' eigenvectors.
    
    Args:
        datasets: list of SPCA matrices (e.g., adjacency matrices after SPCA)
        k: The number of eigenvectors to compute.
    
    Returns:
        A dictionary with 'optimized_matrix' (joint basis for all datasets).
    """
    eigenvectors = []
    eigenvalues = []

    # Compute eigenvectors and eigenvalues for each dataset
    for A in datasets:
        S, U = scipy.sparse.linalg.eigsh(A, k=k)
        eigenvalues.append(S)
        eigenvectors.append(U)

    # Concatenate weighted eigenvectors
    weighted_eigenvectors = np.concatenate([U * S for U, S in zip(eigenvectors, eigenvalues)], axis=1)

    # SVD initialization
    U_init, _, _ = scipy.sparse.linalg.svds(weighted_eigenvectors, k=k)

    # Convert eigenvectors and eigenvalues to PyTorch tensors
    eigenvectors_tensors = [torch.tensor(U.copy()) for U in eigenvectors]
    eigenvalues_tensors = [torch.tensor(S.copy()) for S in eigenvalues]
    U_init_tensor = torch.tensor(U_init.copy())

    # Define manifold
    manifold = Stiefel(U_init.shape[0], k)

    # Define cost function
    @pymanopt.function.pytorch(manifold)
    def cost(point):
        loss = 0
        for U_tensor, S_tensor in zip(eigenvectors_tensors, eigenvalues_tensors):
            UTU = point.T @ U_tensor
            loss -= torch.sum(torch.diag((UTU * S_tensor) @ UTU.T) ** 2)
        return loss

    # Optimization
    problem = pymanopt.Problem(manifold, cost)
    optimizer = SteepestDescent(max_iterations=100, verbosity=0)
    result = optimizer.run(problem, initial_point=np.array(U_init_tensor))

    return {'optimized_matrix': result.point}
