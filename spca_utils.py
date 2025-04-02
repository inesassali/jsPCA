# spca_utils.py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

def compute_spca(adata):
    """
    Perform Sparse PCA on the provided AnnData object after standardizing the connectivities.

    Parameters:
    adata: AnnData object containing the data in adata.X and connectivity information in adata.obsp['connectivities'].

    Returns:
    A LinearOperator that represents the Sparse PCA transformation.
    """
    # Extract the connectivity matrix from the AnnData object
    connectivities = adata.obsp['connectivities'].toarray()
    
    # Standardize the connectivities
    row_sums = connectivities.sum(axis=1)
    row_standardized_connectivities = connectivities / row_sums[:, np.newaxis]

    # Check for NaN values in standardized connectivities
    #if np.isnan(row_standardized_connectivities).any():
        #print("Warning: row_standardized_connectivities contains NaN values")
   # else:
        #print("row_standardized_connectivities does not contain NaN values")

    # Convert connectivity matrix to sparse CSR format
    L_sparse = sp.csr_matrix(row_standardized_connectivities)
    X_sparse = adata.X
    Xt = X_sparse.T
    LLT = 0.5 * (L_sparse + L_sparse.T)
    
    def spca(x):
        return Xt @ (LLT @ (X_sparse @ x))

    A = scipy.sparse.linalg.LinearOperator(
        shape=(X_sparse.shape[1], X_sparse.shape[1]), 
        matvec=spca, 
        rmatvec=spca
    )
    
    return A
