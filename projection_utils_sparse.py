import numpy as np
import pandas as pd
from scipy.sparse import issparse

def project_components(data, eigenvectors, num_components):
    """
    Projects sparse or dense data onto a given number of eigenvectors.

    Parameters:
    - data: The data to project (NumPy array, Pandas DataFrame, or SciPy sparse matrix).
    - eigenvectors: Eigenvectors for projection (NumPy array).
    - num_components: Number of components to use for projection.

    Returns:
    - A Pandas DataFrame containing the projected components.
    """
    if num_components > eigenvectors.shape[1]:
        raise ValueError(f"num_components is greater than available eigenvectors ({eigenvectors.shape[1]}).")

    # Handle different data types
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    elif issparse(data):  # Check if the data is sparse
        data_array = data
    else:
        data_array = np.asarray(data)

    # Use efficient sparse matrix multiplication
    projection = data_array @ eigenvectors[:, :num_components] if not issparse(data) else data_array.dot(eigenvectors[:, :num_components])

    # Convert to DataFrame
    columns = [f'pca_eigv{i+1}' for i in range(num_components)]
    return pd.DataFrame(projection.toarray() if issparse(projection) else projection, columns=columns)
