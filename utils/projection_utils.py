# projection_utils.py

import numpy as np
import pandas as pd

def project_components(data, eigenvectors, num_components):
    """
    Projects data onto a given number of eigenvectors.

    Parameters:
    - data: The data to project (NumPy array or Pandas DataFrame).
    - eigenvectors: Eigenvectors for projection.
    - num_components: Number of components to use for projection.

    Returns:
    - A Pandas DataFrame containing the projected components.
    """
    data_array = data.values if isinstance(data, pd.DataFrame) else data
    if num_components > eigenvectors.shape[1]:
        raise ValueError(f"num_components is greater than available eigenvectors ({eigenvectors.shape[1]}).")
    projection = np.dot(data_array, eigenvectors[:, :num_components])
    columns = [f'pca_eigv{i+1}' for i in range(num_components)]
    return pd.DataFrame(projection, columns=columns)
