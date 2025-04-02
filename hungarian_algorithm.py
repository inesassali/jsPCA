import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def apply_hungarian_algorithm(true_labels, cluster_labels):
    """
    Applies the Hungarian algorithm to map cluster labels to true labels.

    Parameters:
        true_labels (array-like): Ground truth labels.
        cluster_labels (array-like): Predicted cluster labels.

    Returns:
        pd.Categorical: Cluster labels mapped to the true label categories.
    """
    # Convert true labels and cluster labels to integer codes using pd.factorize
    true_labels_int, true_labels_categories = pd.factorize(true_labels)
    cluster_labels_int, cluster_labels_categories = pd.factorize(cluster_labels)

    # Generate confusion matrix
    confusion_matrix = np.zeros((len(np.unique(true_labels_int)), len(np.unique(cluster_labels_int))))
    for i, j in zip(true_labels_int, cluster_labels_int):
        confusion_matrix[i, j] += 1

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    label_mapping = dict(zip(col_ind, row_ind))

    # Map cluster labels back to true labels using the mapping
    mapped_labels = np.array([label_mapping.get(label, -1) for label in cluster_labels_int])

    # Convert mapped labels back to their original categories
    mapped_labels_categories = true_labels_categories[mapped_labels]
    
    return pd.Categorical(mapped_labels_categories, categories=true_labels_categories)
