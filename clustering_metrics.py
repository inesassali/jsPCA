import pandas as pd
from marker_genes import marker_score
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    accuracy_score
)
from sdmbench_utils import compute_HOM, compute_COM, compute_CHAOS, compute_PAS, compute_ASW

def evaluate_clustering(gt_result, cluster_labels, adata, domain_key, pred_key, gt_key):
    # Ensure ground truth and cluster labels are valid
    combined_df = pd.DataFrame({'gt_result': gt_result, 'cluster_labels': cluster_labels}).dropna()
    gt_result_clean = pd.factorize(combined_df['gt_result'])[0]
    cluster_labels_clean = pd.factorize(combined_df['cluster_labels'])[0]

    # Debugging: Check if the keys are present in adata
    print(f"Checking keys in adata.obs: {adata.obs.columns}")
    print(f"gt_key: {gt_key}, pred_key: {pred_key}")

    try:
        # Compute clustering metrics
        clustering_metrics = {
            'NMI': normalized_mutual_info_score(gt_result_clean, cluster_labels_clean),
            'ARI': adjusted_rand_score(gt_result_clean, cluster_labels_clean),
            'AMI': adjusted_mutual_info_score(gt_result_clean, cluster_labels_clean),
            'Accuracy': accuracy_score(gt_result_clean, cluster_labels_clean),
            'HOM': compute_HOM(adata, gt_key, pred_key),
            'COM': compute_COM(adata, gt_key, pred_key),
            'CHAOS': compute_CHAOS(adata, pred_key),
            'PAS': compute_PAS(adata, pred_key),
            'ASW': compute_ASW(adata, pred_key, 'spatial')  # Updated to pass adata and keys
        }

        # Compute marker scores
        moranI, gearyC = marker_score(adata, domain_key)
        clustering_metrics['Moran\'s I'] = moranI
        clustering_metrics['Geary\'s C'] = gearyC

    except KeyError as e:
        print(f"KeyError: {e}")
        print(f"Available columns: {adata.obs.columns}")

    return clustering_metrics
