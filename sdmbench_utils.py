import scanpy as sc
import squidpy as sq
import copy
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    silhouette_score
)
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# CHAOS Utils
def _compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count += 1

    return np.sum(dist_val) / len(clusterlabel)

def fx_1NN(i, location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)

def _compute_PAS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results) / len(clusterlabel)

def fx_kNN(i, location_in, k, cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0

def compute_CHAOS(adata, pred_key, spatial_key='spatial'):
    return _compute_CHAOS(adata.obs[pred_key], adata.obsm[spatial_key])

def compute_PAS(adata, pred_key, spatial_key='spatial'):
    return _compute_PAS(adata.obs[pred_key], adata.obsm[spatial_key])

# ASW Utils
def compute_ASW(adata, pred_key, spatial_key='spatial'):
    d = squareform(pdist(adata.obsm[spatial_key]))
    return silhouette_score(X=d, labels=adata.obs[pred_key], metric='precomputed')

# SDMBench Utils
def res_search(adata, target_k=7, res_start=0.1, res_step=0.1, res_epochs=10): 
    print(f"Searching resolution to k={target_k}")
    res = res_start
    sc.tl.leiden(adata, resolution=res)

    old_k = len(adata.obs['leiden'].cat.categories)
    print("Res = ", res, "Num of clusters = ", old_k)

    run = 0
    while old_k != target_k:
        old_sign = 1 if old_k < target_k else -1
        sc.tl.leiden(adata, resolution=res + res_step * old_sign)
        new_k = len(adata.obs['leiden'].cat.categories)
        print("Res = ", res + res_step * old_sign, "Num of clusters = ", new_k)
        if new_k == target_k:
            res = res + res_step * old_sign
            print("Recommended res = ", str(res))
            return res
        new_sign = 1 if new_k < target_k else -1
        if new_sign == old_sign:
            res = res + res_step * old_sign
            print("Res changed to", res)
            old_k = new_k
        else:
            res_step /= 2
            print("Res changed to", res)
        if run > res_epochs:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res
        run += 1
    print("Recommended res = ", str(res))
    return res

def compute_ARI(adata, gt_key, pred_key):
    return adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_NMI(adata, gt_key, pred_key):
    return normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_HOM(adata, gt_key, pred_key):
    return homogeneity_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_COM(adata, gt_key, pred_key):
    return completeness_score(adata.obs[gt_key], adata.obs[pred_key])

def marker_score(adata, domain_key, top_n=5):
    import squidpy as sq 
    adata = adata.copy()  
    count_dict = adata.obs[domain_key].value_counts()
    adata = adata[adata.obs[domain_key].isin(count_dict[count_dict > 3].index)]
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, groupby=domain_key)
    selected_genes = []
    for i in range(top_n):
        toadd = list(adata.uns['rank_genes_groups']['names'][i])
        selected_genes.extend(toadd)
    selected_genes = np.unique(selected_genes)
    

    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        genes=selected_genes,
        n_perms=100,
        n_jobs=1,
    )
    sq.gr.spatial_autocorr(
        adata,
        mode="geary",
        genes=selected_genes,
        n_perms=100,
        n_jobs=1,
    )
    moranI = np.median(adata.uns["moranI"]['I'])
    gearyC = np.median(adata.uns["gearyC"]['C'])
    return moranI, gearyC

