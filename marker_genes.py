# marker_genes.py

import numpy as np
import scanpy as sc
import squidpy as sq

def marker_score(adata, domain_key, top_n=5):
    adata = adata.copy()  # Copy to avoid changes to original data
    count_dict = adata.obs[domain_key].value_counts()
    adata = adata[adata.obs[domain_key].isin(count_dict[count_dict > 3].index)]
    
    # Normalize and log-transform data
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    
    # Rank genes based on groups
    sc.tl.rank_genes_groups(adata, groupby=domain_key)
    
    # Collect top genes
    selected_genes = []
    for i in range(top_n):
        toadd = list(adata.uns['rank_genes_groups']['names'][i])
        selected_genes.extend(toadd)
    selected_genes = np.unique(selected_genes)
    
    # Compute spatial neighbors and autocorrelations using Squidpy
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
    
    # Return Moran's I and Geary's C statistics
    moranI = np.median(adata.uns["moranI"]['I'])
    gearyC = np.median(adata.uns["gearyC"]['C'])
    return moranI, gearyC
