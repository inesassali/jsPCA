# === Imports ===
import os
import glob
import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import SpaGCN as spg
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Utils
from utils.hungarian_algorithm import apply_hungarian_algorithm
from utils.projection_utils import project_components
from utils.eigen_utils import compute_sorted_eigen
from utils.clustering_utils import perform_gmm_clustering
from utils.adjacency_matrix_knn import compute_mutual_knn_matrices
from utils.spca_utils import compute_spca

# Suppress warnings
warnings.filterwarnings("ignore")

# =================== CONFIGURATION ===================
folder_path = os.path.join("data", "DLPFC")  # Folder with .h5ad files
output_folder = os.path.join("results", "Monoslice")
os.makedirs(output_folder, exist_ok=True)

file_list = sorted(glob.glob(os.path.join(folder_path, "*.h5ad")))
if not file_list:
    print("⚠️ No .h5ad files found in", folder_path)
    exit(0)

# =================== LOOP OVER FILES ===================
for file_path in file_list:
    file_to_read = os.path.basename(file_path)
    sample_name = file_to_read.replace(".h5ad", "")
    print(f"\n=== Processing file: {file_to_read} ===")

    # Read and preprocess
    adata = ad.read_h5ad(file_path)
    sc.pp.filter_genes(adata, min_cells=20)
    sc.experimental.pp.normalize_pearson_residuals(adata)
    sc.pp.scale(adata)

    # Check if 'ground_truth' column exists
    if 'ground_truth' not in adata.obs:
        print(f"Skipping {file_to_read}: 'ground_truth' not found.")
        continue

    # Count number of labels
    num_labels = adata.obs['ground_truth'].nunique()

    # Start memory and time tracking after calculating num_labels
    tracemalloc.start()
    start_time = time.time()

    # Compute KNN matrices
    dist_matrix, connectivity_matrix = compute_mutual_knn_matrices(adata)
    connectivity_matrix = csr_matrix(connectivity_matrix)
    adata.obsp['distances'] = dist_matrix
    adata.obsp['connectivities'] = connectivity_matrix

    # Loop over different numbers of eigenvalues
    best_score, best_data = -1, None
    for k in [10, 20, 30, 40, 50]:
        print(f"\nTesting num_eigenvalues = {k}")

        try:
            # Compute SPCA
            A = compute_spca(adata)
            eigvals, eigvecs = compute_sorted_eigen(A, k)
            projection_df = project_components(adata.X, np.real(eigvecs), k)

            if np.isnan(projection_df.values).any():
                print(f"NaNs in projection. Skipping k={k}")
                continue

            # Perform GMM clustering
            gmm_clusters = perform_gmm_clustering(projection_df, num_labels, random_state=42)

            # Spatial refinement with SpaGCN
            try:
                if "x" in adata.obs and "y" in adata.obs:
                    x_array, y_array = adata.obs["x"].tolist(), adata.obs["y"].tolist()
                elif "spatial" in adata.obsm:
                    x_array, y_array = adata.obsm["spatial"][:, 0].tolist(), adata.obsm["spatial"][:, 1].tolist()
                else:
                    raise ValueError("Missing spatial coordinates")

                adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
                refined_pred = spg.refine(
                    sample_id=adata.obs.index.tolist(),
                    pred=gmm_clusters,
                    dis=adj_2d,
                    shape="hexagon"
                )

                adata.obs["refined_pred"] = pd.Categorical(refined_pred)
                obs_df = adata.obs.dropna(subset=["refined_pred", "ground_truth"])
                ari = adjusted_rand_score(obs_df["refined_pred"], obs_df["ground_truth"])
                nmi = normalized_mutual_info_score(obs_df["refined_pred"], obs_df["ground_truth"])
                score = (ari + nmi) / 2

            except Exception as refine_error:
                print(f"⚠️ Refinement failed at k={k}: {refine_error}")
                continue

            # Measure peak memory usage and elapsed time
            current, peak = tracemalloc.get_traced_memory()
            elapsed = time.time() - start_time
            memory = peak / 1024 / 1024  # in MB

            print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, Score: {score:.4f}, Time: {elapsed:.2f}s, Mem: {memory:.2f}MB")

            # Keep best result
            if score > best_score:
                best_score = score
                best_data = adata.copy()
                best_data.obs['GMM_clusters'] = gmm_clusters
                best_data.obs['refined_pred'] = pd.Categorical(refined_pred)
                best_data.uns.update({
                    'ARI': ari,
                    'NMI': nmi,
                    'time': elapsed,
                    'memory': memory,
                    'best_num_eigenvalues': k
                })

        except Exception as e:
            print(f"❌ Error at k={k}: {e}")
            continue

    tracemalloc.stop()  # Stop memory tracking

    # =================== SAVE RESULTS ===================
    if best_data is not None:
        out_file = os.path.join(output_folder, f"jsPCA_{sample_name}_results.h5ad")
        best_data.write(out_file)
        print(f"💾 Result saved to {out_file}")

        # Save ARI/NMI scores
        pd.DataFrame({
            'Sample': [sample_name],
            'ARI': [best_data.uns['ARI']],
            'NMI': [best_data.uns['NMI']]
        }).to_excel(os.path.join(output_folder, f"{sample_name}_jspca_ari_nmi.xlsx"), index=False)

        # Save performance metrics (time & memory)
        pd.DataFrame({
            'Sample': [sample_name],
            'Time': [best_data.uns['time']],
            'Memory': [best_data.uns['memory']]
        }).to_excel(os.path.join(output_folder, f"{sample_name}_jspca_perf.xlsx"), index=False)

    else:
        print(f"⚠️ No valid result for {sample_name}")
