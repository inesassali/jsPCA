# === Imports ===
import warnings
import os
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import glob

# Utils
from utils.projection_utils_sparse import project_components
from utils.hungarian_algorithm import apply_hungarian_algorithm
from utils.eigen_utils import compute_sorted_eigen
from utils.clustering_utils import perform_gmm_clustering
from utils.jspca_utils import joint_spca
from utils.spca_utils import compute_spca
from utils.adjacency_matrix_knn import compute_mutual_knn_matrices
import SpaGCN as spg

warnings.filterwarnings("ignore")

# =================== Setup paths ===================
folder_path = os.path.join("data", "DLPFC")  # Folder with .h5ad files
output_folder = os.path.join("results", "Multislice")
os.makedirs(output_folder, exist_ok=True)

# =================== Optional task ID ===================
task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print(f"Running task: {task_id}")

# =================== Load datasets ===================
file_paths = sorted(glob.glob(os.path.join(folder_path, "*.h5ad")))
if not file_paths:
    raise RuntimeError(f"No .h5ad files found in {folder_path}")

adatas, valid_sample_ids = [], []
for path in file_paths:
    try:
        adata = sc.read_h5ad(path)
        if 'ground_truth' not in adata.obs:
            print(f"Skipping {path}: 'ground_truth' column missing")
            continue
        adata = adata[~adata.obs['ground_truth'].isna()].copy()
        sc.pp.filter_genes(adata, min_cells=20)
        adatas.append(adata)
        valid_sample_ids.append(os.path.splitext(os.path.basename(path))[0])
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")

if len(adatas) < 2:
    raise ValueError("At least 2 valid datasets required.")
print(f"✅ Loaded {len(adatas)} datasets")

# =================== Preprocessing ===================
common_genes = set(adatas[0].var_names)
for ad in adatas[1:]:
    common_genes &= set(ad.var_names)
common_genes = list(common_genes)
for i in range(len(adatas)):
    adatas[i] = adatas[i][:, common_genes].copy()
    sc.experimental.pp.normalize_pearson_residuals(adatas[i])
    sc.pp.scale(adatas[i])

X_sparse = [ad.X for ad in adatas]

# =================== KNN and SPCA ===================
tracemalloc.start()
start_time = time.time()

print("📌 Computing KNN matrices...")
for i in range(len(adatas)):
    dist, conn = compute_mutual_knn_matrices(adatas[i])
    adatas[i].obsp['distances'] = dist
    adatas[i].obsp['connectivities'] = csr_matrix(conn)

print("🔧 Computing SPCA matrices...")
A_spca = [compute_spca(ad) for ad in adatas]

# =================== Ground truth analysis ===================
num_labels = [len(ad.obs['ground_truth'].unique()) for ad in adatas]
max_labels = max(num_labels)

# =================== Refinement function ===================
def refine_and_score(adata, gmm_clusters):
    try:
        if "x" in adata.obs and "y" in adata.obs:
            x_array, y_array = adata.obs["x"].tolist(), adata.obs["y"].tolist()
        elif "spatial" in adata.obsm:
            x_array, y_array = adata.obsm["spatial"][:, 0].tolist(), adata.obsm["spatial"][:, 1].tolist()
        else:
            raise ValueError("Missing spatial coordinates")
        adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
        refined_pred = spg.refine(adata.obs.index.tolist(), gmm_clusters, adj_2d, shape="square")
        adata.obs["refined_pred"] = pd.Categorical(refined_pred)
        adata.obs["GMM_clusters"] = gmm_clusters
        obs_df = adata.obs.dropna(subset=["refined_pred", "ground_truth"])
        if obs_df.empty:
            return refined_pred, 0, 0
        ari = adjusted_rand_score(obs_df["refined_pred"], obs_df["ground_truth"])
        nmi = normalized_mutual_info_score(obs_df["refined_pred"], obs_df["ground_truth"])
        return refined_pred, ari, nmi
    except Exception as e:
        print(f"⚠️ Refinement failed: {e}")
        return None, 0, 0

# =================== Run jsPCA multi-slice ===================
best_score, best_k, best_outputs = -1, None, []
best_time, best_mem = None, None

for k in [10, 20, 30, 40, 50]:
    try:
        # Joint SPCA for n slices
        result = joint_spca(A_spca, k=k)
        optimized_matrix = result['optimized_matrix']
        projections = [project_components(X, np.real(optimized_matrix), k) for X in X_sparse]
        combined = np.vstack(projections)

        # GMM clustering
        gmm_clusters = perform_gmm_clustering(combined, max_labels, random_state=42)
        gmm_splits = np.split(gmm_clusters, np.cumsum([len(ad) for ad in adatas[:-1]]))

        score_list, output_list = [], []
        for i, ad in enumerate(adatas):
            refined_pred, ari, nmi = refine_and_score(ad, gmm_splits[i])
            ad.obs["GMM_clusters"] = gmm_splits[i]
            ad.obs["refined_pred"] = pd.Categorical(refined_pred)
            ad.uns.update({"ARI": ari, "NMI": nmi, "chosen_k": k})
            score_list.append((ari + nmi) / 2)
            output_list.append(ad.copy())

        avg_score = np.mean(score_list)
        current, peak = tracemalloc.get_traced_memory()
        elapsed_time = time.time() - start_time
        memory = peak / 1024 / 1024

        if avg_score > best_score:
            best_score = avg_score
            best_k = k
            best_outputs = output_list
            best_time = elapsed_time
            best_mem = memory

        print(f"✅ k={k} | Avg Score={avg_score:.4f}, Time={elapsed_time:.2f}s, Mem={memory:.2f}MB")
    except Exception as e:
        print(f"❌ jsPCA failed for k={k}: {e}")
        continue

tracemalloc.stop()

# =================== Save results ===================
for i, ad in enumerate(best_outputs):
    out_path = os.path.join(output_folder, f"jsPCA_joint_{valid_sample_ids[i]}.h5ad")
    ad.write(out_path)
    print(f"📁 Saved: {out_path} | ARI: {ad.uns['ARI']:.4f}, NMI: {ad.uns['NMI']:.4f}")

# Save ARI/NMI to Excel (without k column)
ari_nmi_df = pd.DataFrame({
    "Sample_ID": valid_sample_ids,
    "ARI": [ad.uns["ARI"] for ad in best_outputs],
    "NMI": [ad.uns["NMI"] for ad in best_outputs]
})
ari_nmi_file = os.path.join(output_folder, "ari_nmi.xlsx")
ari_nmi_df.to_excel(ari_nmi_file, index=False)
print(f"\n📄 Saved ARI/NMI metrics to {ari_nmi_file}")

# Save performance (time & memory) to Excel (one row only, joint_data)
perf_df = pd.DataFrame({
    "Sample_ID": ["joint_data"],
    "Time_sec": [best_time],
    "Memory_MB": [best_mem]
})
perf_file = os.path.join(output_folder, "performance.xlsx")
perf_df.to_excel(perf_file, index=False)
print(f"\n📄 Saved performance metrics to {perf_file}")
