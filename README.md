# jsPCA



**Joint Spatial PCA (jsPCA)** is a fast and interpretable method for analyzing spatial transcriptomics data across single or multiple tissue slices or samples. It is based on a spatial covariance, defined as the product of the gene expression covariance with the spatial autocorrelation. The principal components of this spatial covariance yield a biologically meaningful low-dimensional representation, from which:



* **Spatial domains** are derived by simple clustering.
* **Spatially variable genes (SVGs)** can be identified directly from the principal component coefficients.
* **Multi-slice and multi-sample joint analysis** is enabled via joint diagonalization of the spatial covariance matrices, without requiring spatial alignment.



This simple mathematical framework allows jsPCA to efficiently capture biologically relevant spatial structure in transcriptomics data.

## Project Structure
```bash

jsPCA_project/
│
├── data/                   # Input datasets (.h5ad)
├── results/                # Output results (clustering, metrics, plots)
├── scripts/                # Main scripts for mono-slice and multi-slice analysis
│   ├── jspca_monosslice.py
│   ├── jspca_multislice.py
├── utils/                  # Utility modules
│   ├── adjacency_matrix_knn.py
│   ├── clustering_utils.py
│   ├── eigen_utils.py
│   ├── hungarian_algorithm.py
│   ├── jspca_utils.py
│   ├── projection_utils.py
│   ├── spca_utils.py
├── .gitignore              # Git ignore file
├── README.md               # Project overview and instructions
├── environment.yml         # Conda environment
└── requirements.txt        # Python package dependencies

```


## Installation



```bash

conda env create -f environment.yml

conda activate jspca-env

```



## Usage

### Mono-slice Analysis

```bash

python scripts/jspca-monoslice.py

```

### Multi-slice or Multi-sample Analysis

```bash

python scripts/jspca-multislice.py

```
## Outputs
- Processed `.hsad` files with clustering results
- Excel files with **ARI/NMI** and **runtime/memory** metrics
- Best configuration automatically selected


## Datasets

Input format: .h5ad



Required fields:



* Spatial coordinates → .obsm\["spatial"]



* Ground truth labels → .obs\["ground\_truth"] or .obs\["annotation"]



