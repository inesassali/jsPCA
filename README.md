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



## Tutorial
To analyze your own spatial transcriptomics data:

1. Prepare your datasets in `.h5ad` format with a `ground_truth`/`annotation` column.
2. Run the mono-slice or multi-slice script:

```bash

python scripts/jspca_monosslice.py
python scripts/jspca_multislice.py

```
3. Check the output folder (`results/jspca/Monoslice` or `results/jspca/Multislice`) for:
   * `.h5ad` files with clustering results and **best configuration automatically selected**
   * Excel files with **ARI/NMI** and **Time/Memory** metrics

## Datasets

- Input datasets must be in `.h5ad` format.
- Each dataset should contain **spatial coordinates** (`x/y` or `spatial` in `.obsm`).
- Each dataset should include **ground truth labels** (`ground_truth`) and/or **cell/tissue annotations** (`annotation`) for evaluation and interpretation.


## Citation
If you use this project, please cite the corresponding study:

Ines Assali, Paul Escande, Paul Villoutreix. (2025). jsPCA: fast, scalable, and interpretable identification of spatial domains and variable genes across multi-slice and multi-sample spatial transcriptomics data. *bioRxiv*. DOI: [10.1101/2025.09.16.676466](https://doi.org/10.1101/2025.09.16.676466)

