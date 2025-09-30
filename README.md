# jsPCA



**Joint Spatial PCA (jsPCA)** is a fast and interpretable method for analyzing spatial transcriptomics data across single or multiple tissue slices or samples. It is based on a spatial covariance, defined as the product of the gene expression covariance with the spatial autocorrelation. The principal components of this spatial covariance yield a biologically meaningful low-dimensional representation, from which:



* **Spatial domains** are derived by simple clustering.
* **Spatially variable genes (SVGs)** can be identified directly from the principal component coefficients.
* **Multi-slice and multi-sample joint analysis** is enabled via joint diagonalization of the spatial covariance matrices, without requiring spatial alignment.



This simple mathematical framework allows jsPCA to efficiently capture biologically relevant spatial structure in transcriptomics data.



## Installation



```bash

conda env create -f environment.yml

conda activate jspca-env





