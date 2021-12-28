# scVIDE.jl
This is the implementation for **"scVIDE: Single Cell Variational Inference for Designing Experiments"**.

## Abstract  
To investigate the complexity arising from single-cell RNA-sequencing (scRNA-seq) data, researchers increasingly resort to deep generative models, specifically variational autoencoders (VAEs), which are trained by variational inference techniques.  Similar to other dimension reduction approaches, this allows encoding the inherent biological signals of gene expression data, such as pathways or gene programs, into lower-dimensional latent representations. However, the number of cells necessary to adequately uncover such latent representations is often unknown. Therefore, we propose a single-cell variational inference approach for designing experiments (scVIDE) to have sufficient power for detecting cell group structure in a lower-dimensional representation. The approach is based on a test statistic that quantifies the contribution of the data from every single cell to the latent representation. Using a smaller scRNA-seq data set as a starting point, we generate synthetic data sets of various sizes from a fitted VAE. Using a permutation technique for obtaining a null distribution of the test statistic, we subsequently determine statistical power for various numbers of cells, thus guiding experimental design.
We illustrate how scVIDE can be used to determine the statistical power of scRNA-seq studies and also to generate synthetic data for deeper insight, with several data sets from various sequencing protocols. We also consider the setting of transcriptomics studies with large numbers of cells, where scVIDE can be used to determine statistical power for sub-clustering. For this purpose, we use data from the human KPMP Kidney Cell Atlas and evaluate the power for sub-clustering of the immune cells contained therein.

## Main requirements  
Julia: 1.6.0

