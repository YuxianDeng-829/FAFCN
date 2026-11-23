# FAFCN

MATLAB implementation of Fast Anchor-based Fuzzy Clustering with
Nuclear-norm Regularization.

## Quick Start

Open the repository folder in MATLAB and run `DEMO`.
The script runs FAFCN once and prints ACC, NMI, ARI, and F1.

To use another dataset, change `dataset_file` in `DEMO.m`. The MAT file
must contain `X`, with one sample per row, and `Y`, with one reference
label per sample. The demo uses the reference labels to determine the
number of clusters and evaluate the resulting partition.

## Citation

Jianyong Zhu, Yuxian Deng, Hui Yang, and Feiping Nie.
“Fast Anchor-based Fuzzy Clustering With Nuclear-norm Regularization.”
IEEE Transactions on Fuzzy Systems, 2026.
https://doi.org/10.1109/TFUZZ.2026.3752833
