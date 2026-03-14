# MY-MICCAI-PAPER-REPO
FETALFusion — MICCAI 2026 Submission

This repository contains the full training and evaluation code for FETALFusion.

Reproducibility: Running fetalfusion_FINAL_SUBMISSION.ipynb end-to-end reproduces the results reported in Table 1–4 of the paper. All baselines (U-Net, Att. U-Net, nnU-Net v2) are implemented within the same notebook under identical training conditions: AdamW lr=1e-4, 50 epochs, early stopping patience=10, identical augmentation, same train/val split (seed=42).

Note on U-Mamba and VM-Net UNet: Due to Kaggle session time constraints, they were evaluated in a separate notebook with strictly identical hyperparameters, optimizer, augmentation pipeline, and data splits. The architecture implementation follows the original paper (8 transformer layers, 768-d, 12 heads).

Hardware: All experiments run on a single NVIDIA T4 (16GB), consistent with the paper's computational efficiency claims.

Datasets: HC-18 and PSFHS are publicly available. Download links and preprocessing steps are described in Section 3.1 of the paper.
