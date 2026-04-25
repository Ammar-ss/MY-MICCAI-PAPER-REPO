# FETALFusion

> Resolution-aware state-space modelling for multi-domain fetal ultrasound segmentation. Code for a paper currently under double-blind review at MICCAI 2026.

---

## What this is

Fetal ultrasound segmentation models have a persistent deployment problem: a model trained on one acquisition protocol or dataset tends to fail when pointed at another. Different scanners, different resolutions, different anatomical coverage. The standard fix is to retrain or manually reconfigure for each new domain, which is expensive and often not feasible in clinical settings.

FETALFusion addresses this with a dual-path CNN-Mamba encoder that handles multi-resolution inputs natively, a resolution-aware state-space scanning strategy that adapts its directional scan pattern based on feature map scale, and a Conditional Domain Feature Routing (CDFR) bottleneck that learns domain-specific channel gates during training and averages them at inference -- meaning no domain label is needed at deployment.

The result is a single trained model that generalises across HC-18 and PSFHS acquisition protocols without retraining or reconfiguration.

---

## Results

Evaluated on 472 validation images across both domains (HC-18 + PSFHS joint training).

| Metric | FETALFusion | Best Baseline | Improvement |
|---|---|---|---|
| DSC | 0.9635 | 0.9501 (VM-UNet) | +0.0134 |
| HD95 (mm) | 1.07 | 2.31 (VM-UNet) | -1.24 mm |
| HD95 Std | 3.2x lower than VM-UNet | | |
| ICC | 0.9898 | | |
| Pearson r | 0.992 | | |
| Mean bias | -1.60 mm | | |

All comparisons against 6 baselines, p < 0.001 (Wilcoxon signed-rank test).

Baselines included: U-Net, Attention U-Net, nnU-Net v2, U-Mamba, VM-UNet, and one additional domain-generalisation baseline.

---

## Key Design Decisions

**Resolution-aware SSM scanning**
Most Mamba-based segmentation models apply the same 4-directional scan at every feature map scale. FETALFusion uses 4-directional scanning at full and half resolution (128x128 and 64x64) but drops to 1-directional at quarter and eighth resolution (32x32 and 16x16), halving the SSM compute overhead at lower scales where spatial detail matters less.

**CDFR bottleneck**
A set of K=2 domain-specific channel gate vectors is learned during joint training. At inference, the gates are averaged -- no domain oracle, no domain labels, no test-time adaptation. The same model runs on HC-18 images and PSFHS images identically.

**Gate-averaging at inference**
This is the same principle used in the CDFR design: protocol-specific routing during training collapses to a single unified representation at deployment. It is directly analogous to how FETALFusion-v2 (in preparation) will handle the multi-biometry landmark task.

---

## Reproducibility

Running `FETAL FUSION.ipynb` end-to-end reproduces Tables 1-4 from the paper. All baselines (U-Net, Attention U-Net, nnU-Net v2) are implemented within the same notebook under identical conditions:

- Optimizer: AdamW, lr = 1e-4
- Epochs: 50, early stopping patience = 10
- Augmentation: identical pipeline across all models
- Train/val split: seed = 42

**Note on U-Mamba and VM-UNet:** Due to Kaggle session time constraints, these two baselines were evaluated in `v-and-u-mamba.ipynb` with strictly identical hyperparameters, optimizer, augmentation pipeline, and data splits. The U-Mamba implementation follows the original paper (8 transformer layers, 768-d, 12 heads).

**Hardware:** All experiments run on a single NVIDIA T4 (16 GB VRAM), consistent with the computational efficiency claims in the paper.

---

## Datasets

Both datasets are publicly available.

| Dataset | Task | Images | Access |
|---|---|---|---|
| HC-18 | Fetal head circumference | 999 training + 335 test | [grand-challenge.org/HC18](https://grand-challenge.org/HC18) |
| PSFHS | Pubic symphysis and fetal head segmentation | Public | [grand-challenge.org/PSFHS](https://grand-challenge.org/PSFHS) |

Preprocessing steps follow Section 3.1 of the paper. Briefly: images are resized to 256x256, normalised to [0, 1], and augmented with random horizontal flips, rotation (+/-15 degrees), and Gaussian noise during training only.

---

## Repo Contents

```
FETALFusion/
├── FETAL FUSION.ipynb       # Main training + evaluation notebook
│                             # FETALFusion + U-Net + Att.U-Net + nnU-Net v2
│                             # Reproduces Tables 1-4
│
├── v-and-u-mamba.ipynb      # U-Mamba and VM-UNet baselines
│                             # Identical hyperparameters and splits
│
└── README.md
```

---

## How to Run

All notebooks are designed to run on Kaggle with a T4 GPU (free tier).

**1. Get the datasets**

Download HC-18 from [grand-challenge.org/HC18](https://grand-challenge.org/HC18) and PSFHS from [grand-challenge.org/PSFHS](https://grand-challenge.org/PSFHS). Upload both to your Kaggle dataset storage and update the path variables in Cell 2 of the notebook.

**2. Install dependencies**

```bash
pip install torch torchvision segmentation-models-pytorch \
    monai einops causal-conv1d mamba-ssm \
    numpy pandas matplotlib scikit-learn tqdm
```

Note: `mamba-ssm` requires a CUDA-capable GPU. It will not install on CPU-only environments.

**3. Run**

Open `FETAL FUSION.ipynb` on Kaggle and run all cells. Training takes approximately 2-3 hours per model on a T4. Checkpoints are saved automatically.

For U-Mamba and VM-UNet baselines, open `v-and-u-mamba.ipynb` separately.

---

## Code Release Note

This repository contains the full training and evaluation code. The paper is currently under double-blind review at MICCAI 2026. Author information and full paper details will be added here upon acceptance.

If you have questions about the code or methodology, feel free to open an issue.

---

## Citation

Citation details will be added upon paper acceptance. If you use this code or build on this work in the meantime, please reference this repository directly.

---

## Related Work

This is part of a broader line of work on fetal ultrasound analysis:

- **FETALFusion-v2** (in preparation, Medical Image Analysis) -- extends the CDFR bottleneck to simultaneous segmentation and landmark heatmap regression for automated fetal biometry
- **Attention-ResUNet** (ANTIC 2025, Best Paper) -- prior published work on fetal head segmentation using multi-scale attention gates
