RSNA Intracranial Aneurysm Detection

This repository contains code and experiments for the RSNA Intracranial Aneurysm Detection
 Kaggle competition.
The goal is to build ML models that can detect and localize brain aneurysms from multi-modal DICOM imaging (CTA, MRA, MRI), across institutions and protocols.

📂 Repo Structure
.
├── notebooks/
│   ├── 00_eda.ipynb          # Exploratory analysis of dataset
│   └── 10_train_25d.ipynb    # Training notebook (2.5D baseline driver)
├── src/
│   ├── baseline_25D.py       # ResNet-based 2.5D MIL baseline model
│   ├── config.py             # Dataset/config paths
│   ├── dataloader.py         # DICOM/NIfTI loading and preprocessing
│   ├── partition.py          # Data splitting + loader builders
│   ├── slice_bag.py          # Slice bagging logic for 2.5D inputs
│   ├── utils.py              # Seeds, helpers, batch unpacking, etc.
│   ├── metrics.py            # AUC metrics (per-label, weighted)
│   ├── train_eval.py         # Forward, train loop, validation routines
│   ├── class_weights.py      # Compute BCE pos_weight from labels
│   └── __init__.py
├── requirements.txt          # Dependencies for Kaggle (no torch pins)
├── pyproject.toml            # Local environment (uv/poetry style)
└── README.md

🧠 Approach

2.5D Multiple Instance Learning (MIL):
Each DICOM series is split into overlapping bags of slices (K consecutive slices).
A ResNet backbone encodes each slice, and bag features are pooled to predict aneurysm presence across 14 target labels.

Multi-label Classification:

13 location-specific labels (e.g., MCA, ACA, ICA, Basilar Tip, etc.)

1 global target: Aneurysm Present (weighted ×13 in the competition metric).

Loss: BCEWithLogitsLoss with per-class pos_weight to handle severe class imbalance.

Evaluation Metric: Weighted AUC (Kaggle official metric).
Our helper computes 13×1 + AP×13 → averaged.
