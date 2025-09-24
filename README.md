# Disentangled Latent Space for 3D Human Body Models with CVAE and OT
This repository contains the official implementation of the UCAMi 2025 paper **“Learning Disentangled Latent Space for 3D Human Body Models with Conditional Variational Autoencoders and Optimal Transport.”** We integrate a Conditional VAE with Optimal Transport regularization to impose geometric structure on the latent space learned from STAR-based body meshes. The model yields a smooth, compact, and disentangled representation conditioned on anthropometric variables (e.g., weight, height, sex), enabling high-fidelity reconstruction, controlled shape generation and smooth latent interpolations across multi-session scans collected during nutritional treatments.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)


## 👀 Overview

![Methodology Overview](Images/Overview.png)

- **Encoder (graph-based):** Processes the 3D mesh and fuses an external conditioning vector (e.g., height, fat, gender) to output posterior Gaussian parameters.  
- **Partitioned latent space:** Latent vector **z** is split into three disjoint subspaces, nudging each partition to encode a specific attribute and promoting disentanglement.  
- **Decoder (MLP):** Reconstructs the mesh from **[z || y]** (latent + conditions).  
- **Loss:**  
  - Reconstruction (Chamfer Distance)  
  - Latent regularization (KL or MMD)  
  - **Sinkhorn divergence (OT)** for geometric consistency  
  - Attribute consistency (MSE/BCE


## 🎯 Key Features
- **CVAE + OT:** Conditional VAE trained with a Wasserstein/OT objective on the latent space.
- **Disentanglement:** Encourages factorized latent factors aligned with anthropometric attributes.
- **STAR compatibility:** Works with STAR parametric human body models.
- **Smooth interpolations:** OT regularization yields consistent latent traversals.
- **Reconstruction & generation:** Reconstruct meshes and sample new shapes under conditions (e.g., weight/height/gender).

## 🗂️ Repository Structure

```
multimodal-contrastive-learning-for-clinical-data-alignment-via-fat-composition-representations/
├── README.md                          # This file
├── LICENSE                           # MIT License
├── .gitignore                       # Git ignore rules
├── index.html                       # Main project website
├── project_site/                    # Academic project page
│   ├── index.html                  # Project presentation page
│   └── static/                     # Web assets (CSS, JS, images, etc.)
├── src/                            # Source code
│   ├── code/
│   │   ├── clip_tab.py            # Main CLIP-TAB implementation
│   │   └── methods_comparison.py  # Comparison with baseline methods
│   ├── embeddings/                # Generated embeddings storage
│   │   └── exp10.1/               # Experiment results
│   ├── figures/                   # Generated figures and plots
│   └── models_weights/            # Trained model weights
│       └── exp10.1/
└── static/                        # Static assets (duplicate for GitHub Pages)
    ├── css/                       # Stylesheets
    ├── images/                    # Research images and visualizations
    ├── js/                        # JavaScript files
    ├── pdfs/                      # Papers and documentation
    └── videos/                    # Demo videos
```

## 📈 Results (Global Reconstruction Metrics)
| Model       | Vertex Error ↓ | Chamfer Dist. ↓ | Wasserstein Dist. ↓ |
|-------------|----------------:|----------------:|--------------------:|
| CVAE        | 43.1            | 1.7             | 1.5                 |
| **CVAE + OT** | **25.9**        | **0.6**         | **0.5**             |
| CWAE        | 32.7            | 1.1             | 0.9                 |
| **CWAE + OT** | **21.2**        | **0.5**         | **0.3**             |

## Visualization

| <img src="Images/generated1.PNG" width="320"> | <img src="Images/generated2.PNG" width="130"> | <img src="Images/Latent.png" width="420"> |
|---|---|---|
| Generated Model — Front View | Generated Model — Side View | Generated Sample & Closest Match in Latent Space |



## 🔗 Citation

If you use this code in your research, please cite:

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

This work was funded by the Spanish State Research Agency (AEI) through grants **PID2023-149562OB-I00** and **PID2023-152804OB-I00**, awarded by **MCIN/AEI/10.13039/501100011033**, and financed by the Government of the Valencian Community (Generalitat Valenciana).

Project context: [Tech4Diet](https://github.com/Tech4DLab).
