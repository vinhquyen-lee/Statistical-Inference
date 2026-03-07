# Cascaded Diffusion Model (Cas-DM) Implementation

This is the code based on the paper [**Cascaded Diffusion Models for High-Fidelity Image Generation**](https://arxiv.org/pdf/2106.15282).

The goal of this research is to improve standard Diffusion Models (DDPM) to generate higher-quality images, overcoming the instability of the noise prediction process when integrating perceptual metrics (like LPIPS).

## Key Features
* **Cascaded Architecture:** Utilizes two coupled U-Net modules:
    * **Module 1 ($\theta$):** Standard DDPM predicting noise $\epsilon$.
    * **Module 2 ($\phi$):** Predicts clean image $x_0$ and a mixing weight $r_t$, optimized with perceptual losses.
* Integrate Perceptual Loss to enhance image quality and semantic consistency.
* Implements strict gradient blocking to ensure the metric functions do not degrade the noise prediction baseline.

## Installation

The code relies on the OpenAI `improved-diffusion` codebase structure.

1. **Clone the repository:**
```bash
git clone https://github.com/vinhquyen-lee/Cas-DM.git
cd Cas-DM
```

2. **Install dependencies:**
```bash
pip install -e .
```

## Usage

### 1. Data Preparation
Prepare dataset (e.g., CIFAR-10, LSUN, CelebA-HQ) in a folder. The code handles resizing and center-cropping automatically.

### 2. Training Cas-DM and Sampling
Due to limited local GPU resources, the model training and image sampling processes were conducted on Kaggle using the `cas-dm.ipynb` notebook.

### 3. Model Evaluation
The performance of the model is evaluated using two commonly used metrics in generative models:
* **Fréchet Inception Distance (FID):** Measures the similarity between generated images and real images. Lower FID values indicate better performance.

* **Inception Score (IS):** Evaluates both the quality and diversity of the generated images. Higher IS values indicate better performance.

## Reference
If you find this repository useful, please cite the original paper:

```bibtex
@misc{ho2021cascadeddiffusionmodelshigh,
      title={Cascaded Diffusion Models for High Fidelity Image Generation}, 
      author={Jonathan Ho and Chitwan Saharia and William Chan and David J. Fleet and Mohammad Norouzi and Tim Salimans},
      year={2021},
      eprint={2106.15282},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2106.15282}, 
}
```
## Implementation

This repository provides an unofficial implementation of the Cascaded Diffusion Model (Cas-DM).

Implementation by **Lê Quang Vĩnh Quyền** (@vinhquyen-lee) based on the original paper.

## Acknowledgements
This codebase is built upon OpenAI's Improved Diffusion. We thank the authors for their open-source contribution.

---

