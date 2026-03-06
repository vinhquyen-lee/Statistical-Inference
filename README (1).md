# Cascaded Diffusion Model (Cas-DM) Implementation

This repository contains the official PyTorch implementation of the paper **"Cascaded Diffusion Models for High-Fidelity Image Generation"**.

Cas-DM improves standard Diffusion Models (DDPM) by introducing a cascaded architecture that allows the effective integration of perceptual metrics (like LPIPS) during training without destabilizing the noise prediction process.

## 🌟 Key Features
* **Cascaded Architecture:** Utilizes two coupled U-Net modules:
    * **Module 1 ($\theta$):** Standard DDPM predicting noise $\epsilon$.
    * **Module 2 ($\phi$):** Predicts clean image $x_0$ and a mixing weight $r_t$, optimized with perceptual losses.
* Integrate Perceptual Loss to enhance image quality and semantic consistency.
* Implements strict gradient blocking to ensure the metric functions do not degrade the noise prediction baseline.

## 🛠️ Installation

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

## 🚀 Usage

### 1. Data Preparation
Prepare dataset (e.g., CIFAR-10, LSUN, CelebA-HQ) in a folder. The code handles resizing and center-cropping automatically.

### 2. Training Cas-DM and Sampling
Due to limited local GPU resources, the model training and image sampling processes were conducted on Kaggle using the `cas-dm.ipynb` notebook.

### 3. Model Evaluation
The performance of the model is evaluated using two commonly used metrics in generative models:
* **Fréchet Inception Distance (FID):** Measures the similarity between generated images and real images. Lower FID values indicate better performance.

* **Inception Score (IS):** Evaluates both the quality and diversity of the generated images. Higher IS values indicate better performance.

## 📂 Code Structure
* `scripts/casdm_train.py`: Main entry point for training the Cascaded Diffusion Model. Handles the initialization of dual U-Nets and the specific Cas-DM loss computation.
* `improved_diffusion/unet.py`: Defines the U-Net architecture.
* `improved_diffusion/train_util.py`: Utility functions for the training loop, optimizing steps, and logging.
* `improved_diffusion/gaussian_diffusion.py`: Core diffusion process logic (forward/backward steps).

## 📄 Reference
If you find this code useful, please cite our paper:

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

## 🙏 Acknowledgements
This codebase is built upon OpenAI's Improved Diffusion. We thank the authors for their open-source contribution.

---

