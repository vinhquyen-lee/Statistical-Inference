# Cascaded Diffusion Model (Cas-DM) Implementation

This repository contains the official PyTorch implementation of the paper **"Cascaded Diffusion Models for High-Fidelity Image Generation"**.

Cas-DM improves standard Diffusion Models (DDPM) by introducing a cascaded architecture that allows the effective integration of perceptual metrics (like LPIPS) during training without destabilizing the noise prediction process.

## 🌟 Key Features
* **Cascaded Architecture:** Utilizes two coupled U-Net modules:
    * **Module 1 ($\theta$):** Standard DDPM predicting noise $\epsilon$.
    * **Module 2 ($\phi$):** Predicts clean image $x_0$ and a mixing weight $r_t$, optimized with perceptual losses.
* **Perceptual Loss Integration:** Supports LPIPS loss to enhance image quality and semantic consistency.
* **Gradient Isolation:** Implements strict gradient blocking to ensure the metric functions do not degrade the noise prediction baseline.

## 🛠️ Installation

The code relies on the OpenAI `improved-diffusion` codebase structure.

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/cas-dm.git
cd cas-dm
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

*Note: Ensure you have `torch`, `torchvision`, and `piq` (or `lpips`) installed for the perceptual loss functions.*

## 🚀 Usage

### 1. Data Preparation
Prepare your dataset (e.g., CIFAR-10, LSUN, CelebA-HQ) in a folder. The code handles resizing and center-cropping automatically.

### 2. Training Cas-DM
To train the model, use the `casdm_train.py` script. This script initializes both the base noise predictor ($\theta$) and the cascaded refiner ($\phi$).

**Example Training Command:**
```bash
python scripts/casdm_train.py --data_dir /path/to/your/dataset --image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --diffusion_steps 4000 --noise_schedule cosine --lr 1e-4 --batch_size 128 --use_lpips True
```

**Key Arguments:**
* `--use_lpips`: Set to True to enable LPIPS loss for the second module ($\phi$).
* `--learn_sigma`: Recommended True for improved log-likelihood and sample quality.
* `--diffusion_steps`: Total diffusion steps (default: 4000).

### 3. Sampling
Generate images using the trained checkpoints. The sampling process utilizes the learned mixing weight $r_t$ to combine predictions from both modules.

```bash
python scripts/image_sample.py --model_path /path/to/model_phi.pt --num_samples 1000 --batch_size 16 --image_size 64
```

## 📂 Code Structure
* `scripts/casdm_train.py`: Main entry point for training the Cascaded Diffusion Model. Handles the initialization of dual U-Nets and the specific Cas-DM loss computation.
* `improved_diffusion/unet.py`: Defines the U-Net architecture.
* `improved_diffusion/train_util.py`: Utility functions for the training loop, optimizing steps, and logging.
* `improved_diffusion/gaussian_diffusion.py`: Core diffusion process logic (forward/backward steps).

## 📄 Reference
If you find this code useful, please cite our paper:

```bibtex
@article{casdm2024,
  title={Cascaded Diffusion Models for High-Fidelity Image Generation},
  author={Your Name and Co-authors},
  journal={arXiv preprint},
  year={2024}
}
```

## 🙏 Acknowledgements
This codebase is built upon OpenAI's Improved Diffusion. We thank the authors for their open-source contribution.

---

### Các điểm cần lưu ý khi bạn sử dụng file này:
1. **Phần "Installation":** README giả định bạn dùng thư viện `piq` hoặc `lpips`. Nếu bạn dùng thư viện khác, hãy chỉnh lại cho phù hợp.
2. **Đường dẫn script:** Nếu file `casdm_train.py` nằm ở thư mục gốc, hãy bỏ `scripts/` trong lệnh chạy.
3. **Argument `--use_lpips`:** Hãy kiểm tra lại trong `casdm_train.py` tên chính xác của argument kích hoạt LPIPS (ví dụ: `--use_metric_loss`, `--lambda_lpips`, v.v.) để chỉnh lại cho khớp.
