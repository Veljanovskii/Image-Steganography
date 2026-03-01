# Image Steganography with Residual Learning

A deep learning approach to hide secret images within cover images using a UNet-based encoder-decoder architecture with residual learning. This implementation prioritizes imperceptibility (high PSNR) while maintaining effective secret image recovery.

## Overview

Image steganography is the art of hiding secret information within carrier media (in this case, images) in a way that is imperceptible to observers. This project implements **StegoNet**, a neural network that learns to:

1. Embed a secret image into a cover image as a stego image (with minimal visible distortion)
2. Extract and recover the secret image from the stego image

The key innovation is using a **residual learning** approach where the network predicts a small perturbation (delta) that is added to the cover image, strongly preserving its quality.

## Key Features

- **High Imperceptibility**: Maintains PSNR > 35dB between stego and cover images by default
- **Effective Recovery**: Learns strong feature representations for accurate secret image extraction
- **Configurable Perturbation**: Adjustable `eps` parameter to balance imperceptibility vs. recovery quality
- **Residual Architecture**: Predicts delta (perturbation) rather than raw stego, enabling better cover image preservation
- **Automatic Plotting**: Generates training curves for loss and PSNR metrics
- **Mixed Precision Training**: Optional CUDA AMP for faster training on NVIDIA GPUs
- **Checkpoint Management**: Easy resume functionality and periodic checkpointing

## Model Architecture

### StegoNet Components

**Encoder (UNetEncoderResidual)**:
- Takes concatenated cover (3-ch) and secret (3-ch) images as input (6 channels total)
- 4-level U-Net encoder-decoder with skip connections
- Outputs a 3-channel residual delta
- Formulation: `stego = clamp(cover + eps * tanh(delta), 0..1)`

**Decoder**:
- Takes stego image as input
- Extracts and reconstructs the secret image using deconvolutions
- Outputs sigmoid-activated image (values in [0,1])

### Key Hyperparameter

- **eps** (epsilon): Controls maximum per-pixel perturbation magnitude
  - `eps=0.03`: Higher PSNR (~35+ dB), harder secret recovery
  - `eps=0.05`: Lower PSNR, easier secret recovery
  - Adjust based on your imperceptibility vs. recovery tradeoff

## Installation

### Requirements

- Python 3.8+
- PyTorch (with CUDA support for GPU training)
- torchvision
- Pillow
- matplotlib
- tqdm

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Image-Steganography
```

2. Install dependencies:
```bash
pip install torch torchvision pillow matplotlib tqdm
```

## Dataset Setup

### CelebA Dataset

Download the CelebA dataset from:
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

After downloading:
1. Extract the dataset
2. Place it in the project directory as: `data/celeba/`

The dataset structure should be:
```
data/
└── celeba/
    ├── img_align_celeba/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── identity_CelebA.txt
```

## Usage

### Training

Basic training with default parameters:
```bash
cd src
python train.py
```

With custom parameters:
```bash
python train.py \
    --data_root ./data \
    --out_root ./outputs \
    --img_size 256 \
    --batch_size 64 \
    --epochs 6 \
    --lr 2e-4 \
    --eps 0.03 \
    --alpha 5.0 \
    --beta 1.0 \
    --save_every 50
```

**Training Parameters**:
- `--data_root`: Path to dataset directory
- `--out_root`: Output directory for checkpoints and logs
- `--img_size`: Image resolution (256x256 recommended)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (Adam optimizer)
- `--base`: Base channels for UNet (32 is default)
- `--eps`: Max perturbation scale (0..1 space)
- `--alpha`: Weight for cover image loss (emphasize imperceptibility)
- `--beta`: Weight for secret recovery loss
- `--use_amp`: Enable mixed precision training (CUDA only)
- `--max_items`: Limit dataset size (useful for quick testing)
- `--save_every`: Save checkpoints every N steps

**Resume Training**:
```bash
python train.py --resume ./outputs/stegonet_xxxxx/checkpoints/step_500.pt
```

### Inference

Embed a secret image into a cover image:
```bash
cd src
python infer.py \
    --ckpt ./outputs/stegonet_xxxxx/checkpoints/step_1500.pt \
    --cover /path/to/cover.jpg \
    --secret /path/to/secret.jpg \
    --out_dir ./outputs/infer
```

**Inference Parameters**:
- `--ckpt`: Path to trained checkpoint (required)
- `--cover`: Path to cover image (required)
- `--secret`: Path to secret image (required)
- `--out_dir`: Output directory for results
- `--img_size`: Image resolution (must match training)

**Output Files**:
- `cover.png`: Input cover image
- `secret.png`: Input secret image
- `stego.png`: Generated stego image containing the secret
- `recovered_secret.png`: Extracted secret from stego image

### Evaluation

Evaluate model on a dataset:
```bash
cd src
python evaluate.py
```

Edit the script to change:
- `ckpt_path`: Path to your checkpoint
- `img_size`: Image dimensions
- `batch_size`: Evaluation batch size
- `max_items`: Number of images to evaluate

The script outputs:
- Mean PSNR between stego and cover images
- Mean PSNR between recovered and original secret images

## Training Output

The training script generates:

```
outputs/stegonet_TIMESTAMP_szSIZE/
├── checkpoints/
│   ├── step_50.pt
│   ├── step_100.pt
│   ├── epoch_1.pt
│   ├── epoch_2.pt
│   └── ...
├── samples/
│   ├── step_50.png
│   ├── step_100.png
│   └── ...
└── train_log.csv
```

And plots:
- `plots_loss.png`: Training loss curves (total, cover, secret)
- `plots_psnr.png`: PSNR metrics (stego vs cover, recovered vs secret)

## Loss Function

The training objective combines two goals:

```
loss_total = alpha * loss_cover + beta * loss_secret
```

Where:
- `loss_cover = MSE(stego, cover)`: Imperceptibility loss
- `loss_secret = MSE(recovered, secret)`: Recovery loss
- `alpha`: Weight emphasizing imperceptibility (default 5.0)
- `beta`: Weight on recovery quality (default 1.0)

Default values prioritize imperceptibility; adjust if better recovery is needed.

## Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- **PSNR(stego, cover)**: Measures imperceptibility. Higher is better (>35dB is good).
- **PSNR(recovered, secret)**: Measures recovery quality. Higher is better.

## Project Structure

```
src/
├── model.py          # StegoNet architecture (encoder + decoder)
├── dataset.py        # CelebA dataset loading and secret generation
├── train.py          # Training script with logging and checkpoint management
├── infer.py          # Inference script for embedding and recovery
├── evaluate.py       # Evaluation script for metrics computation
├── metrics.py        # PSNR and other metric calculations
└── utils.py          # Utility functions (plotting, image saving, etc.)
```

## Performance Notes

- **GPU Training**: Recommended for faster convergence. Use `--use_amp` flag for mixed precision on CUDA.
- **Image Size**: 256x256 is a good balance. Smaller images train faster but may lose quality.
- **Batch Size**: 64 with 256x256 images fits on most 8GB+ GPUs.
- **Typical Training Time**: ~2-3 hours for 6 epochs on a modern GPU with 30K images.

## License

This project is provided as-is for educational and research purposes.
