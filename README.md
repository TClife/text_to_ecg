<div align="center">

<h1> Text-to-ECG: 12-Lead Electrocardiogram Synthesis conditioned on Clinical Text Reports </h1>

<h5 align="center"> 

<a href='https://arxiv.org/abs/2303.09395'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

<br>

[Hyunseung Chung](https://sites.google.com/view/thschung)<sup>1</sup>,
[Jiho Kim](https://jiho283.github.io/)<sup>1</sup>,
[Joon-myoung Kwon](https://scholar.google.com/citations?user=DMd-2NEAAAAJ&hl=ko)<sup>2,4</sup>,
[Ki-Hyun Jeon](https://scholar.google.com/citations?user=4JSnJkQAAAAJ&hl=ko)<sup>3</sup>,
[Min Sung Lee](https://scholar.google.com/citations?user=726DHOwAAAAJ&hl=ko)<sup>2</sup>,
[Edward Choi](https://mp2893.com/)<sup>1</sup>

<sup>1</sup>KAIST <sup>2</sup>Ajou University School of Medicine

<p align="center">
    <img src="figures/figure_auto_tte.png" width="95%">
</p>

</h5>
</div>

## Components

1. **VQ-VAE**: Vector Quantized Variational Autoencoder for encoding continuous ECG signals into discrete token sequences
2. **Autoregressive Transformer**: A transformer-based model that generates ECG tokens conditioned on text reports (and optionally age/gender)
3. **HiFi-GAN Decoder**: A neural vocoder that converts discrete ECG tokens back into high-fidelity raw ECG waveforms


## Installation

```bash
git clone https://github.com/TClife/text_to_ecg.git
cd text_to_ecg
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (for GPU training)

Install dependencies:
```bash
pip install -r requirements.txt
```

For PTB-XL data preparation, also install:
```bash
pip install wfdb pandas
```

## Quick Start

### 0. Prepare Data

For PTB-XL dataset preparation (downloads not included):

```bash
python scripts/prepare_ptbxl.py \
    --ptbxl_path /path/to/ptb-xl/ \
    --output_dir ./data/
```

This creates train/valid/test `.pt` files and trains a BPE tokenizer. See `scripts/prepare_ptbxl.py` for details.

A sample BPE tokenizer config is provided in `configs/tokenizer.json`. For custom datasets, train a tokenizer on your ECG report corpus:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
tokenizer.pre_tokenizer = Whitespace()

# Train on your ECG report texts
trainer = BpeTrainer(vocab_size=3000, special_tokens=['[PAD]', '[UNK]'])
tokenizer.train_from_iterator(your_report_texts, trainer=trainer)
tokenizer.save('./configs/tokenizer.json')
```

### 1. Train VQ-VAE (ECG Tokenizer)

Train the VQ-VAE to learn discrete ECG representations. The default model type is `vqvae` (vector quantization with EMA codebook updates), which is recommended over `discrete` (Gumbel-softmax) to avoid codebook collapse:

```bash
python scripts/train_vae.py \
    --data_path ./data/ptbxl_ecg_train.pt \
    --model_type vqvae \
    --num_tokens 1024 \
    --num_layers 4 \
    --vq_decay 0.8 \
    --epochs 100 \
    --batch_size 256 \
    --output_path ./checkpoints/vae.pt
```

### 2. Extract VQ-VAE Codes (for HiFi-GAN)

Extract discrete codes from all training ECGs:

```bash
python scripts/extract_codes.py \
    --vae_path ./checkpoints/vae.pt \
    --data_path ./data/ptbxl_ecg_train.pt \
    --output_path ./data/vae_codes_train.pt
```

### 3. Train Autoregressive Transformer (Text-to-ECG)

Train the autoregressive transformer for text-to-ECG generation:

```bash
python scripts/train_dalle.py \
    --vae_path ./checkpoints/vae.pt \
    --data_path ./data/ptbxl_train.pt \
    --bpe_path ./configs/tokenizer.json \
    --dim 512 --depth 12 --heads 8 --dim_head 64 \
    --epochs 200 \
    --batch_size 64 \
    --output_path ./checkpoints/dalle.pt
```

### 4. Train HiFi-GAN Decoder (Optional, for higher quality)

```bash
python scripts/train_hifi_gan.py \
    --config ./configs/hifi_config.json \
    --data_path ./data/ptbxl_ecg_train.pt \
    --codes_path ./data/vae_codes_train.pt \
    --checkpoint_path ./checkpoints/hifi_gan/
```

### 5. Generate ECG from Text

Generate ECG signals from clinical text reports:

```bash
# Using VQ-VAE decoder (default):
python scripts/generate.py \
    --dalle_path ./checkpoints/dalle.pt \
    --bpe_path ./configs/tokenizer.json \
    --text "sinus rhythm normal ecg" \
    --age 55 --gender 1 \
    --visualize \
    --output_path ./outputs/generated_ecg.pt

# Using HiFi-GAN decoder (higher quality):
python scripts/generate.py \
    --dalle_path ./checkpoints/dalle.pt \
    --bpe_path ./configs/tokenizer.json \
    --hifi_path ./checkpoints/hifi_gan/g_00039000 \
    --hifi_config ./configs/hifi_config.json \
    --decoder hifi \
    --text "sinus rhythm normal ecg" \
    --age 55 --gender 1 \
    --visualize \
    --output_path ./outputs/generated_ecg.pt
```

### 6. Compare Decoders (Optional)

Compare ground truth, VQ-VAE decoded, and HiFi-GAN decoded ECGs side by side:

```bash
python scripts/compare_decoders.py \
    --dalle_path ./checkpoints/dalle_best.pt \
    --hifi_path ./checkpoints/hifi_gan/g_00039000 \
    --data_path ./data/ptbxl_train.pt \
    --text "sinus rhythm normal ecg"
```

## Model Components

### VQ-VAE (`text_to_ecg/models/vae.py`)

The VQ-VAE encodes 12-lead ECG signals (5000 samples at 500Hz = 10 seconds) into discrete token sequences:
- **Encoder**: Convolutional layers that downsample ECG to latent representations
- **Codebook**: Learnable embedding dictionary with `num_tokens` entries
- **Decoder**: Transposed convolutions to reconstruct ECG from quantized embeddings

Two implementations are provided:
- **VQVAE** (recommended): Vector quantization with EMA codebook updates, data-based initialization, and unused entry reset. Use `vq_decay=0.8` to avoid codebook collapse.
- **DiscreteVAE**: Gumbel-softmax based. Prone to codebook collapse with default settings.

Key hyperparameters:
- `num_tokens`: Size of codebook (default: 1024)
- `num_layers`: Number of encoder/decoder layers (default: 4)
- `codebook_dim`: Dimension of codebook embeddings (default: 512)

### Autoregressive Transformer (`text_to_ecg/models/dalle.py`)

The transformer generates ECG tokens autoregressively conditioned on:
- **Text tokens**: BPE-tokenized clinical report
- **Age**: Patient age (optional, discretized into 10-year bins)
- **Gender**: Patient gender (optional)

Key hyperparameters:
- `dim`: Model dimension (default: 512)
- `depth`: Number of transformer layers (default: 12)
- `heads`: Number of attention heads (default: 8)
- `text_seq_len`: Maximum text sequence length (default: 104)

### HiFi-GAN Decoder (`text_to_ecg/models/hifi_gan.py`)

The HiFi-GAN converts discrete ECG tokens to high-fidelity waveforms:
- **Generator**: Upsampling network with multi-receptive field fusion
- **Multi-Period Discriminator**: Captures periodic patterns in ECG
- **Multi-Scale Discriminator**: Captures multi-scale features

## Dataset Format

The expected dataset format is a PyTorch dictionary containing:

```python
{
    'data': list[torch.Tensor],  # ECG signals, each [12, 5000]
    'report_data': list[str],    # Text reports
    'age': torch.Tensor,         # Patient ages [N]
    'gender': torch.Tensor       # Patient genders [N] (0: female, 1: male)
}
```

For VQ-VAE training, a simpler format (list of `[12, 5000]` tensors) is also supported.

## Project Structure

```
text-to-ecg/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── tokenizer.json         # BPE tokenizer (sample provided)
│   ├── vae_config.yaml        # VQ-VAE hyperparameters
│   ├── dalle_config.yaml      # DALLE hyperparameters
│   └── hifi_config.json       # HiFi-GAN hyperparameters
├── text_to_ecg/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vae.py              # VQ-VAE / VQVAE models
│   │   ├── dalle.py            # DALLE autoregressive model
│   │   ├── transformer.py      # Transformer backbone
│   │   ├── attention.py        # Attention mechanisms
│   │   ├── hifi_gan.py         # HiFi-GAN generator
│   │   └── discriminators.py   # GAN discriminators
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset classes
│   │   └── tokenizer.py        # Text tokenizers
│   └── utils/
│       ├── __init__.py
│       ├── distributed.py      # Distributed training
│       ├── helpers.py          # Helper functions
│       └── reversible.py       # Reversible layers
└── scripts/
    ├── prepare_ptbxl.py       # PTB-XL data preprocessing
    ├── train_vae.py           # VQ-VAE training
    ├── extract_codes.py       # Extract VQ-VAE codes for HiFi-GAN
    ├── train_dalle.py         # DALLE training
    ├── train_hifi_gan.py      # HiFi-GAN training
    ├── generate.py            # ECG generation
    ├── compare_decoders.py    # Compare VQ-VAE vs HiFi-GAN output
    └── test_components.py     # Component verification tests
```

## Testing

To verify all components work correctly, run the test script:

```bash
python scripts/test_components.py
```

This will test:
- VQ-VAE encoding and decoding (both DiscreteVAE and VQVAE)
- DALLE training loss computation and generation
- HiFi-GAN generator and discriminators
- End-to-end pipeline

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{chung2023text,
  title={Text-to-ecg: 12-lead electrocardiogram synthesis conditioned on clinical text reports},
  author={Chung, Hyunseung and Kim, Jiho and Kwon, Joon-Myoung and Jeon, Ki-Hyun and Lee, Min Sung and Choi, Edward},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation builds upon:
- [DALL-E PyTorch](https://github.com/lucidrains/DALLE-pytorch) by lucidrains
- [HiFi-GAN](https://github.com/jik876/hifi-gan) for neural vocoder
- [Vector Quantize PyTorch](https://github.com/lucidrains/vector-quantize-pytorch)
