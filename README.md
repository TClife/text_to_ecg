# Text-to-ECG: Autoregressive ECG Generation from Clinical Text Reports

This repository contains the official implementation of **Text-to-ECG**, a framework for generating 12-lead ECG signals from clinical text reports. The model architecture consists of three main components:

1. **VQ-VAE**: Vector Quantized Variational Autoencoder for encoding continuous ECG signals into discrete token sequences
2. **Autoregressive Transformer**: A transformer-based model that generates ECG tokens conditioned on text reports (and optionally age/gender)
3. **HiFi-GAN Decoder**: A neural vocoder that converts discrete ECG tokens back into high-fidelity raw ECG waveforms

## Architecture Overview

```
                                    Training Pipeline
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   ┌──────────┐    ┌─────────────┐    ┌──────────────────────────────┐   │
    │   │ ECG Data │───▶│   VQ-VAE    │───▶│ Discrete ECG Token Sequence  │   │
    │   │ (12-lead)│    │  (Encoder)  │    │        (e.g., 312 tokens)    │   │
    │   └──────────┘    └─────────────┘    └──────────────────────────────┘   │
    │                                                      │                   │
    │   ┌────────────────────┐                             │                   │
    │   │  Text Report       │                             ▼                   │
    │   │  (BPE Tokenized)   │──────────▶┌────────────────────────────────┐   │
    │   │  + Age/Gender      │           │   Autoregressive Transformer   │   │
    │   └────────────────────┘           │   (Text + ECG Token Prediction)│   │
    │                                    └────────────────────────────────┘   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

                                   Inference Pipeline
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   ┌────────────────────┐    ┌─────────────────────┐                     │
    │   │  Text Report       │───▶│    Autoregressive   │                     │
    │   │  + Age/Gender      │    │     Transformer     │                     │
    │   └────────────────────┘    └─────────────────────┘                     │
    │                                       │                                 │
    │                                       ▼                                 │
    │                          ┌───────────────────────┐                      │
    │                          │  Generated ECG Tokens │                      │
    │                          └───────────────────────┘                      │
    │                                       │                                 │
    │                    ┌──────────────────┴──────────────────┐              │
    │                    ▼                                     ▼              │
    │          ┌─────────────────┐                   ┌─────────────────┐      │
    │          │  VQ-VAE Decoder │                   │  HiFi-GAN       │      │
    │          │  (Fast, Lower   │                   │  (High-Fidelity │      │
    │          │   Quality)      │                   │   Waveforms)    │      │
    │          └─────────────────┘                   └─────────────────┘      │
    │                    │                                     │              │
    │                    └──────────────────┬──────────────────┘              │
    │                                       ▼                                 │
    │                          ┌───────────────────────┐                      │
    │                          │  Generated 12-lead   │                      │
    │                          │     ECG Signal       │                      │
    │                          └───────────────────────┘                      │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

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

## Quick Start

### 0. Prepare Text Tokenizer (Optional)

A sample BPE tokenizer is provided in `configs/tokenizer.json`. For best results, train a custom tokenizer on your ECG report corpus:

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

First, train the VQ-VAE to learn discrete ECG representations:

```bash
python scripts/train_vae.py \
    --data_path /path/to/ecg/data.pt \
    --num_tokens 1024 \
    --num_layers 4 \
    --epochs 100 \
    --batch_size 256 \
    --output_path ./checkpoints/vae.pt
```

### 2. Train Autoregressive Transformer (Text-to-ECG)

Train the autoregressive transformer for text-to-ECG generation:

```bash
python scripts/train_dalle.py \
    --vae_path ./checkpoints/vae.pt \
    --data_path /path/to/ecg/dataset.pt \
    --bpe_path ./configs/tokenizer.json \
    --epochs 500 \
    --batch_size 128 \
    --output_path ./checkpoints/dalle.pt
```

### 3. Train HiFi-GAN Decoder (Optional, for higher quality)

First, extract VQ-VAE codes from the ECG data:

```bash
python -c "
import torch
from text_to_ecg.models.vae import DiscreteVAE

# Load trained VAE
checkpoint = torch.load('./checkpoints/vae.pt')
vae = DiscreteVAE(**checkpoint['hparams'])
vae.load_state_dict(checkpoint['weights'])
vae.eval()

# Load ECG data and extract codes
ecg_data = torch.load('/path/to/ecg/data.pt')
with torch.no_grad():
    codes = vae.get_codebook_indices(ecg_data)
torch.save(codes, './checkpoints/vae_codes.pt')
"
```

Then train the HiFi-GAN decoder:

```bash
python scripts/train_hifi_gan.py \
    --config ./configs/hifi_config.json \
    --data_path /path/to/ecg/data.pt \
    --codes_path ./checkpoints/vae_codes.pt \
    --checkpoint_path ./checkpoints/hifi_gan/
```

### 4. Generate ECG from Text

Generate ECG signals from clinical text reports:

```bash
python scripts/generate.py \
    --dalle_path ./checkpoints/dalle.pt \
    --bpe_path ./configs/tokenizer.json \
    --text "sinus rhythm with left bundle branch block" \
    --age 65 \
    --gender 1 \
    --decoder hifi \
    --output_path ./outputs/generated_ecg.pt
```

## Model Components

### VQ-VAE (`text_to_ecg/models/vae.py`)

The VQ-VAE encodes 12-lead ECG signals (5000 samples at 500Hz = 10 seconds) into discrete token sequences:
- **Encoder**: Convolutional layers that downsample ECG to latent representations
- **Codebook**: Learnable embedding dictionary with `num_tokens` entries
- **Decoder**: Transposed convolutions to reconstruct ECG from quantized embeddings

Key hyperparameters:
- `num_tokens`: Size of codebook (default: 1024)
- `num_layers`: Number of encoder/decoder layers (default: 4)
- `codebook_dim`: Dimension of codebook embeddings (default: 512)

### Autoregressive Transformer (`text_to_ecg/models/dalle.py`)

The transformer generates ECG tokens autoregressively conditioned on:
- **Text tokens**: BPE-tokenized clinical report
- **Age**: Patient age (optional)
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
    'data': torch.Tensor,        # ECG signals [N, 12, 5000]
    'report_data': List[str],    # Text reports
    'age': torch.Tensor,         # Patient ages [N]
    'gender': torch.Tensor       # Patient genders [N] (0: female, 1: male)
}
```

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
│   │   ├── vae.py              # VQ-VAE model
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
    ├── train_vae.py           # VQ-VAE training
    ├── train_dalle.py         # DALLE training
    ├── train_hifi_gan.py      # HiFi-GAN training
    ├── generate.py            # ECG generation
    └── test_components.py     # Component verification tests
```

## Testing

To verify all components work correctly, run the test script:

```bash
python scripts/test_components.py
```

This will test:
- VQ-VAE encoding and decoding
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
