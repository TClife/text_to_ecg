#!/usr/bin/env python
"""
Extract VQ-VAE codebook indices from ECG data for HiFi-GAN training.

Usage:
    python scripts/extract_codes.py \
        --vae_path ./checkpoints/vae_ptbxl.pt \
        --data_path ./data/ptbxl_ecg_train.pt \
        --output_path ./data/ptbxl_vae_codes_train.pt \
        --batch_size 256
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from text_to_ecg.models.vae import DiscreteVAE, VQVAE
from text_to_ecg.data.dataset import ECGDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Extract VQ-VAE codes')
    parser.add_argument('--vae_path', type=str, required=True,
                        help='Path to trained VAE checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ECG data (.pt file, list of tensors)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for codes tensor')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for inference')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print(f"Loading VAE from {args.vae_path}")
    checkpoint = torch.load(args.vae_path, map_location='cpu')
    model_class = checkpoint.get('model_class', 'DiscreteVAE')
    if model_class == 'VQVAE':
        vae = VQVAE(**checkpoint['hparams'])
    else:
        vae = DiscreteVAE(**checkpoint['hparams'])
    vae.load_state_dict(checkpoint['weights'])
    print(f"Loaded {model_class}")
    vae = vae.to(device)
    vae.eval()

    # Load data
    print(f"Loading ECG data from {args.data_path}")
    dataset = ECGDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Total samples: {len(dataset)}")

    # Extract codes
    all_codes = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting codes"):
            batch = batch.to(device)
            codes = vae.get_codebook_indices(batch)  # [B, 312]
            all_codes.append(codes.cpu())

    all_codes = torch.cat(all_codes, dim=0)
    print(f"Extracted codes shape: {all_codes.shape}")

    torch.save(all_codes, args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == '__main__':
    main()
