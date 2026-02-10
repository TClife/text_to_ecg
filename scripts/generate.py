#!/usr/bin/env python
"""
Generate ECG from text using trained DALLE model.

Usage:
    python scripts/generate.py \
        --dalle_path ./checkpoints/dalle.pt \
        --bpe_path ./configs/tokenizer.json \
        --text "sinus rhythm with left bundle branch block" \
        --age 65 --gender 1
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from text_to_ecg.models.vae import DiscreteVAE
from text_to_ecg.models.dalle import DALLE_concat
from text_to_ecg.data.tokenizer import HugTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Generate ECG from text')

    # Model paths
    parser.add_argument('--dalle_path', type=str, required=True,
                        help='Path to trained DALLE checkpoint')
    parser.add_argument('--bpe_path', type=str, required=True,
                        help='Path to BPE tokenizer JSON')
    parser.add_argument('--hifi_path', type=str, default=None,
                        help='Path to HiFi-GAN checkpoint (optional)')

    # Generation settings
    parser.add_argument('--text', type=str, required=True,
                        help='Text prompt for generation')
    parser.add_argument('--age', type=float, default=50,
                        help='Patient age')
    parser.add_argument('--gender', type=int, default=0,
                        help='Patient gender (0: female, 1: male)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate')
    parser.add_argument('--top_k', type=float, default=0.9,
                        help='Top-k sampling threshold')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--decoder', type=str, default='vqvae',
                        choices=['vqvae', 'hifi'],
                        help='Decoder type')

    # Output
    parser.add_argument('--output_path', type=str, default='./outputs/generated_ecg.pt',
                        help='Output path for generated ECG')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize generated ECG')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {args.bpe_path}")
    tokenizer = HugTokenizer(args.bpe_path)

    # Load DALLE
    print(f"Loading DALLE from {args.dalle_path}")
    checkpoint = torch.load(args.dalle_path, map_location='cpu')
    dalle_params = checkpoint['hparams']
    vae_params = checkpoint['vae_params']

    # Create VAE
    vae = DiscreteVAE(**vae_params)
    dalle = DALLE_concat(vae=vae, **dalle_params)
    dalle.load_state_dict(checkpoint['weights'])
    dalle = dalle.to(device)
    dalle.eval()

    print("Model loaded successfully")
    print(f"\nGenerating ECG for text: \"{args.text}\"")
    print(f"Age: {args.age}, Gender: {'Male' if args.gender == 1 else 'Female'}")

    # Tokenize text
    text_tokens = tokenizer.tokenize(
        [args.text],
        context_length=dalle.text_seq_len,
        truncate_text=True
    ).to(device)

    # Repeat for multiple samples
    if args.num_samples > 1:
        text_tokens = text_tokens.repeat(args.num_samples, 1)

    # Prepare age/gender
    age = torch.tensor([args.age] * args.num_samples, dtype=torch.float).to(device)
    gender = torch.tensor([args.gender] * args.num_samples, dtype=torch.float).to(device)

    # Generate
    print(f"\nGenerating {args.num_samples} sample(s)...")
    with torch.no_grad():
        generated_ecg = dalle.generate_images(
            text_tokens,
            age=age,
            gender=gender,
            filter_thres=args.top_k,
            temperature=args.temperature,
            decoder_model=args.decoder
        )

    print(f"Generated ECG shape: {generated_ecg.shape}")

    # Save
    output = {
        'ecg': generated_ecg.cpu(),
        'text': args.text,
        'age': args.age,
        'gender': args.gender,
    }
    torch.save(output, args.output_path)
    print(f"Saved to {args.output_path}")

    # Visualize
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(12, 1, figsize=(15, 20))
            ecg = generated_ecg[0].cpu().numpy()
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

            for i, (ax, name) in enumerate(zip(axes, lead_names)):
                ax.plot(ecg[i])
                ax.set_ylabel(name)
                ax.set_xlim(0, len(ecg[i]))
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel('Samples')
            plt.suptitle(f'Generated ECG: "{args.text}"')
            plt.tight_layout()

            viz_path = args.output_path.replace('.pt', '.png')
            plt.savefig(viz_path, dpi=150)
            print(f"Saved visualization to {viz_path}")
            plt.close()

        except ImportError:
            print("matplotlib not available for visualization")

    print("\nGeneration complete!")


if __name__ == '__main__':
    main()
