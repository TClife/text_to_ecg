#!/usr/bin/env python
"""
Generate ECG from text using trained DALLE model.

Usage:
    # VQ-VAE decoder (default):
    python scripts/generate.py \
        --dalle_path ./checkpoints/dalle.pt \
        --bpe_path ./configs/tokenizer.json \
        --text "sinus rhythm with left bundle branch block" \
        --age 65 --gender 1

    # HiFi-GAN decoder:
    python scripts/generate.py \
        --dalle_path ./checkpoints/dalle.pt \
        --bpe_path ./configs/tokenizer.json \
        --hifi_path ./checkpoints/hifi_gan/g_00039000 \
        --hifi_config ./configs/hifi_config.json \
        --decoder hifi \
        --text "sinus rhythm with left bundle branch block" \
        --age 65 --gender 1
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from text_to_ecg.models.vae import DiscreteVAE, VQVAE
from text_to_ecg.models.dalle import DALLE_concat
from text_to_ecg.data.tokenizer import HugTokenizer


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def parse_args():
    parser = argparse.ArgumentParser(description='Generate ECG from text')

    # Model paths
    parser.add_argument('--dalle_path', type=str, required=True,
                        help='Path to trained DALLE checkpoint')
    parser.add_argument('--bpe_path', type=str, required=True,
                        help='Path to BPE tokenizer JSON')
    parser.add_argument('--hifi_path', type=str, default=None,
                        help='Path to HiFi-GAN generator checkpoint (e.g. g_00039000)')
    parser.add_argument('--hifi_config', type=str, default='./configs/hifi_config.json',
                        help='Path to HiFi-GAN config JSON')

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
                        help='Decoder type: vqvae or hifi (requires --hifi_path)')

    # Output
    parser.add_argument('--output_path', type=str, default='./outputs/generated_ecg.pt',
                        help='Output path for generated ECG')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize generated ECG')

    return parser.parse_args()


def load_hifi_gan(hifi_path, config_path, device):
    """Load HiFi-GAN CodeGenerator from checkpoint."""
    from text_to_ecg.models.hifi_gan import CodeGenerator

    with open(config_path) as f:
        h = AttrDict(json.load(f))

    generator = CodeGenerator(h).to(device)
    cp = torch.load(hifi_path, map_location=device)
    generator.load_state_dict(cp['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator


def main():
    args = parse_args()

    if args.decoder == 'hifi' and args.hifi_path is None:
        raise ValueError("--hifi_path is required when using --decoder hifi")

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
    vae_model_class = checkpoint.get('vae_model_class', 'DiscreteVAE')
    if vae_model_class == 'VQVAE':
        vae = VQVAE(**vae_params)
    else:
        vae = DiscreteVAE(**vae_params)
    dalle = DALLE_concat(vae=vae, **dalle_params)
    dalle.load_state_dict(checkpoint['weights'])
    dalle = dalle.to(device)
    dalle.eval()

    # Load HiFi-GAN if needed
    hifi_gan = None
    if args.decoder == 'hifi':
        print(f"Loading HiFi-GAN from {args.hifi_path}")
        hifi_gan = load_hifi_gan(args.hifi_path, args.hifi_config, device)

    print("Models loaded successfully")
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
    print(f"\nGenerating {args.num_samples} sample(s) with {args.decoder} decoder...")
    with torch.no_grad():
        if args.decoder == 'hifi':
            # Generate codes via autoregressive loop, then decode with HiFi-GAN
            from text_to_ecg.models.dalle import top_k

            text_seq_len = dalle.text_seq_len
            image_seq_len = dalle.image_seq_len
            num_text_tokens = dalle.num_text_tokens

            text = text_tokens[:, :text_seq_len - 2]
            age_tok = dalle._encode_age(age).unsqueeze(1).to(device)
            gender_tok = dalle._encode_gender(gender).unsqueeze(1).to(device)
            text = torch.cat([gender_tok, age_tok, text], dim=1)
            text = text[:, :text_seq_len]
            out = text
            total_len = text_seq_len + image_seq_len

            for cur_len in range(out.shape[1], total_len):
                is_image = cur_len >= text_seq_len
                text_part = out[:, :text_seq_len]
                image_part = out[:, text_seq_len:]
                logits = dalle.forward_generate(text_part, image_part)[:, -1, :]
                filtered_logits = top_k(logits, thres=args.top_k)
                probs = F.softmax(filtered_logits / args.temperature, dim=-1)
                sample = torch.multinomial(probs, 1)
                sample -= (num_text_tokens if is_image else 0)
                out = torch.cat((out, sample), dim=-1)

            img_seq = out[:, -image_seq_len:]
            generated_ecg = hifi_gan(img_seq)
        else:
            generated_ecg = dalle.generate_images(
                text_tokens,
                age=age,
                gender=gender,
                filter_thres=args.top_k,
                temperature=args.temperature,
                decoder_model="vqvae"
            )

    print(f"Generated ECG shape: {generated_ecg.shape}")

    # Save
    output = {
        'ecg': generated_ecg.cpu(),
        'text': args.text,
        'age': args.age,
        'gender': args.gender,
        'decoder': args.decoder,
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
            decoder_label = 'HiFi-GAN' if args.decoder == 'hifi' else 'VQ-VAE'
            plt.suptitle(f'Generated ECG ({decoder_label}): "{args.text}"')
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
