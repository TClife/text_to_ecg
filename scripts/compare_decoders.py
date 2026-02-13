#!/usr/bin/env python
"""
Compare ECG generation: Ground Truth vs VQ-VAE decoded vs HiFi-GAN decoded.

Generates ECG codes from text using DALLE, decodes with both VQ-VAE and HiFi-GAN,
and compares against a ground truth sample with matching text.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from text_to_ecg.models.vae import DiscreteVAE, VQVAE
from text_to_ecg.models.dalle import DALLE_concat
from text_to_ecg.models.hifi_gan import CodeGenerator
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


def find_matching_gt(data_path, query_text, n=1):
    """Find ground truth ECG samples whose report contains the query text."""
    data = torch.load(data_path, map_location='cpu')
    reports = data['report_data']
    ecgs = data['data']
    ages = data['age']
    genders = data['gender']

    matches = []
    query_lower = query_text.lower()
    for i, report in enumerate(reports):
        if query_lower in report.lower():
            matches.append({
                'index': i,
                'report': report,
                'ecg': ecgs[i],
                'age': ages[i].item(),
                'gender': genders[i].item(),
            })
            if len(matches) >= n:
                break

    return matches


def main():
    parser = argparse.ArgumentParser(description='Compare VQ-VAE vs HiFi-GAN decoding')
    parser.add_argument('--dalle_path', type=str, default='./checkpoints/dalle_ptbxl_vq_best.pt')
    parser.add_argument('--hifi_path', type=str, default='./checkpoints/hifi_gan_vq/g_00039000')
    parser.add_argument('--hifi_config', type=str, default='./configs/hifi_config.json')
    parser.add_argument('--bpe_path', type=str, default='./configs/ptbxl_tokenizer.json')
    parser.add_argument('--data_path', type=str, default='./data/ptbxl_train.pt')
    parser.add_argument('--text', type=str, default='sinus rhythm normal ecg')
    parser.add_argument('--age', type=float, default=55)
    parser.add_argument('--gender', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='./outputs/compare_decoders.png')
    parser.add_argument('--top_k', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # --- Load DALLE ---
    print(f"Loading DALLE from {args.dalle_path}")
    checkpoint = torch.load(args.dalle_path, map_location='cpu')
    dalle_params = checkpoint['hparams']
    vae_params = checkpoint['vae_params']

    vae_model_class = checkpoint.get('vae_model_class', 'DiscreteVAE')
    if vae_model_class == 'VQVAE':
        vae = VQVAE(**vae_params)
    else:
        vae = DiscreteVAE(**vae_params)

    dalle = DALLE_concat(vae=vae, **dalle_params)
    dalle.load_state_dict(checkpoint['weights'])
    dalle = dalle.to(device)
    dalle.eval()
    print("DALLE loaded.")

    # --- Load HiFi-GAN ---
    print(f"Loading HiFi-GAN from {args.hifi_path}")
    with open(args.hifi_config) as f:
        h = AttrDict(json.load(f))

    hifi_gan = CodeGenerator(h).to(device)
    cp = torch.load(args.hifi_path, map_location=device)
    hifi_gan.load_state_dict(cp['generator'])
    hifi_gan.eval()
    hifi_gan.remove_weight_norm()
    print("HiFi-GAN loaded.")

    # --- Load Tokenizer ---
    tokenizer = HugTokenizer(args.bpe_path)

    # --- Generate ECG codes from text ---
    print(f"\nGenerating ECG for: \"{args.text}\"")
    print(f"Age: {args.age}, Gender: {'Male' if args.gender == 1 else 'Female'}")

    text_tokens = tokenizer.tokenize(
        [args.text],
        context_length=dalle.text_seq_len,
        truncate_text=True
    ).to(device)

    age_t = torch.tensor([args.age], dtype=torch.float).to(device)
    gender_t = torch.tensor([args.gender], dtype=torch.float).to(device)

    # Generate: get the raw codes by running the generation loop manually
    with torch.no_grad():
        # Use DALLE's generate method - it returns decoded ECG
        # We need the codes too, so let's extract them from the generation process
        vae_model = dalle.vae
        text_seq_len = dalle.text_seq_len
        image_seq_len = dalle.image_seq_len
        num_text_tokens = dalle.num_text_tokens

        # Prepare text with age/gender
        text = text_tokens[:, :text_seq_len - 2]
        age_tok = dalle._encode_age(age_t).unsqueeze(1).to(device)
        gender_tok = dalle._encode_gender(gender_t).unsqueeze(1).to(device)
        text = torch.cat([gender_tok, age_tok, text], dim=1)
        text = text[:, :text_seq_len]
        out = text

        total_len = text_seq_len + image_seq_len

        # Autoregressive generation
        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len
            text_part = out[:, :text_seq_len]
            image_part = out[:, text_seq_len:]
            logits = dalle.forward_generate(text_part, image_part)[:, -1, :]

            from text_to_ecg.models.dalle import top_k
            filtered_logits = top_k(logits, thres=args.top_k)
            probs = F.softmax(filtered_logits / args.temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            sample -= (num_text_tokens if is_image else 0)
            out = torch.cat((out, sample), dim=-1)

        # Extract generated codes
        img_seq = out[:, -image_seq_len:]  # [1, 312]
        print(f"Generated codes shape: {img_seq.shape}")
        print(f"Code range: [{img_seq.min().item()}, {img_seq.max().item()}]")
        print(f"Unique codes: {img_seq.unique().shape[0]}")

        # --- Decode with VQ-VAE ---
        ecg_vqvae = vae_model.decode(img_seq)  # [1, 12, 5000]
        print(f"VQ-VAE decoded shape: {ecg_vqvae.shape}")

        # --- Decode with HiFi-GAN ---
        ecg_hifi = hifi_gan(img_seq)  # [1, 12, T]
        print(f"HiFi-GAN decoded shape: {ecg_hifi.shape}")

    # --- Find Ground Truth ---
    print(f"\nSearching for GT sample matching: \"{args.text}\"")
    matches = find_matching_gt(args.data_path, args.text)
    if not matches:
        # Try partial match
        words = args.text.split()
        for word in words:
            if len(word) > 3:
                matches = find_matching_gt(args.data_path, word, n=1)
                if matches:
                    print(f"  Found match using keyword: \"{word}\"")
                    break
    if not matches:
        print("  No matching GT found. Using first sample as reference.")
        data = torch.load(args.data_path, map_location='cpu')
        matches = [{
            'index': 0,
            'report': data['report_data'][0],
            'ecg': data['data'][0],
            'age': data['age'][0].item(),
            'gender': data['gender'][0].item(),
        }]

    gt = matches[0]
    print(f"  GT index: {gt['index']}, report: \"{gt['report'][:100]}...\"")
    print(f"  GT age: {gt['age']}, gender: {'Male' if gt['gender'] == 1 else 'Female'}")

    # --- Convert to numpy ---
    ecg_gt = gt['ecg'].numpy()  # [12, 5000]
    ecg_vqvae_np = ecg_vqvae[0].cpu().numpy()  # [12, 5000]
    ecg_hifi_np = ecg_hifi[0].cpu().numpy()  # [12, T]

    # Trim/pad HiFi-GAN output to 5000 samples
    if ecg_hifi_np.shape[1] > 5000:
        ecg_hifi_np = ecg_hifi_np[:, :5000]
    elif ecg_hifi_np.shape[1] < 5000:
        ecg_hifi_np = np.pad(ecg_hifi_np, ((0, 0), (0, 5000 - ecg_hifi_np.shape[1])))

    # --- Plot comparison ---
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axes = plt.subplots(12, 3, figsize=(24, 28))
    time = np.arange(5000) / 500.0  # seconds

    col_titles = [
        f'Ground Truth\n"{gt["report"][:60]}..."',
        f'DALLE + VQ-VAE Decoder\n"{args.text}"',
        f'DALLE + HiFi-GAN Decoder\n"{args.text}"',
    ]

    for col, (ecg_data, title) in enumerate(zip(
        [ecg_gt, ecg_vqvae_np, ecg_hifi_np], col_titles
    )):
        for row, (lead_name, lead_data) in enumerate(zip(lead_names, ecg_data)):
            ax = axes[row, col]
            ax.plot(time, lead_data, linewidth=0.5, color=['tab:blue', 'tab:orange', 'tab:green'][col])
            ax.set_ylabel(lead_name, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(title, fontsize=10)
            if row == 11:
                ax.set_xlabel('Time (s)', fontsize=9)

    plt.suptitle(
        f'ECG Comparison: GT vs VQ-VAE vs HiFi-GAN\n'
        f'Prompt: "{args.text}" | Age: {args.age} | Gender: {"Male" if args.gender == 1 else "Female"}',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to {args.output_path}")
    plt.close()

    # Also save the data
    data_output = args.output_path.replace('.png', '.pt')
    torch.save({
        'gt_ecg': gt['ecg'],
        'gt_report': gt['report'],
        'gt_age': gt['age'],
        'gt_gender': gt['gender'],
        'generated_codes': img_seq.cpu(),
        'vqvae_ecg': ecg_vqvae.cpu(),
        'hifi_ecg': ecg_hifi.cpu(),
        'text': args.text,
        'age': args.age,
        'gender': args.gender,
    }, data_output)
    print(f"Saved data to {data_output}")


if __name__ == '__main__':
    main()
