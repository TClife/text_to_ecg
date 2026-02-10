#!/usr/bin/env python
"""
Train VQ-VAE for ECG tokenization.

Usage:
    python scripts/train_vae.py --data_path /path/to/ecg_data.pt --epochs 100
"""

import argparse
import math
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from text_to_ecg.models.vae import DiscreteVAE
from text_to_ecg.data.dataset import ECGDataset
from text_to_ecg.utils import distributed as distributed_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Train VQ-VAE for ECG tokenization')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ECG data (.pt file)')

    # Model
    parser.add_argument('--num_tokens', type=int, default=1024,
                        help='Number of codebook entries')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of encoder/decoder layers')
    parser.add_argument('--num_resnet_blocks', type=int, default=2,
                        help='Number of residual blocks')
    parser.add_argument('--codebook_dim', type=int, default=512,
                        help='Codebook embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--image_size', type=int, default=5000,
                        help='ECG sequence length')
    parser.add_argument('--smooth_l1_loss', action='store_true',
                        help='Use smooth L1 loss instead of MSE')
    parser.add_argument('--kl_loss_weight', type=float, default=0.,
                        help='KL divergence loss weight')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.98,
                        help='Learning rate decay rate')
    parser.add_argument('--starting_temp', type=float, default=1.0,
                        help='Starting Gumbel-softmax temperature')
    parser.add_argument('--temp_min', type=float, default=0.5,
                        help='Minimum temperature')
    parser.add_argument('--anneal_rate', type=float, default=1e-6,
                        help='Temperature annealing rate')

    # Output
    parser.add_argument('--output_path', type=str, default='./checkpoints/vae.pt',
                        help='Output checkpoint path')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Distributed
    parser = distributed_utils.wrap_arg_parser(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_path}")
    dataset = ECGDataset(args.data_path)
    print(f"Loaded {len(dataset)} ECG samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    vae_params = {
        'image_size': args.image_size,
        'num_tokens': args.num_tokens,
        'codebook_dim': args.codebook_dim,
        'num_layers': args.num_layers,
        'num_resnet_blocks': args.num_resnet_blocks,
        'hidden_dim': args.hidden_dim,
        'channels': 12,  # 12-lead ECG
        'smooth_l1_loss': args.smooth_l1_loss,
        'kl_div_loss_weight': args.kl_loss_weight,
    }

    vae = DiscreteVAE(**vae_params).to(device)
    print(f"Created VAE with {sum(p.numel() for p in vae.parameters()):,} parameters")

    # Optimizer
    optimizer = Adam(vae.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay_rate)

    # Training
    temp = args.starting_temp
    global_step = 0

    for epoch in range(args.epochs):
        vae.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)

            # Forward pass
            loss, recons = vae(batch, return_loss=True, return_recons=True, temp=temp)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Temperature annealing
            temp = max(temp * math.exp(-args.anneal_rate), args.temp_min)
            global_step += 1

            pbar.set_postfix(loss=loss.item(), temp=temp)

        # Learning rate decay
        scheduler.step()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, temp={temp:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_obj = {
                'hparams': vae_params,
                'weights': vae.state_dict(),
                'epoch': epoch + 1,
            }
            torch.save(save_obj, args.output_path)
            print(f"Saved checkpoint to {args.output_path}")

    print("Training complete!")


if __name__ == '__main__':
    main()
