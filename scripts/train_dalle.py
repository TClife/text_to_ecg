#!/usr/bin/env python
"""
Train DALLE autoregressive transformer for text-to-ECG generation.

Usage:
    python scripts/train_dalle.py \
        --vae_path ./checkpoints/vae.pt \
        --data_path /path/to/dataset.pt \
        --bpe_path ./configs/tokenizer.json
"""

import argparse
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from text_to_ecg.models.vae import DiscreteVAE, VQVAE
from text_to_ecg.models.dalle import DALLE_concat
from text_to_ecg.data.dataset import TextImageDataset
from text_to_ecg.data.tokenizer import HugTokenizer
from text_to_ecg.utils import distributed as distributed_utils
from text_to_ecg.utils.helpers import exists


def parse_args():
    parser = argparse.ArgumentParser(description='Train DALLE for text-to-ECG')

    # Model paths
    parser.add_argument('--vae_path', type=str, required=True,
                        help='Path to trained VAE checkpoint')
    parser.add_argument('--dalle_path', type=str, default=None,
                        help='Path to resume DALLE training')
    parser.add_argument('--bpe_path', type=str, required=True,
                        help='Path to BPE tokenizer JSON')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset (.pt file)')
    parser.add_argument('--truncate_captions', action='store_true', default=True,
                        help='Truncate long captions')

    # Model architecture
    parser.add_argument('--dim', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dim_head', type=int, default=64,
                        help='Dimension per head')
    parser.add_argument('--text_seq_len', type=int, default=104,
                        help='Maximum text sequence length')
    parser.add_argument('--loss_img_weight', type=int, default=7,
                        help='Weight for ECG token loss')
    parser.add_argument('--reversible', action='store_true',
                        help='Use reversible layers')
    parser.add_argument('--attn_types', type=str, default='full',
                        help='Attention types (comma-separated)')
    parser.add_argument('--ff_dropout', type=float, default=0.2,
                        help='Feed-forward dropout')
    parser.add_argument('--attn_dropout', type=float, default=0.2,
                        help='Attention dropout')
    parser.add_argument('--stable_softmax', action='store_true',
                        help='Use stable softmax')
    parser.add_argument('--shift_tokens', action='store_true',
                        help='Use token shifting')
    parser.add_argument('--rotary_emb', action='store_true',
                        help='Use rotary embeddings')

    # Training
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--clip_grad_norm', type=float, default=0.5,
                        help='Gradient clipping norm')
    parser.add_argument('--lr_decay', action='store_true',
                        help='Use learning rate decay')

    # Output
    parser.add_argument('--output_path', type=str, default='./checkpoints/dalle.pt',
                        help='Output checkpoint path')
    parser.add_argument('--save_every', type=int, default=1,
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

    # Load tokenizer
    print(f"Loading tokenizer from {args.bpe_path}")
    tokenizer = HugTokenizer(args.bpe_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load VAE
    print(f"Loading VAE from {args.vae_path}")
    vae_checkpoint = torch.load(args.vae_path, map_location='cpu')
    vae_params = vae_checkpoint['hparams']
    vae_model_class = vae_checkpoint.get('model_class', 'DiscreteVAE')
    if vae_model_class == 'VQVAE':
        vae = VQVAE(**vae_params)
    else:
        vae = DiscreteVAE(**vae_params)
    print(f"VAE type: {vae_model_class}")
    vae.load_state_dict(vae_checkpoint['weights'])
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded successfully")

    # Load or create DALLE
    if args.dalle_path and Path(args.dalle_path).exists():
        print(f"Resuming from {args.dalle_path}")
        dalle_checkpoint = torch.load(args.dalle_path, map_location='cpu')
        dalle_params = dalle_checkpoint['hparams']
        resume_epoch = dalle_checkpoint.get('epoch', 0)

        dalle = DALLE_concat(vae=vae, **dalle_params)
        dalle.load_state_dict(dalle_checkpoint['weights'])
    else:
        print("Creating new DALLE model")
        dalle_params = {
            'num_text_tokens': tokenizer.vocab_size,
            'text_seq_len': args.text_seq_len,
            'dim': args.dim,
            'depth': args.depth,
            'heads': args.heads,
            'dim_head': args.dim_head,
            'reversible': args.reversible,
            'loss_img_weight': args.loss_img_weight,
            'attn_types': tuple(args.attn_types.split(',')),
            'ff_dropout': args.ff_dropout,
            'attn_dropout': args.attn_dropout,
            'stable': args.stable_softmax,
            'shift_tokens': args.shift_tokens,
            'rotary_emb': args.rotary_emb,
        }
        dalle = DALLE_concat(vae=vae, **dalle_params)
        resume_epoch = 0

    dalle = dalle.to(device)
    print(f"DALLE parameters: {sum(p.numel() for p in dalle.parameters() if p.requires_grad):,}")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        dalle = torch.nn.DataParallel(dalle)
        dalle_module = dalle.module
    else:
        dalle_module = dalle

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = TextImageDataset(
        args.data_path,
        text_len=args.text_seq_len,
        truncate_captions=args.truncate_captions,
        tokenizer=tokenizer,
    )
    print(f"Dataset size: {len(dataset)}")

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Optimizer
    optimizer = Adam(
        [p for p in dalle_module.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    if args.lr_decay:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
    else:
        scheduler = None

    # Training loop
    def save_model(path, epoch):
        save_obj = {
            'hparams': dalle_params,
            'vae_params': vae_params,
            'vae_model_class': vae_model_class,
            'weights': dalle_module.state_dict(),
            'opt_state': optimizer.state_dict(),
            'epoch': epoch,
        }
        if scheduler:
            save_obj['scheduler_state'] = scheduler.state_dict()
        torch.save(save_obj, path)

    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(resume_epoch, args.epochs):
        # Training
        dalle.train()
        train_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for text, images, age, gender in pbar:
            text = text.to(device)
            images = images.to(device)
            age = age.float().to(device)
            gender = gender.float().to(device)

            loss = dalle(text, images, age, gender, return_loss=True)
            if loss.dim() > 0:
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(dalle_module.parameters(), args.clip_grad_norm)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / num_batches

        # Validation
        dalle.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for text, images, age, gender in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                text = text.to(device)
                images = images.to(device)
                age = age.float().to(device)
                gender = gender.float().to(device)

                loss = dalle(text, images, age, gender, return_loss=True)
                if loss.dim() > 0:
                    loss = loss.mean()
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Learning rate decay
        if scheduler:
            scheduler.step(avg_val_loss)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_model(args.output_path, epoch + 1)
            print(f"Saved checkpoint to {args.output_path}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = args.output_path.replace('.pt', '_best.pt')
            save_model(best_path, epoch + 1)
            print(f"Saved best model to {best_path}")

    print("Training complete!")


if __name__ == '__main__':
    main()
