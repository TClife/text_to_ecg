#!/usr/bin/env python
"""
Train HiFi-GAN decoder for high-fidelity ECG synthesis.

Usage:
    python scripts/train_hifi_gan.py \
        --config ./configs/hifi_config.json \
        --data_path /path/to/ecg_data.pt \
        --codes_path /path/to/vae_codes.pt
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from text_to_ecg.models.hifi_gan import CodeGenerator
from text_to_ecg.models.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from text_to_ecg.utils.helpers import (
    AttrDict,
    load_checkpoint,
    save_checkpoint,
    scan_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train HiFi-GAN for ECG')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ECG data (.pt file)')
    parser.add_argument('--codes_path', type=str, required=True,
                        help='Path to VQ-VAE codes (.pt file)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/hifi_gan',
                        help='Checkpoint directory')
    parser.add_argument('--training_epochs', type=int, default=2000,
                        help='Number of training epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=500,
                        help='Checkpoint save interval (steps)')

    return parser.parse_args()


class ECGCodeDataset(torch.utils.data.Dataset):
    """Dataset for ECG signals with VQ-VAE codes."""

    def __init__(self, data_path, codes_path, segment_size=5000):
        self.data = torch.load(data_path)
        self.codes = torch.load(codes_path)
        self.segment_size = segment_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ecg = self.data[idx]
        codes = self.codes[idx]

        # Ensure correct length
        if ecg.shape[-1] > self.segment_size:
            ecg = ecg[..., :self.segment_size]

        return codes, ecg


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    """Compute mel spectrogram."""
    import librosa
    import numpy as np

    mels = []
    for i in range(y.shape[0]):
        mel = librosa.feature.melspectrogram(
            y=y[i].cpu().numpy(),
            sr=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax
        )
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        mels.append(torch.from_numpy(mel))

    return torch.stack(mels)


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config_data = json.load(f)
    h = AttrDict(config_data)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Create models
    generator = CodeGenerator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")

    # Check for existing checkpoint
    cp_g = scan_checkpoint(args.checkpoint_path, 'g_')
    cp_do = scan_checkpoint(args.checkpoint_path, 'do_')

    steps = 0
    last_epoch = -1

    if cp_g is not None and cp_do is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        print(f"Resuming from step {steps}, epoch {last_epoch}")

    # Optimizers
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2]
    )

    if cp_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # Dataset
    dataset = ECGCodeDataset(args.data_path, args.codes_path)
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=h.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=h.batch_size, shuffle=False, num_workers=4)

    # Tensorboard
    sw = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))

    # Training
    generator.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch + 1), args.training_epochs):
        print(f"\nEpoch {epoch + 1}/{args.training_epochs}")

        for batch in tqdm(train_loader, desc="Training"):
            codes, y = batch
            codes = codes.to(device)
            y = y.to(device).float()

            # Generate
            y_g_hat = generator(codes)

            # Ensure same shape
            if y_g_hat.shape[-1] != y.shape[-1]:
                min_len = min(y_g_hat.shape[-1], y.shape[-1])
                y_g_hat = y_g_hat[..., :min_len]
                y = y[..., :min_len]

            # Compute mel spectrograms
            y_mel = mel_spectrogram(y, h.n_fft, h.get('num_mels', 80), 500,
                                    h.hop_size, h.win_size, h.fmin, h.get('fmax_for_loss', 8000))
            y_g_hat_mel = mel_spectrogram(y_g_hat.detach(), h.n_fft, h.get('num_mels', 80), 500,
                                          h.hop_size, h.win_size, h.fmin, h.get('fmax_for_loss', 8000))

            y_mel = y_mel.to(device)
            y_g_hat_mel = y_g_hat_mel.to(device)

            # Discriminator step
            optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # Generator step
            optim_g.zero_grad()

            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            loss_gen_all.backward()
            optim_g.step()

            # Logging
            sw.add_scalar('training/gen_loss', loss_gen_all.item(), steps)
            sw.add_scalar('training/disc_loss', loss_disc_all.item(), steps)
            sw.add_scalar('training/mel_loss', loss_mel.item(), steps)

            # Checkpoint
            if steps % args.checkpoint_interval == 0 and steps > 0:
                checkpoint_path = f"{args.checkpoint_path}/g_{steps:08d}"
                save_checkpoint(checkpoint_path, {'generator': generator.state_dict()})

                checkpoint_path = f"{args.checkpoint_path}/do_{steps:08d}"
                save_checkpoint(checkpoint_path, {
                    'mpd': mpd.state_dict(),
                    'msd': msd.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'steps': steps,
                    'epoch': epoch,
                })

            steps += 1

        # LR decay
        scheduler_g.step()
        scheduler_d.step()

        # Validation
        generator.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for codes, y in val_loader:
                codes = codes.to(device)
                y = y.to(device).float()

                y_g_hat = generator(codes)

                if y_g_hat.shape[-1] != y.shape[-1]:
                    min_len = min(y_g_hat.shape[-1], y.shape[-1])
                    y_g_hat = y_g_hat[..., :min_len]
                    y = y[..., :min_len]

                y_mel = mel_spectrogram(y, h.n_fft, h.get('num_mels', 80), 500,
                                        h.hop_size, h.win_size, h.fmin, h.get('fmax_for_loss', 8000))
                y_g_hat_mel = mel_spectrogram(y_g_hat, h.n_fft, h.get('num_mels', 80), 500,
                                              h.hop_size, h.win_size, h.fmin, h.get('fmax_for_loss', 8000))

                val_loss += F.l1_loss(y_mel.to(device), y_g_hat_mel.to(device)).item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        sw.add_scalar('validation/mel_loss', avg_val_loss, epoch)
        print(f"Validation mel loss: {avg_val_loss:.4f}")

        generator.train()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
