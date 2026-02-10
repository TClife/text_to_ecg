"""
GAN Discriminators for HiFi-GAN training.

Includes:
- MultiPeriodDiscriminator: Captures periodic patterns in ECG
- MultiScaleDiscriminator: Captures multi-scale features

Adapted from: https://github.com/jik876/hifi-gan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm

from text_to_ecg.utils.helpers import get_padding


LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    """Period-based discriminator.

    Reshapes input into 2D and discriminates periodic patterns.

    Args:
        period: Period for reshaping (e.g., 2, 3, 5, 7, 11)
        kernel_size: Convolution kernel size
        stride: Convolution stride
        use_spectral_norm: Use spectral normalization
    """

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(Conv2d(12, 32, (kernel_size, 1), (stride, 1),
                         padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1),
                         padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1),
                         padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1),
                         padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 12, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Tuple of (output, feature_maps)
        """
        fmap = []

        # 1D to 2D
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator.

    Uses multiple period-based discriminators with different periods
    to capture various periodic patterns in ECG.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        """Forward pass.

        Args:
            y: Real ECG [B, C, T]
            y_hat: Generated ECG [B, C, T]

        Returns:
            Tuple of (real_outputs, fake_outputs, real_fmaps, fake_fmaps)
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    """Scale-based discriminator.

    Args:
        use_spectral_norm: Use spectral normalization
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(Conv1d(12, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 12, 3, 1, padding=1))

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Tuple of (output, feature_maps)
        """
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator.

    Uses multiple scale-based discriminators at different resolutions.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        """Forward pass.

        Args:
            y: Real ECG [B, C, T]
            y_hat: Generated ECG [B, C, T]

        Returns:
            Tuple of (real_outputs, fake_outputs, real_fmaps, fake_fmaps)
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)

            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    """Feature matching loss.

    Args:
        fmap_r: Feature maps from real input
        fmap_g: Feature maps from generated input

    Returns:
        Feature matching loss value
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Discriminator hinge loss.

    Args:
        disc_real_outputs: Discriminator outputs for real input
        disc_generated_outputs: Discriminator outputs for generated input

    Returns:
        Tuple of (total_loss, real_losses, generated_losses)
    """
    loss = 0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """Generator loss.

    Args:
        disc_outputs: Discriminator outputs for generated input

    Returns:
        Tuple of (total_loss, individual_losses)
    """
    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
