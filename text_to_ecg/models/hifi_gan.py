"""
HiFi-GAN: High-fidelity ECG waveform generator.

Converts discrete ECG tokens to continuous waveforms using
a GAN-based neural vocoder architecture.

Adapted from: https://github.com/jik876/hifi-gan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from text_to_ecg.utils.helpers import init_weights, get_padding


LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    """Residual block with dilated convolutions (Type 1).

    Uses three dilated convolutions with different dilation rates.
    """

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0])
            )),
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1])
            )),
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[2],
                padding=get_padding(kernel_size, dilation[2])
            ))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1,
                padding=get_padding(kernel_size, 1)
            )),
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1,
                padding=get_padding(kernel_size, 1)
            )),
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1,
                padding=get_padding(kernel_size, 1)
            ))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    """Residual block with dilated convolutions (Type 2).

    Uses two dilated convolutions.
    """

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0])
            )),
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1])
            ))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(nn.Module):
    """HiFi-GAN Generator for waveform synthesis.

    Args:
        h: Hyperparameter dictionary containing:
            - resblock_kernel_sizes: List of kernel sizes for residual blocks
            - upsample_rates: List of upsample rates
            - upsample_kernel_sizes: List of upsample kernel sizes
            - upsample_initial_channel: Initial channel count
            - resblock: '1' or '2' for ResBlock type
            - model_in_dim: Input dimension (default: 512)
    """

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(Conv1d(
            getattr(h, "model_in_dim", 512),
            h.upsample_initial_channel,
            7, 1, padding=3
        ))

        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            if i == 0:
                self.ups.append(weight_norm(ConvTranspose1d(
                    h.upsample_initial_channel // (2 ** i),
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2, output_padding=1
                )))
            else:
                self.ups.append(weight_norm(ConvTranspose1d(
                    h.upsample_initial_channel // (2 ** i),
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2
                )))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 12, 7, 1, padding=3))  # 12-lead ECG
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class CodeGenerator(Generator):
    """Generator that takes discrete codes as input.

    Embeds discrete codebook indices and passes through the generator.
    """

    def __init__(self, h):
        super().__init__(h)
        self.dict = nn.Embedding(h.num_embeddings, h.embedding_dim)
        self.f0 = h.get('f0', None)
        self.multispkr = h.get('multispkr', None)

        if self.multispkr:
            self.spkr = nn.Embedding(200, h.embedding_dim)

    @staticmethod
    def _upsample(signal, max_frames):
        """Upsample signal to match max_frames length."""
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError(
                'Padding condition signal - misalignment between condition features.'
            )

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, x):
        """Generate ECG from codebook indices.

        Args:
            x: Codebook indices [B, T]

        Returns:
            Generated ECG waveform [B, C, T']
        """
        # Embed codes
        x = self.dict(x).transpose(1, 2)  # [B, D, T]

        # Pass through generator
        return super().forward(x)
