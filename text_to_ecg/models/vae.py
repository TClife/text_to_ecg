"""
Vector Quantized Variational Autoencoder (VQ-VAE) for ECG tokenization.

This module provides two VAE variants:
- DiscreteVAE: Gumbel-softmax based discrete VAE
- VQVAE: Vector quantization with EMA codebook updates
"""

from math import sqrt

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from text_to_ecg.utils.helpers import exists, default, eval_decorator, set_requires_grad
from text_to_ecg.utils import distributed as distributed_utils


class ResBlock(nn.Module):
    """Residual block for encoder/decoder."""

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class ResBlock1(nn.Module):
    """Residual block with dilated convolutions (HiFi-GAN style)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        from torch.nn.utils import weight_norm
        from text_to_ecg.utils.helpers import get_padding

        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                  padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                                  padding=get_padding(kernel_size, dilation[2])))
        ])

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class VectorQuantize(nn.Module):
    """Vector quantization with EMA codebook updates.

    Args:
        dim: Embedding dimension
        codebook_size: Number of codebook entries
        decay: EMA decay rate
        commitment_weight: Commitment loss weight
    """

    def __init__(self, dim, codebook_size, decay=0.8, commitment_weight=1.):
        super().__init__()
        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment_weight = commitment_weight

        embed = torch.randn(dim, codebook_size)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer('inited', torch.tensor([False]))

    def _init_from_data(self, data):
        """Initialize codebook from encoder outputs (first batch)."""
        # data: [N, D]
        n, d = data.shape
        if n >= self.codebook_size:
            indices = torch.randperm(n, device=data.device)[:self.codebook_size]
            embed = data[indices].transpose(0, 1)  # [D, K]
        else:
            # Repeat data to fill codebook, add small noise
            repeats = (self.codebook_size // n) + 1
            embed = data.repeat(repeats, 1)[:self.codebook_size]
            embed = embed + torch.randn_like(embed) * embed.std() * 0.1
            embed = embed.transpose(0, 1)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed)
        self.cluster_size.data.fill_(1.0)
        self.inited.fill_(True)

    def forward(self, x):
        """Quantize input tensor.

        Args:
            x: Input tensor of shape [B, T, D]

        Returns:
            Tuple of (quantized, indices, commitment_loss)
        """
        flatten = rearrange(x, 'b t d -> (b t) d')

        # Initialize codebook from first batch of encoder outputs
        if self.training and not self.inited:
            self._init_from_data(flatten.detach())

        # Find nearest codebook entries
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(flatten.dtype)

        # Quantize
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        # EMA codebook update
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            # Reset unused codebook entries to random encoder outputs
            # An entry is "unused" if its EMA cluster size is very small
            avg_usage = self.cluster_size.sum() / self.codebook_size
            unused = (self.cluster_size < avg_usage * 0.01)
            num_unused = unused.sum().item()
            if num_unused > 0 and flatten.shape[0] > 0:
                rand_idx = torch.randint(0, flatten.shape[0], (int(num_unused),), device=flatten.device)
                self.embed.data[:, unused] = flatten[rand_idx].transpose(0, 1)
                self.embed_avg.data[:, unused] = flatten[rand_idx].transpose(0, 1)
                self.cluster_size.data[unused] = avg_usage

        # Commitment loss
        commitment_loss = F.mse_loss(quantize.detach(), x) * self.commitment_weight

        # Straight-through estimator
        quantize = x + (quantize - x).detach()

        return quantize, embed_ind, commitment_loss


class DiscreteVAE(nn.Module):
    """Discrete VAE using Gumbel-softmax for ECG tokenization.

    Args:
        image_size: ECG sequence length (default: 5000 for 10s at 500Hz)
        num_tokens: Number of codebook entries
        codebook_dim: Codebook embedding dimension
        num_layers: Number of encoder/decoder layers
        num_resnet_blocks: Number of residual blocks
        hidden_dim: Hidden layer dimension
        channels: Number of ECG channels (default: 12-lead)
        smooth_l1_loss: Use smooth L1 loss instead of MSE
        temperature: Gumbel-softmax temperature
        straight_through: Use straight-through gradient estimator
        kl_div_loss_weight: Weight for KL divergence loss
    """

    def __init__(
        self,
        image_size=5000,
        num_tokens=1024,
        codebook_dim=512,
        num_layers=4,
        num_resnet_blocks=0,
        hidden_dim=256,
        channels=12,
        smooth_l1_loss=False,
        temperature=0.9,
        straight_through=False,
        kl_div_loss_weight=0.,
        normalization=None
    ):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        # Encoder
        enc_chans = [hidden_dim] * num_layers
        enc_chans = [channels, *enc_chans]
        enc_chans_io = list(zip(enc_chans[:-1], enc_chans[1:]))

        enc_layers = []
        for enc_in, enc_out in enc_chans_io:
            enc_layers.append(nn.Sequential(
                nn.Conv1d(enc_in, enc_out, 4, stride=2, padding=1),
                nn.ReLU()
            ))

        for _ in range(num_resnet_blocks):
            enc_layers.append(ResBlock(enc_chans[-1]))

        enc_layers.append(nn.Conv1d(enc_chans[-1], num_tokens, 1))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_chans = list(reversed(enc_chans[1:]))
        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]
        dec_chans_io = list(zip(dec_chans[:-1], dec_chans[1:]))

        dec_layers = []
        if has_resblocks:
            dec_layers.append(nn.Conv1d(codebook_dim, dec_chans[1], 1))

        for _ in range(num_resnet_blocks):
            dec_layers.append(ResBlock(dec_chans[1] if has_resblocks else dec_chans[0]))

        count = 0
        for dec_in, dec_out in dec_chans_io:
            if count == 0:
                dec_layers.append(nn.Sequential(
                    nn.ConvTranspose1d(dec_in, dec_out, 4, stride=2, padding=1, output_padding=1),
                    nn.ReLU()
                ))
            else:
                dec_layers.append(nn.Sequential(
                    nn.ConvTranspose1d(dec_in, dec_out, 4, stride=2, padding=1),
                    nn.ReLU()
                ))
            count += 1

        dec_layers.append(nn.Conv1d(dec_chans[-1], channels, 1))
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight
        self.normalization = normalization

        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if not distributed_utils.is_distributed:
            return

        if distributed_utils.using_backend(distributed_utils.DeepSpeedBackend):
            deepspeed = distributed_utils.backend.backend_module
            deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        """Get discrete codebook indices for input ECG.

        Args:
            images: ECG tensor of shape [B, C, T]

        Returns:
            Tensor of codebook indices [B, T']
        """
        logits = self(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    def decode(self, img_seq):
        """Decode from codebook indices.

        Args:
            img_seq: Codebook indices [B, T']

        Returns:
            Reconstructed ECG [B, C, T]
        """
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        image_embeds = rearrange(image_embeds, 'b t d -> b d t')
        images = self.decoder(image_embeds)
        return images

    def forward(self, ecg, return_loss=False, return_recons=False, return_logits=False, temp=None):
        """Forward pass through VAE.

        Args:
            ecg: Input ECG tensor [B, C, T]
            return_loss: Return reconstruction loss
            return_recons: Return reconstructed ECG along with loss
            return_logits: Return encoder logits (for getting codebook indices)
            temp: Temperature for Gumbel-softmax

        Returns:
            Depending on flags: logits, reconstruction, or (loss, reconstruction)
        """
        device = ecg.device
        num_tokens = self.num_tokens
        kl_div_loss_weight = self.kl_div_loss_weight

        ecg = ecg.to(device)
        logits = self.encoder(ecg.float())

        if return_logits:
            return logits

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=self.straight_through)
        sampled = einsum('b n t, n d -> b d t', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # Reconstruction loss
        recon_loss = self.loss_fn(ecg, out)

        # KL divergence
        logits = rearrange(logits, 'b n t -> b t n')
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device=device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out


class VQVAE(nn.Module):
    """Vector Quantized VAE with EMA codebook updates.

    Args:
        image_size: ECG sequence length
        num_tokens: Number of codebook entries
        codebook_dim: Codebook embedding dimension
        num_layers: Number of encoder/decoder layers
        num_resnet_blocks: Number of residual blocks
        hidden_dim: Hidden layer dimension
        channels: Number of ECG channels
        vq_decay: EMA decay for codebook
        commitment_weight: Commitment loss weight
    """

    def __init__(
        self,
        image_size=312,
        num_tokens=1024,
        codebook_dim=512,
        num_layers=4,
        num_resnet_blocks=0,
        hidden_dim=256,
        channels=12,
        vq_decay=0.8,
        commitment_weight=1.
    ):
        super().__init__()
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        self.vq = VectorQuantize(
            dim=codebook_dim,
            codebook_size=num_tokens,
            decay=vq_decay,
            commitment_weight=commitment_weight
        )

        # Encoder
        enc_chans = [hidden_dim] * num_layers
        enc_chans = [channels, *enc_chans]
        enc_chans_io = list(zip(enc_chans[:-1], enc_chans[1:]))

        enc_layers = []
        for enc_in, enc_out in enc_chans_io:
            enc_layers.append(nn.Sequential(
                nn.Conv1d(enc_in, enc_out, 4, stride=2, padding=1),
                nn.ReLU()
            ))

        for _ in range(num_resnet_blocks):
            enc_layers.append(ResBlock(enc_chans[-1]))

        enc_layers.append(nn.Conv1d(enc_chans[-1], codebook_dim, 1))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_chans = list(reversed(enc_chans[1:]))
        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]
        dec_chans_io = list(zip(dec_chans[:-1], dec_chans[1:]))

        dec_layers = []
        if has_resblocks:
            dec_layers.append(nn.Conv1d(codebook_dim, dec_chans[1], 1))

        for _ in range(num_resnet_blocks):
            dec_layers.append(ResBlock(dec_chans[1] if has_resblocks else dec_chans[0]))

        count = 0
        for dec_in, dec_out in dec_chans_io:
            if count == 0:
                dec_layers.append(nn.Sequential(
                    nn.ConvTranspose1d(dec_in, dec_out, 4, stride=2, padding=1, output_padding=1),
                    nn.ReLU()
                ))
            else:
                dec_layers.append(nn.Sequential(
                    nn.ConvTranspose1d(dec_in, dec_out, 4, stride=2, padding=1),
                    nn.ReLU()
                ))
            count += 1

        dec_layers.append(nn.Conv1d(dec_chans[-1], channels, 1))
        self.decoder = nn.Sequential(*dec_layers)

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        """Get discrete codebook indices for input ECG."""
        encoded = self.forward(images, return_encoded=True)
        encoded = rearrange(encoded, 'b c t -> b t c')
        _, indices, _ = self.vq(encoded)
        return indices

    def decode(self, img_seq):
        """Decode from codebook indices."""
        codebook = rearrange(self.vq.embed, 'd n -> n d')
        image_embeds = codebook[img_seq]
        b, n, d = image_embeds.shape
        image_embeds = rearrange(image_embeds, 'b t d -> b d t')
        images = self.decoder(image_embeds)
        return images

    def forward(self, img, return_loss=False, return_encoded=False):
        """Forward pass through VQ-VAE."""
        encoded = self.encoder(img)

        if return_encoded:
            return encoded

        encoded = rearrange(encoded, 'b c t -> b t c')
        quantized, _, commit_loss = self.vq(encoded)
        quantized = rearrange(quantized, 'b t c -> b c t')
        out = self.decoder(quantized)

        if not return_loss:
            return out

        # Reconstruction loss + commitment loss
        recon_loss = F.mse_loss(img, out)

        return recon_loss + commit_loss
