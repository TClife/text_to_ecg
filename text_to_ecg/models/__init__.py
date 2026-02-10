"""
Text-to-ECG Models

This module contains all the neural network architectures used in the
Text-to-ECG pipeline:

- VQ-VAE: For encoding ECG signals into discrete tokens
- DALLE: Autoregressive transformer for text-to-ECG generation
- HiFi-GAN: Neural vocoder for high-fidelity ECG reconstruction
"""

from text_to_ecg.models.vae import DiscreteVAE, VQVAE
from text_to_ecg.models.dalle import DALLE, DALLE_concat, CLIP
from text_to_ecg.models.transformer import Transformer
from text_to_ecg.models.attention import Attention, SparseAttention
from text_to_ecg.models.hifi_gan import CodeGenerator, Generator
from text_to_ecg.models.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)

__all__ = [
    # VAE
    "DiscreteVAE",
    "VQVAE",
    # DALLE
    "DALLE",
    "DALLE_concat",
    "CLIP",
    # Transformer
    "Transformer",
    "Attention",
    "SparseAttention",
    # HiFi-GAN
    "CodeGenerator",
    "Generator",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "feature_loss",
    "generator_loss",
    "discriminator_loss",
]
