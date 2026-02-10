"""
Text-to-ECG: Autoregressive ECG Generation from Clinical Text Reports

This package provides models and utilities for generating 12-lead ECG signals
from clinical text reports using an autoregressive transformer architecture.
"""

from text_to_ecg.models import (
    DiscreteVAE,
    VQVAE,
    DALLE,
    DALLE_concat,
    Transformer,
    CodeGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from text_to_ecg.data import (
    TextImageDataset,
    HugTokenizer,
    SimpleTokenizer,
)

__version__ = "1.0.0"
__all__ = [
    # Models
    "DiscreteVAE",
    "VQVAE",
    "DALLE",
    "DALLE_concat",
    "Transformer",
    "CodeGenerator",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    # Data
    "TextImageDataset",
    "HugTokenizer",
    "SimpleTokenizer",
]
