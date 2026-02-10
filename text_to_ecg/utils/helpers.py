"""
Helper functions and utilities for Text-to-ECG.
"""

import os
import glob
import shutil

import torch
from torch.nn.utils import weight_norm


def exists(val):
    """Check if value is not None."""
    return val is not None


def default(val, d):
    """Return value if it exists, otherwise return default."""
    return val if exists(val) else d


def load_checkpoint(filepath, device='cpu'):
    """Load a checkpoint from file.

    Args:
        filepath: Path to checkpoint file
        device: Device to load checkpoint to

    Returns:
        Checkpoint dictionary
    """
    assert os.path.isfile(filepath), f"Checkpoint not found: {filepath}"
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    """Save a checkpoint to file.

    Args:
        filepath: Path to save checkpoint
        obj: Object to save
    """
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    """Scan for the latest checkpoint with a given prefix.

    Args:
        cp_dir: Checkpoint directory
        prefix: Checkpoint file prefix

    Returns:
        Path to latest checkpoint or None
    """
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def build_env(config, config_name, path):
    """Copy config file to output directory.

    Args:
        config: Source config path
        config_name: Target config filename
        path: Output directory path
    """
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def init_weights(m, mean=0.0, std=0.01):
    """Initialize weights for convolutional layers.

    Args:
        m: Module to initialize
        mean: Mean for normal distribution
        std: Standard deviation for normal distribution
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    """Apply weight normalization to convolutional layers.

    Args:
        m: Module to apply weight norm to
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    """Calculate padding for 'same' convolution.

    Args:
        kernel_size: Convolution kernel size
        dilation: Convolution dilation

    Returns:
        Padding size
    """
    return int((kernel_size * dilation - dilation) / 2)


class AttrDict(dict):
    """Dictionary that allows attribute-style access."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def set_requires_grad(model, value):
    """Set requires_grad for all parameters in a model.

    Args:
        model: PyTorch model
        value: Boolean value for requires_grad
    """
    for param in model.parameters():
        param.requires_grad = value


def eval_decorator(fn):
    """Decorator that sets model to eval mode during function execution."""
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def top_k(logits, thres=0.5):
    """Apply top-k filtering to logits.

    Args:
        logits: Input logits tensor
        thres: Threshold for keeping top k logits (1 - thres = k fraction)

    Returns:
        Filtered logits with non-top-k set to -inf
    """
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs
