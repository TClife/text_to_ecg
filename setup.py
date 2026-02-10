from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="text-to-ecg",
    version="1.0.0",
    author="Hyunseung Chung et al.",
    author_email="",
    description="Text-to-ECG: Autoregressive ECG Generation from Clinical Text Reports",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TClife/text_to_ecg",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.0",
        "einops>=0.3.0",
        "tokenizers>=0.10.0",
        "transformers>=4.0.0",
        "rotary-embedding-torch>=0.2.0",
        "tqdm>=4.60.0",
        "omegaconf>=2.0.0",
        "librosa>=0.8.0",
        "scipy>=1.7.0",
        "ftfy>=6.0.0",
        "regex>=2021.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "wandb": ["wandb>=0.12.0"],
        "distributed": ["deepspeed>=0.5.0", "horovod>=0.22.0"],
    },
    entry_points={
        "console_scripts": [
            "train-vae=scripts.train_vae:main",
            "train-dalle=scripts.train_dalle:main",
            "train-hifi-gan=scripts.train_hifi_gan:main",
            "generate-ecg=scripts.generate:main",
        ],
    },
)
