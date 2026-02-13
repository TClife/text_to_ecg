#!/usr/bin/env python
"""
Prepare PTB-XL dataset for text-to-ECG training pipeline.

Reads the translated PTB-XL CSV and WFDB records, splits by strat_fold,
saves .pt files for each stage, and trains a BPE tokenizer on train reports.

Usage:
    python scripts/prepare_ptbxl.py \
        --csv_path ./ptbxl_database_translated.csv \
        --wfdb_root /path/to/ptb-xl-1.0.1 \
        --output_dir ./data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare PTB-XL data')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to ptbxl_database_translated.csv')
    parser.add_argument('--wfdb_root', type=str, required=True,
                        help='Root directory of PTB-XL (containing records500/)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for .pt files')
    parser.add_argument('--tokenizer_output', type=str,
                        default='./configs/ptbxl_tokenizer.json',
                        help='Output path for BPE tokenizer')
    parser.add_argument('--vocab_size', type=int, default=3000,
                        help='BPE vocabulary size')
    return parser.parse_args()


def load_ecg_record(wfdb_root, filename_hr):
    """Load a single WFDB record and return (12, 5000) float32 tensor."""
    record_path = str(Path(wfdb_root) / filename_hr)
    record = wfdb.rdrecord(record_path)
    # record.p_signal is (5000, 12), transpose to (12, 5000)
    signal = record.p_signal.T.astype(np.float32)
    return torch.from_numpy(signal)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.tokenizer_output).parent.mkdir(parents=True, exist_ok=True)

    # Read CSV
    print(f"Reading CSV from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Total records in CSV: {len(df)}")

    # Clean data
    # Cap age at 100 (some entries have unrealistic values)
    df['age'] = df['age'].clip(upper=100)
    # Fill missing age with median
    df['age'] = df['age'].fillna(df['age'].median())
    # PTB-XL sex: 0=male, 1=female; codebase: 0=female, 1=male → flip
    df['gender'] = 1 - df['sex']
    # Drop rows with empty reports
    df = df.dropna(subset=['report'])
    df = df[df['report'].str.strip().astype(bool)]
    print(f"Records after cleaning: {len(df)}")

    # Load all ECG records
    print("Loading ECG records...")
    ecg_data = []
    valid_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            ecg = load_ecg_record(args.wfdb_root, row['filename_hr'])
            if ecg.shape == (12, 5000):
                ecg_data.append(ecg)
                valid_indices.append(idx)
            else:
                print(f"  Skipping {row['filename_hr']}: unexpected shape {ecg.shape}")
        except Exception as e:
            print(f"  Error loading {row['filename_hr']}: {e}")

    df = df.loc[valid_indices].reset_index(drop=True)
    print(f"Successfully loaded {len(ecg_data)} ECG records")

    # Split by strat_fold: 1-8 → train, 9 → valid, 10 → test
    splits = {
        'train': df['strat_fold'].isin(range(1, 9)),
        'valid': df['strat_fold'] == 9,
        'test': df['strat_fold'] == 10,
    }

    for split_name, mask in splits.items():
        split_df = df[mask]
        split_indices = split_df.index.tolist()

        split_data = {
            'data': [ecg_data[i] for i in split_indices],
            'report_data': [split_df.loc[i, 'report'] for i in split_indices],
            'age': torch.tensor([split_df.loc[i, 'age'] for i in split_indices], dtype=torch.float32),
            'gender': torch.tensor([split_df.loc[i, 'gender'] for i in split_indices], dtype=torch.float32),
        }

        out_path = output_dir / f'ptbxl_{split_name}.pt'
        torch.save(split_data, out_path)
        print(f"Saved {split_name}: {len(split_indices)} records → {out_path}")

    # Save ECG-only list for VQ-VAE training (train split)
    train_mask = splits['train']
    train_indices = df[train_mask].index.tolist()
    ecg_train_list = [ecg_data[i] for i in train_indices]
    ecg_train_path = output_dir / 'ptbxl_ecg_train.pt'
    torch.save(ecg_train_list, ecg_train_path)
    print(f"Saved ECG-only train data: {len(ecg_train_list)} records → {ecg_train_path}")

    # Train BPE tokenizer on train reports
    print("\nTraining BPE tokenizer...")
    train_reports = [df.loc[i, 'report'] for i in train_indices]
    train_tokenizer(train_reports, args.tokenizer_output, args.vocab_size)
    print(f"Tokenizer saved to {args.tokenizer_output}")

    # Print summary
    print("\n=== Summary ===")
    for split_name, mask in splits.items():
        print(f"  {split_name}: {mask.sum()} records")
    print(f"  ECG shape: (12, 5000)")
    print(f"  Tokenizer vocab size: {args.vocab_size}")


def train_tokenizer(texts, output_path, vocab_size):
    """Train a HuggingFace BPE tokenizer on the given texts."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import ByteLevel

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
    )

    # Train from iterator
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = ByteLevel(trim_offsets=True)

    tokenizer.save(output_path)

    # Verify
    loaded = Tokenizer.from_file(output_path)
    test_text = "sinus rhythm normal ecg"
    encoded = loaded.encode(test_text)
    print(f"  Test: '{test_text}' → {encoded.ids} ({len(encoded.ids)} tokens)")
    print(f"  Vocab size: {loaded.get_vocab_size()}")


if __name__ == '__main__':
    main()
