#!/usr/bin/env python3
"""
Unified Dataset Creation Script for PLM_Sol Fine-Tuning
------------------------------------------------------

This script merges and supersedes both create_fine_tuning_datasets.py and create_expanded_datasets.py.
It allows creation of:
- Standard fine-tuning datasets (biophysical enrichment, 2σ, 1.5σ, 1σ, custom splits)
- Expanded datasets with relaxed thresholds
- Train/test/val splits and FASTA/CSV outputs

Usage:
  python dataset_creation.py --mode [fine_tuning|expanded] [additional options]

See --help for details.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# --- Utility Functions ---
def load_annotated_data(file_path):
    print(f"Loading annotated data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} sequences")
    return df

def filter_by_length(df, min_length=200, max_length=400):
    print(f"Filtering sequences to {min_length}-{max_length} AA...")
    return df[(df['sequence_length'] >= min_length) & (df['sequence_length'] <= max_length)].copy()

def analyze_biophysical_features(df):
    features = ['proline_content_calc', 'rk_content', 'wyfl_content']
    stats = {}
    for feature in features:
        values = df[feature].str.rstrip('%').astype(float)
        stats[feature] = {'mean': values.mean(), 'std': values.std()}
    return stats

def compute_thresholds(stats, sigmas=[2, 1.5, 1]):
    thresholds = {}
    for sigma in sigmas:
        thresholds[f'{sigma}sigma'] = {
            'proline': stats['proline_content_calc']['mean'] + sigma * stats['proline_content_calc']['std'],
            'rk': stats['rk_content']['mean'] + sigma * stats['rk_content']['std'],
            'wyfl': stats['wyfl_content']['mean'] + sigma * stats['wyfl_content']['std']
        }
    return thresholds

def create_enriched_datasets(df, thresholds, sigma_level='2sigma'):
    thresh = thresholds[sigma_level]
    df['proline_pct'] = df['proline_content_calc'].str.rstrip('%').astype(float)
    df['rk_pct'] = df['rk_content'].str.rstrip('%').astype(float)
    df['wyfl_pct'] = df['wyfl_content'].str.rstrip('%').astype(float)
    datasets = {}
    # Combined high-content
    datasets['combined_high'] = df[(df['proline_pct'] >= thresh['proline']) |
                                   (df['rk_pct'] >= thresh['rk']) |
                                   (df['wyfl_pct'] >= thresh['wyfl'])].copy()
    # Individual
    datasets['high_proline'] = df[df['proline_pct'] >= thresh['proline']].copy()
    datasets['high_rk'] = df[df['rk_pct'] >= thresh['rk']].copy()
    datasets['high_wyfl'] = df[df['wyfl_pct'] >= thresh['wyfl']].copy()
    return datasets

def create_train_test_val_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    X = df[['sequence_id', 'sequence', 'sequence_length', 'proline_pct', 'rk_pct', 'wyfl_pct']]
    y = df['solubility']
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp)
    return {'train': {'X': X_train, 'y': y_train}, 'val': {'X': X_val, 'y': y_val}, 'test': {'X': X_test, 'y': y_test}}

def save_datasets(splits, output_dir, prefix):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    for split_name, split_data in splits.items():
        X, y = split_data['X'], split_data['y']
        fasta_file = output_path / f"{prefix}_{split_name}.fasta"
        with open(fasta_file, 'w') as f:
            for idx, row in X.iterrows():
                label = int(y.loc[idx])
                f.write(f">{row['sequence_id']} description soluble-{label}\n{row['sequence']}\n")
        csv_file = output_path / f"{prefix}_{split_name}.csv"
        combined_df = X.copy()
        combined_df['solubility'] = y
        combined_df.to_csv(csv_file, index=False)
        print(f"  {split_name.capitalize()}: {len(X)} sequences → {fasta_file.name}, {csv_file.name}")

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Unified PLM_Sol Dataset Creation Script")
    parser.add_argument('--mode', choices=['fine_tuning', 'expanded'], required=True, help='Which dataset creation workflow to run')
    parser.add_argument('--input', required=True, help='Path to annotated sequence CSV')
    parser.add_argument('--output_dir', required=True, help='Output directory for datasets')
    parser.add_argument('--sigma', type=float, default=2.0, help='Sigma threshold for enrichment (2, 1.5, 1)')
    parser.add_argument('--min_length', type=int, default=200, help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=400, help='Maximum sequence length')
    args = parser.parse_args()

    df = load_annotated_data(args.input)
    df = filter_by_length(df, min_length=args.min_length, max_length=args.max_length)
    stats = analyze_biophysical_features(df)
    thresholds = compute_thresholds(stats, sigmas=[2, 1.5, 1])

    if args.mode == 'fine_tuning':
        # Standard fine-tuning datasets (2σ by default)
        sigma_level = f"{args.sigma}sigma" if args.sigma in [2, 1.5, 1] else '2sigma'
        datasets = create_enriched_datasets(df, thresholds, sigma_level=sigma_level)
        for name, dset in datasets.items():
            if len(dset) < 30:
                print(f"Skipping {name}: too few sequences ({len(dset)})")
                continue
            splits = create_train_test_val_splits(dset)
            save_datasets(splits, args.output_dir, prefix=name + '_' + sigma_level)
    elif args.mode == 'expanded':
        # Expanded datasets: run for all sigma levels
        for sigma_level in thresholds:
            print(f"\nCreating datasets for {sigma_level}")
            datasets = create_enriched_datasets(df, thresholds, sigma_level=sigma_level)
            for name, dset in datasets.items():
                if len(dset) < 30:
                    print(f"Skipping {name}: too few sequences ({len(dset)})")
                    continue
                splits = create_train_test_val_splits(dset)
                save_datasets(splits, args.output_dir, prefix=name + '_' + sigma_level)
    print("\n✅ Dataset creation complete.")

if __name__ == "__main__":
    main()
