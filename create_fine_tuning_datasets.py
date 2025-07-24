#!/usr/bin/env python3
"""
Create specialized fine-tuning datasets for PLM_Sol based on PeptideFrontEnd target characteristics:
1. Length filter: 200-400 amino acids (PeptideFrontEnd target range)
2. Biophysical enrichment: High proline, R+K, and WFYL content (2+ std from mean)
3. Train/Test/Validation splits: 70/15/15 split for proper fine-tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def load_annotated_data(file_path):
    """Load the annotated sequence data"""
    print(f"Loading annotated data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} sequences")
    return df

def filter_by_length(df, min_length=200, max_length=400):
    """Filter sequences to PeptideFrontEnd target length range"""
    print(f"\nFiltering sequences to {min_length}-{max_length} amino acids...")
    initial_count = len(df)
    
    df_filtered = df[(df['sequence_length'] >= min_length) & 
                     (df['sequence_length'] <= max_length)].copy()
    
    final_count = len(df_filtered)
    print(f"Length filtering: {initial_count} → {final_count} sequences ({final_count/initial_count*100:.1f}% retained)")
    
    return df_filtered

def analyze_biophysical_features(df):
    """Analyze distribution of biophysical features"""
    print("\n=== Biophysical Feature Analysis ===")
    
    features = ['proline_content_calc', 'rk_content', 'wyfl_content']
    stats = {}
    
    for feature in features:
        # Convert percentage strings to floats
        values = df[feature].str.rstrip('%').astype(float)
        
        mean_val = values.mean()
        std_val = values.std()
        
        stats[feature] = {
            'mean': mean_val,
            'std': std_val,
            'min': values.min(),
            'max': values.max(),
            'threshold_2std': mean_val + 2 * std_val
        }
        
        print(f"\n{feature.replace('_', ' ').title()}:")
        print(f"  Mean: {mean_val:.2f}%")
        print(f"  Std:  {std_val:.2f}%")
        print(f"  Range: {values.min():.2f}% - {values.max():.2f}%")
        print(f"  2σ threshold: {stats[feature]['threshold_2std']:.2f}%")
        
        # Count sequences above 2σ
        high_count = sum(values >= stats[feature]['threshold_2std'])
        print(f"  Sequences ≥2σ: {high_count} ({high_count/len(values)*100:.1f}%)")
    
    return stats

def create_enriched_datasets(df, stats):
    """Create datasets enriched for high proline, R+K, and WFYL content"""
    print("\n=== Creating Enriched Datasets ===")
    
    # Convert percentage strings to floats for all features
    df['proline_pct'] = df['proline_content_calc'].str.rstrip('%').astype(float)
    df['rk_pct'] = df['rk_content'].str.rstrip('%').astype(float)
    df['wyfl_pct'] = df['wyfl_content'].str.rstrip('%').astype(float)
    
    datasets = {}
    
    # High Proline Dataset (≥2σ from mean)
    proline_threshold = stats['proline_content_calc']['threshold_2std']
    high_proline = df[df['proline_pct'] >= proline_threshold].copy()
    datasets['high_proline'] = high_proline
    print(f"\nHigh Proline Dataset (≥{proline_threshold:.2f}%):")
    print(f"  Sequences: {len(high_proline)}")
    print(f"  Soluble: {sum(high_proline['solubility'])}")
    print(f"  Insoluble: {len(high_proline) - sum(high_proline['solubility'])}")
    print(f"  Solubility rate: {sum(high_proline['solubility'])/len(high_proline)*100:.1f}%")
    
    # High R+K Dataset (≥2σ from mean)
    rk_threshold = stats['rk_content']['threshold_2std']
    high_rk = df[df['rk_pct'] >= rk_threshold].copy()
    datasets['high_rk'] = high_rk
    print(f"\nHigh R+K Dataset (≥{rk_threshold:.2f}%):")
    print(f"  Sequences: {len(high_rk)}")
    print(f"  Soluble: {sum(high_rk['solubility'])}")
    print(f"  Insoluble: {len(high_rk) - sum(high_rk['solubility'])}")
    print(f"  Solubility rate: {sum(high_rk['solubility'])/len(high_rk)*100:.1f}%")
    
    # High WFYL Dataset (≥2σ from mean)
    wyfl_threshold = stats['wyfl_content']['threshold_2std']
    high_wyfl = df[df['wyfl_pct'] >= wyfl_threshold].copy()
    datasets['high_wyfl'] = high_wyfl
    print(f"\nHigh WFYL Dataset (≥{wyfl_threshold:.2f}%):")
    print(f"  Sequences: {len(high_wyfl)}")
    print(f"  Soluble: {sum(high_wyfl['solubility'])}")
    print(f"  Insoluble: {len(high_wyfl) - sum(high_wyfl['solubility'])}")
    print(f"  Solubility rate: {sum(high_wyfl['solubility'])/len(high_wyfl)*100:.1f}%")
    
    # Combined High-Content Dataset (any feature ≥2σ)
    combined_high = df[
        (df['proline_pct'] >= proline_threshold) |
        (df['rk_pct'] >= rk_threshold) |
        (df['wyfl_pct'] >= wyfl_threshold)
    ].copy()
    datasets['combined_high'] = combined_high
    print(f"\nCombined High-Content Dataset (any feature ≥2σ):")
    print(f"  Sequences: {len(combined_high)}")
    print(f"  Soluble: {sum(combined_high['solubility'])}")
    print(f"  Insoluble: {len(combined_high) - sum(combined_high['solubility'])}")
    print(f"  Solubility rate: {sum(combined_high['solubility'])/len(combined_high)*100:.1f}%")
    
    return datasets

def create_train_test_val_splits(datasets, test_size=0.15, val_size=0.15, random_state=42):
    """Create train/test/validation splits for each dataset"""
    print("\n=== Creating Train/Test/Validation Splits ===")
    
    splits = {}
    
    for dataset_name, df in datasets.items():
        print(f"\n{dataset_name.replace('_', ' ').title()} Dataset:")
        
        if len(df) < 10:
            print(f"  Warning: Only {len(df)} sequences - too small for splitting")
            continue
        
        # First split: separate test set
        X = df[['sequence_id', 'sequence', 'sequence_length', 'proline_pct', 'rk_pct', 'wyfl_pct']]
        y = df['solubility']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate validation from remaining
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        splits[dataset_name] = {
            'train': {'X': X_train, 'y': y_train},
            'val': {'X': X_val, 'y': y_val},
            'test': {'X': X_test, 'y': y_test}
        }
        
        print(f"  Train: {len(X_train)} sequences ({sum(y_train)} soluble, {len(y_train)-sum(y_train)} insoluble)")
        print(f"  Val:   {len(X_val)} sequences ({sum(y_val)} soluble, {len(y_val)-sum(y_val)} insoluble)")
        print(f"  Test:  {len(X_test)} sequences ({sum(y_test)} soluble, {len(y_test)-sum(y_test)} insoluble)")
        
        # Check class balance
        train_balance = sum(y_train) / len(y_train) * 100
        val_balance = sum(y_val) / len(y_val) * 100
        test_balance = sum(y_test) / len(y_test) * 100
        
        print(f"  Solubility rates - Train: {train_balance:.1f}%, Val: {val_balance:.1f}%, Test: {test_balance:.1f}%")
    
    return splits

def save_datasets(splits, output_dir):
    """Save datasets in FASTA and CSV formats for fine-tuning"""
    print(f"\n=== Saving Datasets to {output_dir} ===")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, dataset_splits in splits.items():
        dataset_dir = output_path / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"\n{dataset_name.replace('_', ' ').title()}:")
        
        for split_name, split_data in dataset_splits.items():
            X, y = split_data['X'], split_data['y']
            
            # Save as FASTA for PLM_Sol training with correct header format
            fasta_file = dataset_dir / f"{split_name}.fasta"
            with open(fasta_file, 'w') as f:
                for idx, row in X.iterrows():
                    # PLM_Sol expects: >id description field1 field2-solubility_label
                    label = int(y.loc[idx])
                    f.write(f">{row['sequence_id']} protein sequence soluble-{label}\n")
                    f.write(f"{row['sequence']}\n")
            
            # Save as CSV for analysis
            csv_file = dataset_dir / f"{split_name}.csv"
            combined_df = X.copy()
            combined_df['solubility'] = y
            combined_df.to_csv(csv_file, index=False)
            
            print(f"  {split_name.capitalize()}: {len(X)} sequences → {fasta_file.name}, {csv_file.name}")
    
    print(f"\nAll datasets saved to: {output_path}")
    return output_path

def create_summary_report(df_original, df_filtered, datasets, splits, stats, output_dir):
    """Create a comprehensive summary report"""
    report_file = Path(output_dir) / "dataset_summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("PLM_Sol Fine-Tuning Dataset Creation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write(f"Original sequences: {len(df_original)}\n")
        f.write(f"Length filtered (200-400 AA): {len(df_filtered)}\n")
        f.write(f"Retention rate: {len(df_filtered)/len(df_original)*100:.1f}%\n\n")
        
        f.write("BIOPHYSICAL FEATURE STATISTICS:\n")
        for feature, stat in stats.items():
            f.write(f"{feature.replace('_', ' ').title()}:\n")
            f.write(f"  Mean ± Std: {stat['mean']:.2f}% ± {stat['std']:.2f}%\n")
            f.write(f"  2σ threshold: {stat['threshold_2std']:.2f}%\n\n")
        
        f.write("ENRICHED DATASETS:\n")
        for name, df in datasets.items():
            soluble_count = sum(df['solubility'])
            f.write(f"{name.replace('_', ' ').title()}:\n")
            f.write(f"  Total sequences: {len(df)}\n")
            f.write(f"  Soluble: {soluble_count} ({soluble_count/len(df)*100:.1f}%)\n")
            f.write(f"  Insoluble: {len(df)-soluble_count} ({(len(df)-soluble_count)/len(df)*100:.1f}%)\n\n")
        
        f.write("TRAIN/TEST/VALIDATION SPLITS:\n")
        for dataset_name, dataset_splits in splits.items():
            f.write(f"{dataset_name.replace('_', ' ').title()}:\n")
            for split_name, split_data in dataset_splits.items():
                X, y = split_data['X'], split_data['y']
                soluble = sum(y)
                f.write(f"  {split_name.capitalize()}: {len(X)} sequences ({soluble} soluble, {len(y)-soluble} insoluble)\n")
            f.write("\n")
        
        f.write("USAGE RECOMMENDATIONS:\n")
        f.write("1. Start with 'combined_high' dataset for general fine-tuning\n")
        f.write("2. Use individual datasets (high_proline, high_rk, high_wyfl) for specialized fine-tuning\n")
        f.write("3. All datasets are filtered to 200-400 AA range matching PeptideFrontEnd targets\n")
        f.write("4. FASTA files include solubility labels in headers for training\n")
        f.write("5. CSV files provide detailed analysis and feature information\n")
    
    print(f"\nSummary report saved to: {report_file}")

def main():
    # Configuration
    input_file = "/Users/davidnunn/Desktop/Benchmark_Results/Annotated_Sequences_All_Filtered_Predictors.csv"
    output_dir = "/Users/davidnunn/Desktop/Apps/PeptideFusionProject/PLM_Sol/fine_tuning_datasets"
    
    # Load and process data
    df_original = load_annotated_data(input_file)
    df_filtered = filter_by_length(df_original, min_length=200, max_length=400)
    
    # Analyze biophysical features
    stats = analyze_biophysical_features(df_filtered)
    
    # Create enriched datasets
    datasets = create_enriched_datasets(df_filtered, stats)
    
    # Create train/test/validation splits
    splits = create_train_test_val_splits(datasets)
    
    # Save datasets
    output_path = save_datasets(splits, output_dir)
    
    # Create summary report
    create_summary_report(df_original, df_filtered, datasets, splits, stats, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Fine-tuning dataset creation completed!")
    print(f"📁 Output directory: {output_path}")
    print("🚀 Ready for PLM_Sol fine-tuning experiments!")

if __name__ == "__main__":
    main()
