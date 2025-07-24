#!/usr/bin/env python3
"""
Create expanded fine-tuning datasets with relaxed standard deviation thresholds
to get larger training sets for more robust fine-tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def create_expanded_datasets():
    """Create datasets with 1.5σ and 1σ thresholds for larger training sets"""
    
    # Load the filtered data (200-400 AA)
    input_file = "/Users/davidnunn/Desktop/Benchmark_Results/Annotated_Sequences_All_Filtered_Predictors.csv"
    print(f"Loading data from {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} sequences")
    
    # Filter by length (200-400 AA)
    df_filtered = df[(df['sequence_length'] >= 200) & 
                     (df['sequence_length'] <= 400)].copy()
    print(f"Length filtered: {len(df_filtered)} sequences")
    
    # Convert percentage strings to floats
    df_filtered['proline_pct'] = df_filtered['proline_content_calc'].str.rstrip('%').astype(float)
    df_filtered['rk_pct'] = df_filtered['rk_content'].str.rstrip('%').astype(float)
    df_filtered['wyfl_pct'] = df_filtered['wyfl_content'].str.rstrip('%').astype(float)
    
    # Calculate statistics
    proline_mean = df_filtered['proline_pct'].mean()
    proline_std = df_filtered['proline_pct'].std()
    rk_mean = df_filtered['rk_pct'].mean()
    rk_std = df_filtered['rk_pct'].std()
    wyfl_mean = df_filtered['wyfl_pct'].mean()
    wyfl_std = df_filtered['wyfl_pct'].std()
    
    print(f"\nBiophysical Feature Statistics:")
    print(f"Proline: {proline_mean:.2f}% ± {proline_std:.2f}%")
    print(f"R+K: {rk_mean:.2f}% ± {rk_std:.2f}%")
    print(f"WFYL: {wyfl_mean:.2f}% ± {wyfl_std:.2f}%")
    
    # Define thresholds for different sigma levels
    thresholds = {
        '2sigma': {
            'proline': proline_mean + 2 * proline_std,
            'rk': rk_mean + 2 * rk_std,
            'wyfl': wyfl_mean + 2 * wyfl_std
        },
        '1_5sigma': {
            'proline': proline_mean + 1.5 * proline_std,
            'rk': rk_mean + 1.5 * rk_std,
            'wyfl': wyfl_mean + 1.5 * wyfl_std
        },
        '1sigma': {
            'proline': proline_mean + 1 * proline_std,
            'rk': rk_mean + 1 * rk_std,
            'wyfl': wyfl_mean + 1 * wyfl_std
        }
    }
    
    # Create datasets for each threshold
    results = {}
    
    for sigma_level, thresh in thresholds.items():
        print(f"\n{'='*50}")
        print(f"Creating {sigma_level} datasets")
        print(f"{'='*50}")
        
        print(f"Thresholds - Proline: {thresh['proline']:.2f}%, R+K: {thresh['rk']:.2f}%, WFYL: {thresh['wyfl']:.2f}%")
        
        # Combined high-content dataset
        combined_high = df_filtered[
            (df_filtered['proline_pct'] >= thresh['proline']) |
            (df_filtered['rk_pct'] >= thresh['rk']) |
            (df_filtered['wyfl_pct'] >= thresh['wyfl'])
        ].copy()
        
        soluble_count = sum(combined_high['solubility'])
        print(f"\nCombined High-Content ({sigma_level}):")
        print(f"  Total sequences: {len(combined_high)}")
        print(f"  Soluble: {soluble_count} ({soluble_count/len(combined_high)*100:.1f}%)")
        print(f"  Insoluble: {len(combined_high)-soluble_count} ({(len(combined_high)-soluble_count)/len(combined_high)*100:.1f}%)")
        
        # Create train/test/val splits if dataset is large enough
        if len(combined_high) >= 30:  # Minimum for meaningful splits
            X = combined_high[['sequence_id', 'sequence', 'sequence_length', 'proline_pct', 'rk_pct', 'wyfl_pct']]
            y = combined_high['solubility']
            
            # First split: separate test set (15%)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
            
            # Second split: separate validation from remaining (15% of total)
            val_size_adjusted = 0.15 / (1 - 0.15)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
            )
            
            print(f"  Train: {len(X_train)} sequences ({sum(y_train)} soluble)")
            print(f"  Val:   {len(X_val)} sequences ({sum(y_val)} soluble)")
            print(f"  Test:  {len(X_test)} sequences ({sum(y_test)} soluble)")
            
            results[f'combined_high_{sigma_level}'] = {
                'total': len(combined_high),
                'train': len(X_train),
                'val': len(X_val),
                'test': len(X_test),
                'solubility_rate': soluble_count/len(combined_high)*100,
                'data': {
                    'train': {'X': X_train, 'y': y_train},
                    'val': {'X': X_val, 'y': y_val},
                    'test': {'X': X_test, 'y': y_test}
                }
            }
        else:
            print(f"  ⚠️  Dataset too small for splitting ({len(combined_high)} sequences)")
    
    return results

def save_expanded_datasets(results, output_dir="/Users/davidnunn/Desktop/Apps/PeptideFusionProject/PLM_Sol/fine_tuning_datasets"):
    """Save the expanded datasets"""
    output_path = Path(output_dir)
    
    for dataset_name, dataset_info in results.items():
        if 'data' not in dataset_info:
            continue
            
        dataset_dir = output_path / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving {dataset_name}...")
        
        for split_name, split_data in dataset_info['data'].items():
            X, y = split_data['X'], split_data['y']
            
            # Save as FASTA
            fasta_file = dataset_dir / f"{split_name}.fasta"
            with open(fasta_file, 'w') as f:
                for idx, row in X.iterrows():
                    label = int(y.loc[idx])
                    f.write(f">{row['sequence_id']}_label_{label}\n")
                    f.write(f"{row['sequence']}\n")
            
            # Save as CSV
            csv_file = dataset_dir / f"{split_name}.csv"
            combined_df = X.copy()
            combined_df['solubility'] = y
            combined_df.to_csv(csv_file, index=False)
            
            print(f"  {split_name.capitalize()}: {len(X)} sequences → {fasta_file.name}, {csv_file.name}")

def print_recommendations(results):
    """Print dataset size recommendations"""
    print(f"\n{'='*60}")
    print("📊 DATASET SIZE RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print("\n🎯 Fine-tuning Dataset Size Guidelines:")
    print("  • Minimum viable: ~200-500 training sequences")
    print("  • Good performance: ~500-1000 training sequences")
    print("  • Optimal: ~1000+ training sequences")
    
    print(f"\n📈 Available Dataset Options:")
    
    for dataset_name, info in results.items():
        if 'train' not in info:
            continue
            
        train_size = info['train']
        total_size = info['total']
        solubility_rate = info['solubility_rate']
        
        # Determine recommendation
        if train_size >= 1000:
            recommendation = "🟢 OPTIMAL"
        elif train_size >= 500:
            recommendation = "🟡 GOOD"
        elif train_size >= 200:
            recommendation = "🟠 VIABLE"
        else:
            recommendation = "🔴 TOO SMALL"
        
        print(f"\n  {dataset_name}:")
        print(f"    Total: {total_size} sequences")
        print(f"    Train: {train_size} sequences")
        print(f"    Solubility: {solubility_rate:.1f}%")
        print(f"    Status: {recommendation}")

def main():
    print("🔬 Creating Expanded Fine-Tuning Datasets")
    print("=" * 50)
    
    # Create expanded datasets
    results = create_expanded_datasets()
    
    # Save datasets
    save_expanded_datasets(results)
    
    # Print recommendations
    print_recommendations(results)
    
    print(f"\n🎉 Expanded dataset creation completed!")
    print(f"💡 Recommendation: Use the largest viable dataset for best fine-tuning results")

if __name__ == "__main__":
    main()
