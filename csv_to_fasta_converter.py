#!/usr/bin/env python3
"""
CSV to FASTA Converter for PLM_Sol Fine-Tuning
----------------------------------------------

Converts existing CSV files (train.csv, val.csv, test.csv) to FASTA format
with proper headers for PLM_Sol fine-tuning pipeline.

Usage:
  python csv_to_fasta_converter.py --dataset combined_high_1_5sigma
  python csv_to_fasta_converter.py --dataset combined_high_1_5sigma --splits train val test
  python csv_to_fasta_converter.py --input_dir /path/to/datasets --dataset dataset_name

Output:
  Creates train.fasta, val.fasta, test.fasta in the dataset directory
  Headers format: >seq_id description soluble-{0|1}
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

def csv_to_fasta(csv_file, output_fasta):
    """Convert CSV file to FASTA format for PLM_Sol fine-tuning"""
    print(f"🔄 Converting {csv_file} to {output_fasta}")
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} sequences from CSV")
        
        # Check required columns
        required_cols = ['sequence', 'solubility']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return False
        
        with open(output_fasta, 'w') as f:
            for idx, row in df.iterrows():
                # Extract sequence ID (or create one)
                if 'sequence_id' in df.columns and pd.notna(row['sequence_id']):
                    seq_id = str(row['sequence_id'])
                elif 'id' in df.columns and pd.notna(row['id']):
                    seq_id = str(row['id'])
                else:
                    seq_id = f'seq_{idx}'
                
                sequence = str(row['sequence'])
                solubility = int(row['solubility'])
                
                # PLM_Sol expected format: >seq_id description soluble-{0|1}
                header = f">{seq_id} description soluble-{solubility}"
                f.write(f"{header}\n{sequence}\n")
        
        print(f"✅ Created {output_fasta} with {len(df)} sequences")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {csv_file}: {e}")
        return False

def convert_dataset(dataset_dir, splits=['train', 'val', 'test']):
    """Convert all CSV files in a dataset directory to FASTA format"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    print(f"🎯 Converting dataset: {dataset_path.name}")
    print(f"📁 Dataset directory: {dataset_path}")
    
    success_count = 0
    total_count = 0
    
    for split in splits:
        csv_file = dataset_path / f"{split}.csv"
        fasta_file = dataset_path / f"{split}.fasta"
        
        total_count += 1
        if csv_to_fasta(csv_file, fasta_file):
            success_count += 1
    
    print(f"\n📋 Conversion Summary:")
    print(f"✅ Successful: {success_count}/{total_count}")
    
    if success_count == total_count:
        print(f"🎉 All files converted successfully!")
        print(f"📁 FASTA files created in: {dataset_path}")
        return True
    else:
        print(f"⚠️  Some conversions failed")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to FASTA format for PLM_Sol fine-tuning")
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., combined_high_1_5sigma)')
    parser.add_argument('--input_dir', help='Base directory containing datasets (default: ./fine_tuning_datasets)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], 
                        help='Splits to convert (default: train val test)')
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.input_dir:
        base_dir = Path(args.input_dir)
    else:
        # Default to fine_tuning_datasets in current directory
        base_dir = Path.cwd() / "fine_tuning_datasets"
    
    dataset_dir = base_dir / args.dataset
    
    print("🔧 CSV to FASTA Converter for PLM_Sol Fine-Tuning")
    print("=" * 55)
    print(f"🎯 Dataset: {args.dataset}")
    print(f"📁 Directory: {dataset_dir}")
    print(f"📋 Splits: {args.splits}")
    print()
    
    success = convert_dataset(dataset_dir, args.splits)
    
    if success:
        print(f"\n🚀 Ready for fine-tuning!")
        print(f"Run: python fine_tune_plm_sol.py --datasets {args.dataset} --force-regenerate")
        sys.exit(0)
    else:
        print(f"\n❌ Conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
