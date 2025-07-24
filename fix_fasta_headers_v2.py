#!/usr/bin/env python3
"""
Fix FASTA headers in existing fine-tuning datasets to match PLM_Sol parsing expectations.

PLM_Sol parsing logic: record.description.split(' ')[2].split('-')[-1]
This means the solubility label must be in the THIRD space-separated field, after a dash.

Current format: >seq_50631 protein sequence soluble-0
Problem: split(' ')[2] = "sequence", split('-')[-1] = "sequence" (no dash)

Target format:  >seq_50631 description soluble-0
Result: split(' ')[2] = "soluble-0", split('-')[-1] = "0" ✓
"""

import os
from pathlib import Path
import re

def fix_fasta_headers(fasta_file_path):
    """Fix FASTA headers in a single file to match PLM_Sol parsing logic."""
    print(f"🔧 Fixing headers in {fasta_file_path}")
    
    # Read original file
    with open(fasta_file_path, 'r') as f:
        lines = f.readlines()
    
    # Process lines
    fixed_lines = []
    for line in lines:
        if line.startswith('>'):
            # Extract sequence ID and label from current format
            # Current: >seq_50631 protein sequence soluble-0
            # Target:  >seq_50631 description soluble-0
            match = re.match(r'>(.+) protein sequence soluble-(\d+)', line.strip())
            if match:
                seq_id = match.group(1)
                label = match.group(2)
                new_header = f">{seq_id} description soluble-{label}\n"
                fixed_lines.append(new_header)
                print(f"  Fixed: {line.strip()} -> {new_header.strip()}")
            else:
                # Try alternative patterns or keep original
                alt_match = re.match(r'>(.+)_label_(\d+)', line.strip())
                if alt_match:
                    seq_id = alt_match.group(1)
                    label = alt_match.group(2)
                    new_header = f">{seq_id} description soluble-{label}\n"
                    fixed_lines.append(new_header)
                    print(f"  Fixed: {line.strip()} -> {new_header.strip()}")
                else:
                    # Keep original if pattern doesn't match
                    fixed_lines.append(line)
                    print(f"  Kept: {line.strip()}")
        else:
            # Keep sequence lines unchanged
            fixed_lines.append(line)
    
    # Write fixed file
    with open(fasta_file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fixed {fasta_file_path}")

def main():
    """Fix FASTA headers in all fine-tuning datasets."""
    print("🔧 Fixing FASTA Headers for PLM_Sol Fine-Tuning (v2)")
    print("=" * 55)
    
    # Base directory for fine-tuning datasets (local path)
    base_dir = Path("/Users/davidnunn/Desktop/Apps/PeptideFusionProject/PLM_Sol/fine_tuning_datasets")
    
    if not base_dir.exists():
        print(f"❌ Dataset directory not found: {base_dir}")
        return
    
    # Find all FASTA files in dataset directories
    fasta_files = list(base_dir.glob("*/*.fasta"))
    
    if not fasta_files:
        print(f"❌ No FASTA files found in {base_dir}")
        return
    
    print(f"📁 Found {len(fasta_files)} FASTA files to fix:")
    for fasta_file in fasta_files:
        print(f"  - {fasta_file}")
    
    print("\n🔧 Starting header fixes...")
    print("Target format: >seq_id description soluble-0")
    print("PLM_Sol parsing: split(' ')[2].split('-')[-1] should extract '0' or '1'")
    
    # Fix each FASTA file
    for fasta_file in fasta_files:
        try:
            fix_fasta_headers(fasta_file)
        except Exception as e:
            print(f"❌ Error fixing {fasta_file}: {e}")
    
    print(f"\n🎉 Completed fixing {len(fasta_files)} FASTA files!")
    print("Ready to rerun fine-tuning with correct header parsing.")

if __name__ == "__main__":
    main()
