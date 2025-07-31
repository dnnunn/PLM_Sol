#!/usr/bin/env python3
"""
Merged FASTA Header Fixer for PLM_Sol
-------------------------------------

Unifies and supersedes fix_fasta_headers.py and fix_fasta_headers_v2.py.
- Fixes headers in all fine-tuning or benchmarking FASTA files to match PLM_Sol parsing requirements.
- Handles all known legacy header formats.
- Can operate in batch mode (directory) or on a single file.

Usage:
  python fix_fasta_headers_merged.py --input <file_or_dir> [--mode auto|v1|v2]

Modes:
- auto: Detect and fix any known legacy format (default, recommended)
- v1: Only fix >seq_id_label_X → >seq_id description soluble-X
- v2: Only fix >seq_id protein sequence soluble-X → >seq_id description soluble-X

See --help for details.
"""

import os
from pathlib import Path
import re
import argparse

def fix_header(line, mode='auto'):
    """Fix a single FASTA header line according to the specified mode."""
    if not line.startswith('>'):
        return line
    # v2: >seq_id protein sequence soluble-X → >seq_id description soluble-X
    if mode in ('auto', 'v2'):
        match = re.match(r'>(.+) protein sequence soluble-(\d+)', line.strip())
        if match:
            seq_id, label = match.groups()
            return f">{seq_id} description soluble-{label}\n"
    # v1: >seq_id_label_X → >seq_id description soluble-X
    if mode in ('auto', 'v1'):
        match = re.match(r'>(.+)_label_(\d+)', line.strip())
        if match:
            seq_id, label = match.groups()
            return f">{seq_id} description soluble-{label}\n"
    # alt: >seq_id_solubility_X → >seq_id description soluble-X
    if mode == 'auto':
        match = re.match(r'>(.+)_solubility_(\d+)', line.strip())
        if match:
            seq_id, label = match.groups()
            return f">{seq_id} description soluble-{label}\n"
    # Already correct or unknown format
    return line

def fix_fasta_headers(fasta_file_path, mode='auto'):
    print(f"\ud83d\udd27 Fixing headers in {fasta_file_path} (mode: {mode})")
    with open(fasta_file_path, 'r') as f:
        lines = f.readlines()
    fixed_lines = [fix_header(line, mode=mode) for line in lines]
    with open(fasta_file_path, 'w') as f:
        f.writelines(fixed_lines)
    print(f"\u2705 Fixed {fasta_file_path}")

def batch_fix_headers(input_path, mode='auto'):
    path = Path(input_path)
    if path.is_file():
        fix_fasta_headers(path, mode=mode)
    elif path.is_dir():
        fasta_files = list(path.glob("*.fasta")) + list(path.glob("*/*.fasta"))
        if not fasta_files:
            print(f"\u274c No FASTA files found in {path}")
            return
        print(f"\ud83d\udcc1 Found {len(fasta_files)} FASTA files to fix:")
        for fasta_file in fasta_files:
            fix_fasta_headers(fasta_file, mode=mode)
        print(f"\n\ud83c\udf89 Completed fixing {len(fasta_files)} FASTA files!")
    else:
        print(f"\u274c Input path not found: {input_path}")

def main():
    parser = argparse.ArgumentParser(description="Merged FASTA Header Fixer for PLM_Sol")
    parser.add_argument('--input', required=True, help='FASTA file or directory to fix')
    parser.add_argument('--mode', choices=['auto', 'v1', 'v2'], default='auto', help='Header fix mode (default: auto)')
    args = parser.parse_args()
    batch_fix_headers(args.input, mode=args.mode)

if __name__ == "__main__":
    main()
