#!/usr/bin/env python3
"""
convert_fasta_headers_high_proline.py

Converts FASTA headers in train/val/test FASTA files for the high-proline dataset to the PLM_Sol-compatible format:
>seq_id something soluble-0
or
>seq_id something soluble-1

Usage:
    python convert_fasta_headers_high_proline.py --input train.fasta --output train.fixed.fasta
    (repeat for val.fasta and test.fasta)

This script assumes the solubility label is present in the header or can be inferred from the sequence record description.
"""
import argparse
import re

def extract_label_from_header(header):
    # Try to find 'soluble-0' or 'soluble-1' in the header
    match = re.search(r'soluble-([01])', header)
    if match:
        return f'soluble-{match.group(1)}'
    # Otherwise, try to infer from other conventions (customize as needed)
    # Default to unknown if not found
    return 'soluble-0'  # or raise an error if strict

def fix_header(original_header):
    # Remove '>' and split by whitespace
    parts = original_header[1:].split()
    seq_id = parts[0]
    # Optionally, add a placeholder or keep any extra info
    extra = ' '.join(parts[1:-1]) if len(parts) > 2 else ''
    label = extract_label_from_header(original_header)
    # Compose new header
    new_header = f'>{seq_id} {extra} {label}'.strip()
    return new_header

def convert_fasta_headers(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.startswith('>'):
                fixed = fix_header(line.strip())
                outfile.write(fixed + '\n')
            else:
                outfile.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix high-proline FASTA headers for PLM_Sol training.')
    parser.add_argument('--input', required=True, help='Input FASTA file')
    parser.add_argument('--output', required=True, help='Output FASTA file with fixed headers')
    args = parser.parse_args()
    convert_fasta_headers(args.input, args.output)
