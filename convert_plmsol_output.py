#!/usr/bin/env python
"""
Convert PLM_Sol output to a format compatible with the benchmarking system.
Maps MD5 hashes back to original sequence identifiers using the remapped sequences file.
"""
import os
import sys
import pandas as pd
import hashlib
from Bio import SeqIO

def md5_hash(seq):
    """Calculate MD5 hash of a sequence string."""
    return hashlib.md5(seq.encode()).hexdigest()

def convert_output(plmsol_output_path, fasta_path, output_path):
    """
    Convert PLM_Sol output to benchmark format.
    
    Args:
        plmsol_output_path: Path to PLM_Sol output CSV
        fasta_path: Path to original FASTA file with sequence identifiers
        output_path: Path to save the converted output
    """
    print(f"Converting PLM_Sol output from {plmsol_output_path}")
    
    # Read PLM_Sol output
    plmsol_df = pd.read_csv(plmsol_output_path)
    
    # Read original FASTA sequences
    original_seqs = {}
    seq_to_id = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        original_seqs[record.id] = str(record.seq)
        seq_to_id[str(record.seq)] = record.id
    
    # Map sequences or hashes back to original IDs
    id_map = {}
    for hash_id, seq in zip(plmsol_df['protein_ID'], plmsol_df['sequence']):
        # Try direct sequence match first
        if seq in seq_to_id:
            id_map[hash_id] = seq_to_id[seq]
        else:
            # Try matching by computing hash of original sequences
            for orig_id, orig_seq in original_seqs.items():
                if hash_id == md5_hash(orig_seq):
                    id_map[hash_id] = orig_id
                    break
    
    # Create new dataframe with original IDs
    result_df = pd.DataFrame({
        'sequence_id': [id_map.get(hash_id, hash_id) for hash_id in plmsol_df['protein_ID']],
        'solubility_score': plmsol_df['predict_result']
    })
    
    # Save to output file
    result_df.to_csv(output_path, index=False)
    print(f"✓ Converted output saved to {output_path}")
    print(f"  Mapped {len([v for v in id_map.values() if v])}/{len(plmsol_df)} protein IDs to original identifiers")
    
    return result_df

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convert_plmsol_output.py <plmsol_output.csv> <original.fasta> <output.csv>")
        sys.exit(1)
        
    plmsol_output_path = sys.argv[1]
    fasta_path = sys.argv[2]
    output_path = sys.argv[3]
    
    convert_output(plmsol_output_path, fasta_path, output_path)
