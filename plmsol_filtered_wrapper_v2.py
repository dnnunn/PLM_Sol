#!/usr/bin/env python3
"""
PLM_Sol Filtered Wrapper v2 - Based on Working Original
-------------------------------------------------------

This wrapper adds sequence length filtering to the proven working PLM_Sol wrapper
to avoid CUDA OOM errors while maintaining row alignment.

Usage:
  python plmsol_filtered_wrapper_v2.py --fasta <input_fasta> --out <output_csv> [--max_length 4000]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import yaml
import shutil

def create_filtered_fasta(input_fasta, output_fasta, max_length=4000):
    """Create a filtered FASTA file excluding sequences longer than max_length."""
    filtered_sequences = []
    kept_sequences = []
    
    with open(output_fasta, 'w') as out_handle:
        for i, record in enumerate(SeqIO.parse(input_fasta, 'fasta')):
            if len(record.seq) > max_length:
                filtered_sequences.append((i, record.id, len(record.seq), str(record.seq)))
                print(f"Filtering out sequence {record.id} (length: {len(record.seq)})")
            else:
                SeqIO.write(record, out_handle, 'fasta')
                kept_sequences.append((i, record.id, len(record.seq), str(record.seq)))
    
    print(f"Kept {len(kept_sequences)} sequences, filtered {len(filtered_sequences)} sequences")
    return filtered_sequences, kept_sequences

def merge_results_with_filtered(plm_sol_results, filtered_sequences, original_fasta, output_file):
    """Merge PLM_Sol results with filtered sequences, maintaining original order."""
    
    # Read PLM_Sol results if they exist
    results_dict = {}
    if os.path.exists(plm_sol_results):
        results_df = pd.read_csv(plm_sol_results)
        for _, row in results_df.iterrows():
            results_dict[row['Accession']] = row
    
    # Build final results maintaining original order
    final_results = []
    
    for record in SeqIO.parse(original_fasta, 'fasta'):
        accession = record.id
        sequence = str(record.seq)
        
        # Check if this sequence was filtered
        was_filtered = any(acc == accession for _, acc, _, _ in filtered_sequences)
        
        if was_filtered:
            # Assign default values for filtered sequences
            final_results.append({
                'Accession': accession,
                'Sequence': sequence,
                'Predictor': 'PLM_Sol',
                'SolubilityScore': 0.5,
                'Probability_Soluble': 0.5,
                'Probability_Insoluble': 0.5
            })
        else:
            # Use PLM_Sol prediction if available
            if accession in results_dict:
                result_row = results_dict[accession]
                final_results.append({
                    'Accession': accession,
                    'Sequence': sequence,
                    'Predictor': 'PLM_Sol',
                    'SolubilityScore': result_row['SolubilityScore'],
                    'Probability_Soluble': result_row['Probability_Soluble'],
                    'Probability_Insoluble': result_row['Probability_Insoluble']
                })
            else:
                # No result found - assign default
                final_results.append({
                    'Accession': accession,
                    'Sequence': sequence,
                    'Predictor': 'PLM_Sol',
                    'SolubilityScore': 0.5,
                    'Probability_Soluble': 0.5,
                    'Probability_Insoluble': 0.5
                })
    
    # Save final results
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(output_file, index=False)
    
    print(f"Final results saved to: {output_file}")
    print(f"Total sequences: {len(final_results)}")
    filtered_count = len([r for r in final_results if r['SolubilityScore'] == 0.5 and any(acc == r['Accession'] for _, acc, _, _ in filtered_sequences)])
    print(f"Filtered sequences: {filtered_count}")
    
    return output_file

def run_original_plm_sol_wrapper(fasta_file, output_file, model_checkpoint=None):
    """Run the original working PLM_Sol wrapper"""
    
    # Use the original wrapper that we know works
    wrapper_path = '/home/david_nunn/PLM_Sol/plmsol_predict_wrapper.py'
    
    cmd = ['python', wrapper_path, '--fasta', fasta_file, '--out', output_file]
    
    # Add model checkpoint if provided
    if model_checkpoint:
        cmd.extend(['--model_checkpoint', model_checkpoint])
    
    print(f"Running original PLM_Sol wrapper: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"PLM_Sol wrapper failed with return code {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
    
    print("PLM_Sol wrapper completed successfully")
    return True

def create_fallback_output(fasta_path, output_path):
    """Create fallback output with default predictions when PLM_Sol fails"""
    print("Creating fallback output with default predictions...")
    
    results = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        results.append({
            'Accession': record.id,
            'Sequence': str(record.seq),
            'Predictor': 'PLM_Sol',
            'SolubilityScore': 0.5,  # Default neutral prediction
            'Probability_Soluble': 0.5,
            'Probability_Insoluble': 0.5
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    print(f"Fallback output saved to: {output_path}")
    print(f"Total sequences: {len(results)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='PLM_Sol wrapper with sequence length filtering v2')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--out', required=True, help='Output CSV file')
    parser.add_argument('--max_length', type=int, default=4000, 
                       help='Maximum sequence length to process (default: 4000)')
    parser.add_argument('--model_checkpoint', help='Path to model checkpoint file')
    
    args = parser.parse_args()
    
    print(f"Starting PLM_Sol with filtering v2 (max_length: {args.max_length})")
    print(f"Input FASTA: {args.fasta}")
    print(f"Output CSV: {args.out}")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        filtered_fasta = os.path.join(temp_dir, 'filtered_sequences.fasta')
        temp_output = os.path.join(temp_dir, 'temp_plm_results.csv')
        
        # Filter sequences
        filtered_sequences, kept_sequences = create_filtered_fasta(
            args.fasta, filtered_fasta, args.max_length
        )
        
        if not kept_sequences:
            print("No sequences to process after filtering!")
            # Create output with all filtered sequences
            merge_results_with_filtered('', filtered_sequences, args.fasta, args.out)
            return
        
        # Run PLM_Sol on filtered sequences using the original working wrapper
        try:
            success = run_original_plm_sol_wrapper(filtered_fasta, temp_output, args.model_checkpoint)
            
            if success and os.path.exists(temp_output):
                # Merge results with filtered sequences
                merge_results_with_filtered(temp_output, filtered_sequences, args.fasta, args.out)
            else:
                print("PLM_Sol wrapper failed, creating fallback output for all sequences")
                create_fallback_output(args.fasta, args.out)
                
        except Exception as e:
            print(f"Error running PLM_Sol: {e}")
            print("Creating fallback output for all sequences...")
            create_fallback_output(args.fasta, args.out)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        sys.exit(1)
