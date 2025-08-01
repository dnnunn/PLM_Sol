#!/usr/bin/env python3
"""
PLM_Sol Filtered Wrapper v2 - **PRODUCTION ENTRYPOINT**
------------------------------------------------------

**This is the main production wrapper for PLM_Sol batch/parallel inference.**
- Supports server-side embeddings for fast inference (use --server_embeddings_file argument)
- Handles sequence length filtering to avoid CUDA OOM errors
- Maintains row alignment and output format for downstream workflows

**All other wrappers (predict, server, end-to-end, direct) are legacy or for troubleshooting only.**

Usage (production):
  python plmsol_filtered_wrapper_v2.py --fasta <input_fasta> --out <output_csv> --model_checkpoint <model.t7> --server_embeddings_file <embeddings.json>

Legacy usage (not recommended):
  python plmsol_filtered_wrapper_v2.py --fasta <input_fasta> --out <output_csv> [--max_length 4000]

See PLM_Sol_Script_Reference.md for full workflow and script map.
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
import json

def run_plm_sol_with_server_embeddings(fasta_file, output_file, embeddings_file, model_checkpoint):
    """
    Run PLM_Sol using server-provided embeddings (FAST PATH).
    Calls inference.py directly with pre-computed embeddings.
    """
    try:
        import tempfile
        import yaml
        import json
        
        # Load server embeddings
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        
        # Handle different embedding data formats
        if 'embeddings' in embeddings_data:
            server_embeddings = embeddings_data['embeddings']
        elif isinstance(embeddings_data, list):
            server_embeddings = embeddings_data
        else:
            print(f"ERROR: Invalid embeddings format in {embeddings_file}")
            return False
        
        print(f"Loaded {len(server_embeddings)} server embeddings")
        
        # Create remapping file for sequence IDs
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as remap_file:
            remap_path = remap_file.name
            for i, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
                remap_file.write(f">{record.id}\n{str(record.seq)}\n")
        
        # Create inference config
        config_data = {
            'model_type': 'biLSTM_TextCNN',
            'model_parameters': {
                'dropout': 0.2,
                'kernel_size': 5,
                'output_dim': 2
            },
            'optimizer': 'Adam',
            'remapping': remap_path,
            'key_format': 'fasta_descriptor',
            'output_files_name': output_file
        }
        
        # Add model checkpoint if provided
        if model_checkpoint and os.path.exists(model_checkpoint):
            config_data['checkpoint'] = model_checkpoint
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as config_file:
            config_path = config_file.name
            yaml.dump(config_data, config_file)
        
        print(f"Created inference config: {config_path}")
        print(f"Remapping file: {remap_path}")
        
        # Call inference.py directly with server embeddings
        cmd = [
            "conda", "run", "-n", "PLM_Sol",
            "python", "/home/david_nunn/PLM_Sol/inference.py",
            "--config", config_path
        ]
        
        print(f"Running PLM_Sol inference with server embeddings: {' '.join(cmd)}")
        
        # Import the inference function directly to pass server embeddings
        import sys
        sys.path.insert(0, '/home/david_nunn/PLM_Sol')
        
        try:
            from inference import inference
            from argparse import Namespace
            
            # Create args object from config
            args = Namespace(**config_data)
            
            # Call inference with server embeddings
            result = inference(args, server_embeddings=server_embeddings)
            
            print(f"PLM_Sol inference completed")
            
            # Check if output file was created
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                print(f"ERROR: PLM_Sol output file not created or empty: {output_file}")
                return False
            
            # Validate CSV format
            import pandas as pd
            results_df = pd.read_csv(output_file)
            print(f"PLM_Sol produced {len(results_df)} results")
            print(f"CSV columns: {list(results_df.columns)}")
            
            # Check for real predictions (not all 0.5)
            unique_scores = results_df['SolubilityScore'].nunique()
            if unique_scores == 1 and results_df['SolubilityScore'].iloc[0] == 0.5:
                print(f"WARNING: All predictions are 0.5 fallback values")
                return False
            
            print(f"SUCCESS: PLM_Sol produced {unique_scores} unique prediction scores")
            return True
            
        except ImportError as e:
            print(f"ERROR: Could not import PLM_Sol inference module: {e}")
            return False
        
        finally:
            # Clean up temporary files
            try:
                if 'config_path' in locals():
                    os.unlink(config_path)
                if 'remap_path' in locals():
                    os.unlink(remap_path)
            except:
                pass
            
    except Exception as e:
        print(f"Error running PLM_Sol with server embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    # Debug: check file existence and size
    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        print(f"[DEBUG] Output file exists: {os.path.abspath(output_file)}, size: {size} bytes")
    else:
        print(f"[DEBUG][WARNING] Output file missing immediately after write: {os.path.abspath(output_file)}")
    
    print(f"Final results saved to: {output_file}")
    print(f"Total sequences: {len(final_results)}")
    filtered_count = len([r for r in final_results if r['SolubilityScore'] == 0.5 and any(acc == r['Accession'] for _, acc, _, _ in filtered_sequences)])
    print(f"Filtered sequences: {filtered_count}")
    
    return output_file

def run_plm_sol_traditional(fasta_file, output_file, model_checkpoint=None):
    """
    Traditional PLM_Sol prediction using conda environment and proper working directory.
    Fixed to use conda environment and validate output.
    """
    print(f" TRADITIONAL WRAPPER DEBUG - Starting execution")
    print(f"   Input FASTA: {fasta_file}")
    print(f"   Output file: {output_file}")
    print(f"   Model checkpoint: {model_checkpoint}")
    
    try:
        # Use conda environment for proper dependencies
        wrapper_path = os.path.join(os.path.dirname(__file__), "plmsol_predict_wrapper.py")
        print(f"   Wrapper script: {wrapper_path}")
        print(f"   Wrapper exists: {os.path.exists(wrapper_path)}")
        
        cmd = ['conda', 'run', '-n', 'PLM_Sol', 'python', wrapper_path, 
               '--fasta', fasta_file, '--out', output_file]
        
        if model_checkpoint:
            cmd.extend(['--model_checkpoint', model_checkpoint])
        
        print(f"   Command: {' '.join(cmd)}")
        
        # Set working directory to PLM_Sol root
        working_dir = os.path.dirname(__file__)
        print(f"   Working directory: {working_dir}")
        
        print(f" Executing traditional PLM_Sol wrapper...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=working_dir)
        
        print(f" Traditional wrapper result:")
        print(f"   Return code: {result.returncode}")
        print(f"   STDOUT length: {len(result.stdout)}")
        print(f"   STDERR length: {len(result.stderr)}")
        
        if result.stdout:
            print(f"   STDOUT (first 10 lines):")
            for i, line in enumerate(result.stdout.split('\n')[:10], 1):
                if line.strip():
                    print(f"     {i:2d}: {line}")
        
        if result.stderr:
            print(f"   STDERR (first 10 lines):")
            for i, line in enumerate(result.stderr.split('\n')[:10], 1):
                if line.strip():
                    print(f"     {i:2d}: {line}")
        
        if result.returncode != 0:
            print(f" ERROR: PLM_Sol traditional wrapper failed with code {result.returncode}")
            print(f"Full STDOUT: {result.stdout}")
            print(f"Full STDERR: {result.stderr}")
            return False
        
        # Validate output file exists and has content
        print(f" Validating output file...")
        if not os.path.exists(output_file):
            print(f" ERROR: Output file {output_file} does not exist")
            return False
        
        file_size = os.path.getsize(output_file)
        print(f"   Output file size: {file_size} bytes")
        
        if file_size == 0:
            print(f" ERROR: Output file {output_file} is empty")
            return False
        
        # Check if all predictions are fallback values (0.5)
        print(f" Analyzing prediction scores...")
        try:
            import pandas as pd
            df = pd.read_csv(output_file)
            print(f"   CSV shape: {df.shape}")
            print(f"   CSV columns: {list(df.columns)}")
            
            if 'SolubilityScore' in df.columns:
                scores = df['SolubilityScore'].values
                unique_scores = set(scores)
                print(f"   Unique scores: {unique_scores}")
                print(f"   All scores are 0.5: {unique_scores == {0.5}}")
                
                if len(unique_scores) == 1 and scores[0] == 0.5:
                    print(f"  WARNING: All predictions are 0.5 fallback values - PLM_Sol inference may have failed")
                    print(f"   This suggests the underlying PLM_Sol model did not run successfully")
                    return False
                else:
                    print(f" SUCCESS: Real PLM_Sol predictions detected (not all 0.5)")
            else:
                print(f"  WARNING: SolubilityScore column not found in output")
        except Exception as e:
            print(f"  WARNING: Could not validate prediction scores: {e}")
        
        print(f" Traditional wrapper completed successfully")
        return True
        
    except Exception as e:
        print(f" ERROR in traditional PLM_Sol: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

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
    parser.add_argument('--server_embeddings_file', help='JSON file with server-provided embeddings (optional)')
    
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
        
        # Run PLM_Sol on filtered sequences using the original working wrapper OR server embeddings
        try:
            if args.server_embeddings_file:
                # Use server embeddings (FAST PATH)
                print(f"Using server embeddings from: {args.server_embeddings_file}")
                success = run_plm_sol_with_server_embeddings(filtered_fasta, temp_output, args.server_embeddings_file, args.model_checkpoint)
            else:
                # Use traditional approach (SLOW PATH)
                print("Using traditional PLM_Sol wrapper (no server embeddings)")
                success = run_plm_sol_traditional(filtered_fasta, temp_output, args.model_checkpoint)
            
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
