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
import logging

# Setup debug logging to file
WRAPPER_DEBUG_LOG = os.path.join(os.path.dirname(__file__), 'plmsol_filtered_wrapper_v2_debug.log')
logging.basicConfig(
    filename=WRAPPER_DEBUG_LOG,
    filemode='a',
    format='%(asctime)s %(levelname)s [%(module)s]: %(message)s',
    level=logging.DEBUG
)
logging.info('\n\n--- NEW RUN: plmsol_filtered_wrapper_v2.py ---')
logging.info(f'Current working directory: {os.getcwd()}')
logging.info(f'Python executable: {sys.executable}')
logging.info(f'Args: {sys.argv}')


def run_plm_sol_with_server_embeddings(fasta_file, output_file, embeddings_file, model_checkpoint):
    """
    Run PLM_Sol using server-provided embeddings (FAST PATH).
    Calls inference.py directly with pre-computed embeddings.
    """
    try:
        import tempfile
        import yaml
        import json
        
        logging.info(f'Loading embeddings from: {embeddings_file}')
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        logging.info(f'Embeddings keys: {list(embeddings_data.keys()) if hasattr(embeddings_data, "keys") else "[list]"}')
        logging.info(f'Embeddings preview: {str(embeddings_data)[:300]}')
        
        # Handle different embedding data formats
        if 'embeddings' in embeddings_data:
            server_embeddings = embeddings_data['embeddings']
        elif isinstance(embeddings_data, list):
            server_embeddings = embeddings_data
        else:
            logging.error(f"ERROR: Invalid embeddings format in {embeddings_file}")
            print(f"ERROR: Invalid embeddings format in {embeddings_file}")
            return False
        
        logging.info(f"Loaded {len(server_embeddings)} server embeddings")
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
            'optimizer_parameters': {
                'lr': 1e-4
            },
            'remapping': remap_path,
            'key_format': 'fasta_descriptor',
            'output_files_name': output_file
        }
        
        if model_checkpoint:
            logging.info(f'Checking model checkpoint: {model_checkpoint}')
            if os.path.exists(model_checkpoint):
                logging.info(f'Model checkpoint exists: {model_checkpoint}')
                config_data['checkpoint'] = model_checkpoint
            else:
                logging.error(f'Model checkpoint MISSING: {model_checkpoint}')
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as config_file:
            config_path = config_file.name
            yaml.dump(config_data, config_file)
        logging.info(f'Created inference config: {config_path}')
        logging.info(f'Config YAML content:\n{yaml.dump(config_data)}')
        logging.info(f'Remapping file: {remap_path}')
        
        # Call inference.py directly with server embeddings
        cmd = [
            "conda", "run", "-n", "PLM_Sol",
            "python", "/home/david_nunn/PLM_Sol/inference.py",
            "--config", config_path,
            "--server_embeddings_file", embeddings_file
        ]
        logging.info(f'Inference command: {" ".join(cmd)}')
        print(f"Running PLM_Sol inference with server embeddings: {' '.join(cmd)}")
        
        # Run inference.py as a subprocess and capture stdout/stderr
        import subprocess
        logging.info('Running inference.py as subprocess with server embeddings...')
        cmd = [
            "conda", "run", "-n", "PLM_Sol",
            "python", "/home/david_nunn/PLM_Sol/inference.py",
            "--config", config_path,
            "--server_embeddings_file", embeddings_file
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            logging.info(f'inference.py return code: {proc.returncode}')
            logging.info(f'inference.py STDOUT:\n{proc.stdout[:1000]}')
            logging.info(f'inference.py STDERR:\n{proc.stderr[:1000]}')
            print(f'inference.py return code: {proc.returncode}')
            if proc.returncode != 0:
                print(f'ERROR: inference.py exited with code {proc.returncode}')
            # Log output file existence and size
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                logging.info(f'Output file {output_file} exists, size: {size} bytes')
                print(f'Output file {output_file} exists, size: {size} bytes')
            else:
                logging.error(f'Output file {output_file} does not exist after inference.py')
                print(f'ERROR: Output file {output_file} does not exist after inference.py')
            # Validate CSV format
            try:
                # Read and validate the output CSV
                df = pd.read_csv(output_file)
                logging.info(f'PLM_Sol produced {len(df)} results')
                logging.info(f'CSV columns: {list(df.columns)}')
                logging.info(f'CSV head:\n{df.head()}')
                
                # Map PLM_Sol output columns to expected format
                if 'protein_ID' in df.columns and 'predict_result' in df.columns:
                    df = df.rename(columns={'protein_ID': 'Accession', 'predict_result': 'SolubilityScore'})
                    logging.info(f'Renamed columns: protein_ID -> Accession, predict_result -> SolubilityScore')
                    # Save the renamed DataFrame back to the temp file
                    df.to_csv(output_file, index=False)
                    logging.info(f'Saved renamed columns to temp file: {output_file}')
                
                # Validate required columns
                if 'Accession' not in df.columns or 'SolubilityScore' not in df.columns:
                    raise ValueError(f"Output CSV missing required columns. Found: {list(df.columns)}")
                
                unique_scores = df['SolubilityScore'].nunique()
                if unique_scores == 1 and df['SolubilityScore'].iloc[0] == 0.5:
                    msg = f"WARNING: All predictions are 0.5 fallback values (possible inference/model failure)"
                    logging.warning(msg)
                    print(msg)
                    return False
                logging.info(f"SUCCESS: PLM_Sol produced {unique_scores} unique prediction scores")
                print(f"SUCCESS: PLM_Sol produced {unique_scores} unique prediction scores")
            except Exception as e:
                logging.error(f"Exception reading or validating output CSV: {e}")
                print(f"ERROR: Exception reading or validating output CSV: {e}")
                return None
            return output_file
        finally:
            # Clean up temporary files
            try:
                if 'config_path' in locals():
                    os.unlink(config_path)
                if 'remap_path' in locals():
                    os.unlink(remap_path)
            except Exception as e:
                logging.warning(f'Error cleaning up temp files: {e}')

    except Exception as e:
        print(f"Error running PLM_Sol with server embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_filtered_fasta(input_fasta, output_fasta, max_length=4000):
    """Create a filtered FASTA file excluding sequences longer than max_length."""
    filtered_sequences = []
    kept_sequences = []
    
    print(f"\n[FILTER DEBUG] === SEQUENCE FILTERING ANALYSIS ===")
    print(f"[FILTER DEBUG] Input FASTA: {input_fasta}")
    print(f"[FILTER DEBUG] Max length limit: {max_length}")
    
    with open(output_fasta, 'w') as out_handle:
        for i, record in enumerate(SeqIO.parse(input_fasta, 'fasta')):
            seq_len = len(record.seq)
            print(f"[FILTER DEBUG] Sequence {i}: ID='{record.id}', Length={seq_len}")
            
            if seq_len > max_length:
                filtered_sequences.append((i, record.id, len(record.seq), str(record.seq)))
                print(f"[FILTER DEBUG] ❌ FILTERED OUT: {record.id} (length: {seq_len} > {max_length})")
            else:
                SeqIO.write(record, out_handle, 'fasta')
                kept_sequences.append((i, record.id, len(record.seq), str(record.seq)))
                print(f"[FILTER DEBUG] ✅ KEPT: {record.id} (length: {seq_len} <= {max_length})")
    
    print(f"\n[FILTER DEBUG] === FILTERING SUMMARY ===")
    print(f"[FILTER DEBUG] Kept {len(kept_sequences)} sequences, filtered {len(filtered_sequences)} sequences")
    
    if len(kept_sequences) == 0:
        print(f"[FILTER DEBUG] ⚠️  ALL SEQUENCES FILTERED OUT! This will trigger early return and fallback values!")
        print(f"[FILTER DEBUG] Consider increasing max_length or checking sequence validity")
    
    return filtered_sequences, kept_sequences

def merge_results_with_filtered(plm_sol_results, filtered_sequences, original_fasta, output_file):
    """
    Merge PLM_Sol results with filtered sequences, maintaining original order.
    Filtered sequences get default values (0.5).
    """
    print(f"\n[MERGE DEBUG] === STARTING MERGE FUNCTION ===")
    print(f"[MERGE DEBUG] PLM_Sol results file: {plm_sol_results}")
    print(f"[MERGE DEBUG] Original FASTA: {original_fasta}")
    print(f"[MERGE DEBUG] Output file: {output_file}")
    print(f"[MERGE DEBUG] Filtered sequences count: {len(filtered_sequences)}")
    
    # CRITICAL FIX: If no sequences were filtered, directly copy temp results
    if len(filtered_sequences) == 0 and os.path.exists(plm_sol_results):
        print(f"[MERGE DEBUG] ✅ NO SEQUENCES FILTERED - DIRECT COPY MODE")
        print(f"[MERGE DEBUG] Copying real predictions directly from {plm_sol_results} to {output_file}")
        
        # Read temp results and copy directly to output
        results_df = pd.read_csv(plm_sol_results)
        print(f"[MERGE DEBUG] Direct copy: {len(results_df)} real predictions")
        print(f"[MERGE DEBUG] Sample predictions: {list(results_df['SolubilityScore'].head())}")
        
        # Save directly to output file
        results_df.to_csv(output_file, index=False)
        
        # Verify the copy worked
        if os.path.exists(output_file):
            verify_df = pd.read_csv(output_file)
            print(f"[MERGE DEBUG] ✅ Direct copy successful: {len(verify_df)} predictions in output")
            print(f"[MERGE DEBUG] Output predictions: {list(verify_df['SolubilityScore'].head())}")
            return
        else:
            print(f"[MERGE DEBUG] ❌ Direct copy failed - falling back to merge logic")
    
    # Original merge logic for cases with filtered sequences
    results_dict = {}
    if os.path.exists(plm_sol_results):
        results_df = pd.read_csv(plm_sol_results)
        print(f"[DEBUG] Merge reading temp results with columns: {list(results_df.columns)}")
        print(f"[DEBUG] Temp results head:\n{results_df.head()}")
        
        # DEBUG: Show all sequence names in temp results
        temp_sequence_names = list(results_df['Accession'].values)
        print(f"[DEBUG] Temp results sequence names: {temp_sequence_names}")
        
        for _, row in results_df.iterrows():
            clean_key = str(row['Accession']).strip().lower()
            results_dict[clean_key] = row
            
        print(f"[DEBUG] Loaded {len(results_dict)} results into merge dictionary")
        print(f"[DEBUG] Results dict keys: {list(results_dict.keys())}")
    
    # Build final results maintaining original order
    final_results = []
    
    # DEBUG: Show all sequence names expected from original FASTA
    original_sequence_names = []
    for record in SeqIO.parse(original_fasta, 'fasta'):
        original_sequence_names.append(record.id)
    print(f"[DEBUG] Original FASTA sequence names: {original_sequence_names}")
    
    for record in SeqIO.parse(original_fasta, 'fasta'):
        accession_raw = record.id
        accession = accession_raw.strip().lower()
        sequence = str(record.seq)
        
        # Check if this sequence was filtered
        # The 'accession' here is already normalized (lower, stripped)
        # The 'acc' from filtered_sequences is the raw record.id
        was_filtered = any(acc.strip().lower() == accession for _, acc, _, _ in filtered_sequences)
        
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
            print(f"[MERGE DEBUG] Looking up sequence: '{accession_raw}' -> normalized: '{accession}'")
            
            if accession in results_dict:
                result_row = results_dict[accession]
                print(f"[MERGE DEBUG] ✅ FOUND REAL PREDICTION for '{accession}': {result_row['SolubilityScore']}")
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
                print(f"[MERGE DEBUG] ❌ NO MATCH FOUND for '{accession}' - FALLBACK TO 0.5")
                print(f"[MERGE DEBUG] Available keys in results_dict: {list(results_dict.keys())}")
                print(f"[MERGE DEBUG] Key comparison:")
                for key in results_dict.keys():
                    print(f"[MERGE DEBUG]   '{key}' == '{accession}' ? {key == accession}")
                
                final_results.append({
                    'Accession': accession,
                    'Sequence': sequence,
                    'Predictor': 'PLM_Sol',
                    'SolubilityScore': 0.5,
                    'Probability_Soluble': 0.5,
                    'Probability_Insoluble': 0.5
                })
                print(f"[MERGE DEBUG] Added fallback result for '{accession}'")
    
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
        
        # Set working directory to PLM_Sol root; fall back to current dir if blank
        working_dir = os.path.dirname(os.path.abspath(wrapper_path))
        if not working_dir:
            working_dir = os.getcwd()
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
            return None
        
        # Validate output file exists and has content
        print(f" Validating output file...")
        if not os.path.exists(output_file):
            print(f" ERROR: Output file {output_file} does not exist")
            return None
        
        file_size = os.path.getsize(output_file)
        print(f"   Output file size: {file_size} bytes")
        
        if file_size == 0:
            print(f" ERROR: Output file {output_file} is empty")
            return None
        
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
                    return None
                else:
                    print(f" SUCCESS: Real PLM_Sol predictions detected (not all 0.5)")
            else:
                print(f"  WARNING: SolubilityScore column not found in output")
        except Exception as e:
            print(f"  WARNING: Could not validate prediction scores: {e}")
        
        print(f" Traditional wrapper completed successfully")
        return output_file
        
    except Exception as e:
        print(f" ERROR in traditional PLM_Sol: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None

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
        print(f"\n[MAIN DEBUG] === ABOUT TO RUN PLM_SOL ===")
        print(f"[MAIN DEBUG] filtered_fasta = {filtered_fasta}")
        print(f"[MAIN DEBUG] temp_output = {temp_output}")
        print(f"[MAIN DEBUG] server_embeddings_file = {args.server_embeddings_file}")
        
        try:
            plm_sol_results_path = None
            if args.server_embeddings_file:
                # Use server embeddings (FAST PATH)
                print(f"[MAIN DEBUG] Taking SERVER EMBEDDINGS path")
                print(f"Using server embeddings from: {args.server_embeddings_file}")
                plm_sol_results_path = run_plm_sol_with_server_embeddings(filtered_fasta, temp_output, args.server_embeddings_file, args.model_checkpoint)
                print(f"[MAIN DEBUG] Server embeddings function returned: {plm_sol_results_path}")
            else:
                # Use traditional approach (SLOW PATH)
                print(f"[MAIN DEBUG] Taking TRADITIONAL path")
                print("Using traditional PLM_Sol wrapper (no server embeddings)")
                plm_sol_results_path = run_plm_sol_traditional(filtered_fasta, temp_output, args.model_checkpoint)
                print(f"[MAIN DEBUG] Traditional function returned: {plm_sol_results_path}")
            
            if plm_sol_results_path and os.path.exists(plm_sol_results_path):
                # Merge results with filtered sequences
                print(f"\n[WRAPPER DEBUG] === CALLING MERGE FUNCTION ===")
                print(f"[WRAPPER DEBUG] plm_sol_results_path = {plm_sol_results_path}")
                print(f"[WRAPPER DEBUG] temp_output exists = {os.path.exists(plm_sol_results_path)}")
                print(f"[WRAPPER DEBUG] filtered_sequences count = {len(filtered_sequences)}")
                print(f"[WRAPPER DEBUG] args.fasta = {args.fasta}")
                print(f"[WRAPPER DEBUG] args.out = {args.out}")
                merge_results_with_filtered(plm_sol_results_path, filtered_sequences, args.fasta, args.out)
            else:
                print(f"PLM_Sol wrapper failed (results path: {plm_sol_results_path}), creating fallback output for all sequences")
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
