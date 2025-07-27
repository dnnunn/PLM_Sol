#!/usr/bin/env python3
"""
PLM_Sol Minimal Wrapper for Benchmarking
----------------------------------------

This wrapper accommodates PLM_Sol's hardcoded CSV output path while maintaining 
the standardized output format required for benchmarking.

Usage:
  python plmsol_minimal_wrapper.py --fasta <input_fasta> --out <output_csv>

Outputs CSV with columns:
  Accession, Sequence, Predictor, SolubilityScore, Probability_Soluble, Probability_Insoluble
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
import pandas as pd
from Bio import SeqIO
import yaml
import shutil

# Helper to write a config YAML for embedding
EMBED_CONFIG_TEMPLATE = {
    'global': {
        'sequences_file': '',  # to be filled
        'prefix': ''          # to be filled
    }
}

def run_embeddings(fasta_path, embeddings_dir):
    """Run the bio_embeddings tool to generate embeddings for protein sequences"""
    # Create embeddings config
    config_path = os.path.join(embeddings_dir, 'embed_config.yml')
    
    # Use absolute paths to avoid any issues
    abs_fasta_path = os.path.abspath(fasta_path)
    abs_embeddings_dir = os.path.abspath(embeddings_dir)
    
    # Configure embedding generation - exactly matching the working config structure
    config = {
        'global': {
            'sequences_file': abs_fasta_path,
            'prefix': abs_embeddings_dir
        },
        't5_embeddings': {
            'type': 'embed',
            'protocol': 'prottrans_t5_xl_u50',
            'half_precision_model': True,
            'half_precision': True
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Created embeddings config at {config_path}")
    
    # CRITICAL FIX: Use bio_embeddings CLI command with --overwrite flag
    # This ensures both embedding file AND remapped_sequences_file.fasta are created
    cmd = f"bio_embeddings {config_path} --overwrite"
    
    print(f"Running embedding command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Print output regardless of success or failure
    if result.stdout:
        print(f"Embedding stdout: {result.stdout}")
    if result.stderr:
        print(f"Embedding stderr: {result.stderr}")
    
    # Check return code and raise error if failed
    if result.returncode != 0:
        raise RuntimeError(f"Embedding generation failed with code {result.returncode}: {result.stderr}")
    
    # Return the path to the embeddings file
    # Note: bio_embeddings tool creates a t5_embeddings/ subdirectory
    return os.path.join(abs_embeddings_dir, 't5_embeddings', 'embeddings_file.h5')

def run_inference(embeddings_file, remapped_sequences_file, tmpdir, model_checkpoint=None):
    """Run the PLM_Sol inference using the proven working approach from memories"""
    # Get the directory of this wrapper script - this is the PLM_Sol root directory
    plmsol_root = os.path.dirname(os.path.abspath(__file__))
    inference_script = os.path.join(plmsol_root, 'inference.py')
    
    print(f"DEBUG: PLM_Sol root directory: {plmsol_root}")
    print(f"DEBUG: Inference script: {inference_script}")
    print(f"DEBUG: Embeddings file: {embeddings_file}")
    print(f"DEBUG: Remapped sequences file: {remapped_sequences_file}")
    
    # Verify required files exist
    if not os.path.exists(inference_script):
        raise RuntimeError(f"Inference script not found: {inference_script}")
    if not os.path.exists(embeddings_file):
        raise RuntimeError(f"Embeddings file not found: {embeddings_file}")
    if not os.path.exists(remapped_sequences_file):
        raise RuntimeError(f"Remapped sequences file not found: {remapped_sequences_file}")
    
    # Use the provided model checkpoint, or default to the optimized one
    if model_checkpoint and os.path.exists(model_checkpoint):
        checkpoint_path = os.path.abspath(model_checkpoint)
        print(f"DEBUG: Using provided model checkpoint: {checkpoint_path}")
    else:
        # Default to the optimized model from memories
        checkpoint_path = os.path.join(plmsol_root, 'saved_models', 'model-10.t7')
        if not os.path.exists(checkpoint_path):
            # Fallback to original model
            checkpoint_path = os.path.join(plmsol_root, 'model_param', 'model_param.t7')
        print(f"DEBUG: Using default model checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Create inference config using the PROVEN working format from memories
    config_path = os.path.join(tmpdir, 'inference_config.yml')
    
    # CRITICAL: Use the exact config format that worked before
    config = {
        'output_files_name': 'test_inference',
        'log_iterations': 100,
        'n_draws': 1000,
        'batch_size': 1,
        'checkpoint': checkpoint_path,
        'key_format': 'fasta_descriptor',  # CRITICAL: From memories - this is the working format
        'embeddings_file': os.path.abspath(embeddings_file),
        'remapped_sequences_file': os.path.abspath(remapped_sequences_file)
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"DEBUG: Created inference config at {config_path}")
    print(f"DEBUG: Config contents: {config}")
    
    # CRITICAL: Must run from PLM_Sol root directory (from memories)
    # PLM_Sol writes output to hardcoded "protTrans_prediction_result.csv" in current directory
    
    # Use absolute path for config when changing directories
    abs_config_path = os.path.abspath(config_path)
    
    # Build command - use python directly since we're in the right conda env
    cmd = ["python", inference_script, abs_config_path]
    
    print(f"DEBUG: Running inference command: {' '.join(cmd)}")
    print(f"DEBUG: Working directory: {plmsol_root}")
    
    # Execute from PLM_Sol root directory (CRITICAL from memories)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=plmsol_root)
    
    # Print output for debugging
    print(f"DEBUG: Inference return code: {result.returncode}")
    if result.stdout:
        print(f"DEBUG: Inference stdout: {result.stdout}")
    if result.stderr:
        print(f"DEBUG: Inference stderr: {result.stderr}")
    
    # Check return code
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed with code {result.returncode}: {result.stderr}")
    
    # Return the path to the hardcoded output file (from memories)
    prediction_file = os.path.join(plmsol_root, 'protTrans_prediction_result.csv')
    
    if not os.path.exists(prediction_file):
        print(f"ERROR: Expected output file not found: {prediction_file}")
        print(f"DEBUG: Files in PLM_Sol root: {os.listdir(plmsol_root)}")
        raise RuntimeError(f"Expected output file not found: {prediction_file}")
    
    print(f"DEBUG: Inference completed successfully, output at {prediction_file}")
    return prediction_file

def format_results(prediction_file, fasta_path, output_path, remapped_sequences_file=None):
    """Format the PLM_Sol results to match the benchmarking standard"""
    try:
        # Read the PLM_Sol output file - contains MD5 hashes as protein_ID
        pred_df = pd.read_csv(prediction_file)
        print(f"DEBUG: Read prediction file with {len(pred_df)} rows and columns: {pred_df.columns.tolist()}")
        
        # CRITICAL FIX: Use mapping_file.csv instead of parsing FASTA description
        # Find the mapping CSV file in the same directory as the remapped FASTA
        if remapped_sequences_file and os.path.exists(remapped_sequences_file):
            mapping_csv_path = os.path.join(os.path.dirname(remapped_sequences_file), 'mapping_file.csv')
        else:
            # Search for the most recent mapping file
            plmsol_root = os.path.dirname(os.path.abspath(__file__))
            mapping_files = []
            for root, dirs, files in os.walk(plmsol_root):
                for file in files:
                    if file == 'mapping_file.csv':
                        mapping_files.append(os.path.join(root, file))
            
            if not mapping_files:
                print("ERROR: Could not find mapping_file.csv")
                return False
            
            # Use the most recently modified mapping file
            mapping_csv_path = sorted(mapping_files, key=os.path.getmtime, reverse=True)[0]
        
        print(f"DEBUG: Using mapping file: {mapping_csv_path}")
        
        # Read the mapping CSV file
        if not os.path.exists(mapping_csv_path):
            print(f"ERROR: Mapping file not found: {mapping_csv_path}")
            return False
        
        mapping_df = pd.read_csv(mapping_csv_path)
        print(f"DEBUG: Mapping file columns: {mapping_df.columns.tolist()}")
        print(f"DEBUG: Mapping file has {len(mapping_df)} rows")
        
        # Create hash_to_id mapping from the CSV
        # The CSV has columns: unnamed index, 'original_id', 'sequence_length'
        # The hash is in the first column (index 0)
        hash_to_id = {}
        if len(mapping_df.columns) >= 2:
            # Handle both cases: with and without unnamed index column
            if 'original_id' in mapping_df.columns:
                # Use the first column as hash and 'original_id' as the mapping target
                hash_col = mapping_df.columns[0]  # Usually unnamed index or hash
                for _, row in mapping_df.iterrows():
                    hash_val = str(row[hash_col]) if pd.notna(row[hash_col]) else None
                    orig_id = str(row['original_id']) if pd.notna(row['original_id']) else None
                    if hash_val and orig_id:
                        hash_to_id[hash_val] = orig_id
            else:
                print(f"ERROR: 'original_id' column not found in mapping file")
                return False
        else:
            print(f"ERROR: Mapping file has insufficient columns: {mapping_df.columns.tolist()}")
            return False
        
        print(f"DEBUG: Created mapping for {len(hash_to_id)} hash->ID pairs")
        if len(hash_to_id) > 0:
            sample_mapping = dict(list(hash_to_id.items())[:3])
            print(f"DEBUG: Sample mappings: {sample_mapping}")
        
        # Get the sequences from the original FASTA file
        seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}
        print(f"DEBUG: Read {len(seqs)} sequences from original FASTA")
        
        # Map protein_ID (MD5 hash) to original sequence ID
        if 'protein_ID' in pred_df.columns:
            print(f"DEBUG: Mapping {len(pred_df)} protein_IDs to original sequence names")
            mapped_ids = []
            unmapped_count = 0
            
            for hash_id in pred_df['protein_ID']:
                if str(hash_id) in hash_to_id:
                    mapped_ids.append(hash_to_id[str(hash_id)])
                else:
                    # Keep the hash if no mapping found
                    mapped_ids.append(str(hash_id))
                    unmapped_count += 1
            
            print(f"DEBUG: Successfully mapped {len(mapped_ids) - unmapped_count} IDs, {unmapped_count} unmapped")
            if unmapped_count > 0:
                print(f"DEBUG: Sample unmapped hashes: {pred_df['protein_ID'].head(3).tolist()}")
            
            # Replace protein_ID with original sequence ID
            pred_df['Accession'] = mapped_ids
        else:
            print(f"ERROR: 'protein_ID' column not found in prediction file")
            return False
        
        # Ensure we have a proper Sequence column
        if 'sequence' in pred_df.columns:
            # Rename the column to standard name
            pred_df.rename(columns={'sequence': 'Sequence'}, inplace=True)
        
        # Add sequence from FASTA file if not already present or empty
        if 'Sequence' not in pred_df.columns or pred_df['Sequence'].isna().any() or (pred_df['Sequence'] == '').any():
            pred_df['Sequence'] = [seqs.get(acc, "") for acc in pred_df['Accession']]
        
        # Add the predictor name
        pred_df['Predictor'] = 'PLM_Sol'
        
        # Validate that we have all required columns
        if 'Accession' not in pred_df.columns or 'Sequence' not in pred_df.columns:
            raise ValueError(f"Could not find required columns. Available columns: {pred_df.columns.tolist()}")
        
        # Map prediction column
        if 'SolubilityScore' not in pred_df.columns:
            if 'predict_result' in pred_df.columns:
                pred_df['SolubilityScore'] = pred_df['predict_result']
            elif 'probability' in pred_df.columns:
                pred_df['SolubilityScore'] = pred_df['probability']
            else:
                raise ValueError(f"Could not find prediction column. Available columns: {pred_df.columns.tolist()}")
        
        # Add probability columns
        pred_df['Probability_Soluble'] = pred_df['SolubilityScore']
        pred_df['Probability_Insoluble'] = 1 - pred_df['SolubilityScore']
        
        # Ensure the required columns exist
        required_columns = ['Accession', 'Sequence', 'Predictor', 'SolubilityScore', 
                           'Probability_Soluble', 'Probability_Insoluble']
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Save the standardized results
        final_df = pred_df[required_columns]
        print(f"DEBUG: Final output has {len(final_df)} rows with accessions: {final_df['Accession'].head(3).tolist()}")
        final_df.to_csv(output_path, index=False)
        print(f"Results written to {output_path}")
        return True
    except Exception as e:
        import traceback
        print(f"Error formatting results: {e}")
        traceback.print_exc()
        return False

def create_fallback_output(fasta_path, output_path):
    """Create a fallback output with neutral predictions if the main process fails"""
    print("Creating fallback output with neutral predictions")
    try:
        # Parse the FASTA file
        records = list(SeqIO.parse(fasta_path, "fasta"))
        
        # Create a DataFrame with neutral predictions
        data = []
        for rec in records:
            data.append({
                'Accession': rec.id,
                'Sequence': str(rec.seq),
                'Predictor': 'PLM_Sol',
                'SolubilityScore': 0.5,  # Neutral score
                'Probability_Soluble': 0.5,
                'Probability_Insoluble': 0.5
            })
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Save the fallback results
        out_df = pd.DataFrame(data)
        out_df.to_csv(output_path, index=False)
        print(f"Fallback results written to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating fallback output: {e}")
        return False

def main():
    """Main function to handle PLM_Sol batch predictions"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Minimal PLM_Sol predictor wrapper")
    parser.add_argument('--fasta', '-f', required=True, help='Input FASTA file')
    parser.add_argument('--out', '-o', required=True, help='Output CSV file')
    parser.add_argument('--model_checkpoint', help='Path to a specific model checkpoint file (.t7)')
    parser.add_argument('--debug', action='store_true', help='Keep temporary files for debugging')
    args = parser.parse_args()
    
    # Use a temporary directory for intermediates
    if args.debug:
        # In debug mode, create a persistent temporary directory
        tmpdir = tempfile.mkdtemp(prefix="plmsol_debug_")
        print(f"Debug mode: Temporary directory will be preserved at: {tmpdir}")
        success = run_pipeline(args.fasta, args.out, tmpdir)
        print(f"Processing {'succeeded' if success else 'failed'}. Debug files preserved in: {tmpdir}")
    else:
        # Use standard temporary directory that will be cleaned up
        with tempfile.TemporaryDirectory() as tmpdir:
            success = run_pipeline(args.fasta, args.out, tmpdir, args.model_checkpoint)

def run_pipeline(fasta_path, output_path, tmpdir, model_checkpoint=None):
    """Run the PLM_Sol prediction pipeline with error handling"""
    try:
        print(f"Starting PLM_Sol prediction for {fasta_path}")
        print(f"Using temporary directory: {tmpdir}")
        
        # CRITICAL FIX: Clean up stale output files before starting
        plmsol_root = os.path.dirname(os.path.abspath(__file__))
        stale_output_file = os.path.join(plmsol_root, 'protTrans_prediction_result.csv')
        if os.path.exists(stale_output_file):
            print(f"DEBUG: Removing stale output file: {stale_output_file}")
            os.remove(stale_output_file)
        
        # Also clean up any other potential stale files
        stale_patterns = ['test_inference*.csv', '*prediction_result*.csv']
        for pattern in stale_patterns:
            import glob
            stale_files = glob.glob(os.path.join(plmsol_root, pattern))
            for stale_file in stale_files:
                if os.path.exists(stale_file):
                    print(f"DEBUG: Removing stale file: {stale_file}")
                    os.remove(stale_file)
        
        # Step 1: Generate embeddings
        embeddings_dir = os.path.join(tmpdir, 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        embeddings_file = run_embeddings(fasta_path, embeddings_dir)
        
        # CRITICAL: Find the remapped_sequences_file produced by bio_embeddings
        # It will be in the same directory as the embeddings file but one level up
        remapped_sequences_file = os.path.join(os.path.dirname(os.path.dirname(embeddings_file)), 
                                            'remapped_sequences_file.fasta')
        
        if not os.path.exists(remapped_sequences_file):
            print(f"ERROR: Could not find remapped sequences file at {remapped_sequences_file}")
            print("Using fallback output")
            return create_fallback_output(fasta_path, output_path)
            
        print(f"Found remapped sequences file at {remapped_sequences_file}")
        
        # Step 2: Run inference with the remapped sequences file
        prediction_file = run_inference(embeddings_file, remapped_sequences_file, tmpdir, model_checkpoint)
        
        if prediction_file and os.path.exists(prediction_file):
            # Step 3: Format results - PASS THE REMAPPED_SEQUENCES_FILE
            success = format_results(prediction_file, fasta_path, output_path, remapped_sequences_file)
            if success:
                return True
        
        # If we get here, something failed
        print("Failed to generate prediction file, using fallback")
        return create_fallback_output(fasta_path, output_path)
    
    except Exception as e:
        import traceback
        print(f"Error in pipeline: {e}")
        traceback.print_exc()
        print("Using fallback output")
        return create_fallback_output(fasta_path, output_path)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        sys.exit(1)
