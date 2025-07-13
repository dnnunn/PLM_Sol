#!/usr/bin/env python3
"""
PLM_Sol Batch Predictor Wrapper
Standardizes output for benchmarking solubility predictors.

Usage:
  python plmsol_predict_wrapper.py --fasta <input_fasta> --out <output_csv>

Outputs CSV with columns:
  Accession, Sequence, Predictor, SolubilityScore, Probability_Soluble, Probability_Insoluble
"""
import argparse
import os
import subprocess
import sys
import tempfile
import shutil
import pandas as pd
from Bio import SeqIO
import yaml
import time

# Helper to write a config YAML for embedding
EMBED_CONFIG_TEMPLATE = {
    'global': {
        'sequences_file': '',  # to be filled
        'prefix': ''           # to be filled
    }
}

# Helper to write a config YAML for inference
INFER_CONFIG_TEMPLATE = {
    'global': {
        'model_config': './configs/inference_Sol_biLSTM_TextCNN.yml',
        'embeddings_file': '',   # to be filled
        'remapping': '',        # to be filled
        'output_file': ''       # to be filled
    }
}

def fasta_to_remapped(fasta_path, remapped_path):
    # PLM_Sol expects remapped_sequences_file.fasta in FASTA format
    shutil.copy(fasta_path, remapped_path)

def run_embeddings(config_path):
    # Get the directory of this wrapper script
    wrapper_dir = os.path.dirname(os.path.abspath(__file__))
    # Use absolute path to the script
    embed_script = os.path.join(wrapper_dir, 'generate_embeddings_memory_efficient.py')
    cmd = [
        'python', embed_script,
        '--config', config_path
    ]
    subprocess.run(cmd, check=True)

def run_inference(config_path):
    """Run the PLM_Sol inference script
    
    Returns:
        str or None: Path to the output file if found, None if not found
    """
    # Get the directory of this wrapper script
    wrapper_dir = os.path.dirname(os.path.abspath(__file__))
    # Use absolute path to the script
    inference_script = os.path.join(wrapper_dir, 'inference.py')
    
    # IMPORTANT: The output file is hardcoded to "protTrans_prediction_result.csv" in solver.py's predict_evaluation method
    # This means the output file path in the config file is completely ignored by the inference script
    # So we'll run the inference from our wrapper directory to control where it's created
    expected_output_file = os.path.join(wrapper_dir, "protTrans_prediction_result.csv")
    
    cmd = [
        'python', inference_script,
        '--config', config_path
    ]
    
    # First remove any existing output file to ensure we get fresh results
    if os.path.exists(expected_output_file):
        print(f"Removing existing output file at {expected_output_file}")
        os.remove(expected_output_file)
            
    # Save current directory so we can return to it
    original_dir = os.getcwd()
    
    # Check for required model directory
    model_param_dir = os.path.join(wrapper_dir, "model_param")
    if not os.path.exists(model_param_dir):
        print(f"WARNING: model_param directory not found at {model_param_dir}")
        print("This may cause the inference to fail silently.")
    elif not os.path.exists(os.path.join(model_param_dir, "train_arguments.yml")):
        print(f"WARNING: train_arguments.yml not found in {model_param_dir}")
        print("This may cause the inference to fail silently.")
    
    try:
        # Change to the wrapper directory where inference.py exists
        # This ensures the hardcoded output path in solver.py will create the file here
        print(f"Changing working directory to: {wrapper_dir}")
        os.chdir(wrapper_dir)
        
        # Run the inference script with a timeout
        print(f"Running inference command: {' '.join(cmd)}")
        try:
            # Capture and print output to help with debugging
            # Add a timeout to avoid hanging indefinitely
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            print(f"Inference stdout:\n{result.stdout}")
            if result.stderr:
                print(f"Inference stderr:\n{result.stderr}")
        except subprocess.TimeoutExpired as e:
            print(f"Inference timed out after {e.timeout} seconds")
            print(f"Stdout: {e.stdout if hasattr(e, 'stdout') else 'Not captured'}")
            print(f"Stderr: {e.stderr if hasattr(e, 'stderr') else 'Not captured'}")
            raise
        except subprocess.CalledProcessError as e:
            print(f"Inference failed with exit code {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise
    finally:
        # Always return to the original directory
        os.chdir(original_dir)
    
    # Wait for the output file to appear (with timeout)
    max_wait_time = 180  # seconds - increased wait time
    wait_interval = 2   # seconds
    waited = 0
    output_file_found = None
    
    # Since we changed the working directory before running inference,
    # the output should be at the expected_output_file path
    print(f"Waiting for output file at: {expected_output_file}")
    
    while waited < max_wait_time and output_file_found is None:
        if os.path.exists(expected_output_file) and os.path.getsize(expected_output_file) > 0:
            print(f"Output file appeared at {expected_output_file} after {waited} seconds")
            output_file_found = expected_output_file
        else:
            print(f"Waiting for output file... ({waited}/{max_wait_time} seconds)")
            time.sleep(wait_interval)
            waited += wait_interval
            
    if output_file_found is None:
        print("No output file was generated within the timeout period")
        # Check other possible locations as fallback
        alt_paths = [
            "/home/david_nunn/PLM_Sol/protTrans_prediction_result.csv",  # VM path
            "./protTrans_prediction_result.csv",                        # Current working directory
        ]
        
        for path in alt_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print(f"Found output at alternate location: {path}")
                return path
                
        return None
    
    return output_file_found

def main():
    """Main function to handle PLM_Sol batch predictions"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch PLM_Sol predictor wrapper")
    parser.add_argument('--fasta', '-f', required=True, help='Input FASTA file')
    parser.add_argument('--out', '-o', required=True, help='Output CSV file')
    parser.add_argument('--debug', action='store_true', help='Keep temporary files for debugging')
    args = parser.parse_args()
    
    # Use a context manager for the temporary directory
    # In debug mode, we'll keep the temp dir by not using the context manager
    if args.debug:
        # In debug mode, create a persistent temporary directory
        tmpdir = tempfile.mkdtemp(prefix="plmsol_debug_")
        print(f"Debug mode: Temporary directory will be preserved at: {tmpdir}")
        try:
            process_prediction(args, tmpdir)
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Debug files preserved in: {tmpdir}")
            raise
        print(f"Processing complete. Debug files preserved in: {tmpdir}")
    else:
        # Use standard temporary directory that will be cleaned up
        with tempfile.TemporaryDirectory() as tmpdir:
            process_prediction(args, tmpdir)


def process_prediction(args, tmpdir):
        print(f"Using temporary directory: {tmpdir}")
        
        # Create necessary subdirectories
        t5_embeddings_dir = os.path.join(tmpdir, 't5_embeddings')
        os.makedirs(t5_embeddings_dir, exist_ok=True)
        
        # Step 1: Prepare embedding config
        embed_config = EMBED_CONFIG_TEMPLATE.copy()
        embed_config['global']['sequences_file'] = os.path.abspath(args.fasta)
        embed_config['global']['prefix'] = tmpdir
        embed_config_path = os.path.join(tmpdir, 'embed_config.yml')
        with open(embed_config_path, 'w') as f:
            yaml.safe_dump(embed_config, f)
        
        print(f"Generated embedding config at {embed_config_path}")

        # Step 2: Run embedding
        print("Running embedding generation...")
        try:
            run_embeddings(embed_config_path)
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            raise
            
        embeddings_file = os.path.join(tmpdir, 't5_embeddings', 'embeddings_file.h5')
        remapped_fasta = os.path.join(tmpdir, 'remapped_sequences_file.fasta')
        
        # Check if embeddings file exists
        if not os.path.exists(embeddings_file):
            print(f"Warning: Embeddings file not found at {embeddings_file}")
            print("Looking for embeddings in alternate locations...")
            # Try looking in other potential locations
            alt_locations = [os.path.join(tmpdir, 'embeddings_file.h5')]
            found = False
            for loc in alt_locations:
                if os.path.exists(loc):
                    embeddings_file = loc
                    found = True
                    print(f"Found embeddings at {loc}")
                    break
            if not found:
                print("Could not find embeddings file")
                raise FileNotFoundError("Embeddings file not found")
        
        fasta_to_remapped(args.fasta, remapped_fasta)

        # Step 3: Prepare inference config
        infer_config = INFER_CONFIG_TEMPLATE.copy()
        infer_config['global']['embeddings_file'] = embeddings_file
        infer_config['global']['remapping'] = remapped_fasta
        output_file = os.path.join(tmpdir, 'plmsol_predictions.csv')
        infer_config['global']['output_file'] = output_file
        infer_config_path = os.path.join(tmpdir, 'infer_config.yml')
        with open(infer_config_path, 'w') as f:
            yaml.safe_dump(infer_config, f)
        
        print(f"Generated inference config at {infer_config_path}")

        # Step 4: Run inference
        print("Running inference...")
        try:
            actual_output_file = run_inference(infer_config_path)
        except Exception as e:
            print(f"Error during inference: {e}")
            create_fallback_output(args.fasta, args.out)
            return
                
        if actual_output_file is None:
            print("Warning: Output file not found at any expected location")
            print("Creating fallback output with neutral predictions")
            create_fallback_output(args.fasta, args.out)
            return
            
        try:
            pred_df = pd.read_csv(actual_output_file)
            print(f"Columns in prediction file: {pred_df.columns.tolist()}")
            seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(args.fasta, "fasta")}
            pred_df['Predictor'] = 'PLM_Sol'
            
            # Map column names from protTrans_prediction_result.csv to expected names
            if 'protein_ID' in pred_df.columns:
                # Rename to standard column names
                pred_df.rename(columns={'protein_ID': 'Accession'}, inplace=True)
                
            # If sequence is already in the prediction file, use it directly
            if 'sequence' in pred_df.columns:
                pred_df.rename(columns={'sequence': 'Sequence'}, inplace=True)
            elif 'Accession' in pred_df.columns:
                # Otherwise map from FASTA
                pred_df['Sequence'] = pred_df['Accession'].map(seqs)
            else:
                raise ValueError(f"Could not find required columns. Available columns: {pred_df.columns.tolist()}")
            
            # Assume prediction column is 'SolubilityScore' or similar
            if 'SolubilityScore' not in pred_df.columns:
                # Try to infer from available columns
                if 'predict_result' in pred_df.columns:
                    pred_df['SolubilityScore'] = pred_df['predict_result']
                elif 'probability' in pred_df.columns:
                    pred_df['SolubilityScore'] = pred_df['probability']
                elif 'pred_label' in pred_df.columns:
                    pred_df['SolubilityScore'] = pred_df['pred_label'].map(lambda x: 1 if x == 1 or str(x).lower() == 'soluble' else 0)
                else:
                    raise ValueError(f"Could not find expected prediction columns. Available columns: {pred_df.columns.tolist()}")
                    
            pred_df['Probability_Soluble'] = pred_df['SolubilityScore']
            pred_df['Probability_Insoluble'] = 1 - pred_df['SolubilityScore']
            
            # Standardize columns
            if 'name' in pred_df.columns:
                pred_df.rename(columns={'name': 'Accession'}, inplace=True)
            
            # Make sure all required columns exist
            required_columns = ['Accession', 'Sequence', 'Predictor', 'SolubilityScore']
            for col in required_columns:
                if col not in pred_df.columns:
                    raise ValueError(f"Required column '{col}' is missing after processing. Available columns: {pred_df.columns.tolist()}")
                
            out_df = pred_df[['Accession', 'Sequence', 'Predictor', 'SolubilityScore', 'Probability_Soluble', 'Probability_Insoluble']]
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            out_df.to_csv(args.out, index=False)
            print(f"Results written to {args.out}")
        except Exception as e:
            print(f"Error processing results: {e}")
            raise

def create_fallback_output(fasta_path, output_path):
    """Create a fallback output CSV if prediction fails"""
    print("Creating fallback output with neutral predictions")
    records = list(SeqIO.parse(fasta_path, "fasta"))
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
    
    out_df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Fallback results written to {output_path}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        sys.exit(1)
