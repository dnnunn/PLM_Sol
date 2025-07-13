#!/usr/bin/env python
"""
Standard test script to verify PLM_Sol functionality with random proteins.
This follows the exact pattern of the working test_plmsol_functionality.py script.
"""
import os
import sys
import yaml
import pandas as pd
import subprocess
import time
from pathlib import Path

# Constants
PLMSOL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FASTA_PATH = os.path.join(os.path.dirname(__file__), "random_proteins_combined.fasta")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "standard_test_results")

def create_embedding_config(fasta_path, output_dir):
    """Create a bio-embeddings configuration file for ProtT5."""
    print("Creating embedding configuration...")
    
    # Get base names without extensions
    fasta_basename = os.path.basename(fasta_path).split('.')[0]
    output_prefix = os.path.join(output_dir, fasta_basename + "_emb")
    
    config = {
        "global": {
            "sequences_file": fasta_path,
            "prefix": output_prefix
        },
        "t5_embeddings": {
            "type": "embed",
            "protocol": "prottrans_t5_xl_u50",
            "half_precision_model": True,
            "half_precision": True
        },
        "annotations_from_t5": {
            "type": "extract",
            "protocol": "la_prott5",
            "depends_on": "t5_embeddings"
        }
    }
    
    config_path = os.path.join(output_dir, "standard_embedding_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Embedding configuration created at {config_path}")
    return config_path, output_prefix

def create_inference_config(embedding_path, remapping_path, model_param_path, output_dir):
    """Create an inference configuration file for PLM_Sol."""
    print("Creating inference configuration...")
    
    config = {
        "batch_size": 1,
        "checkpoints_list": [model_param_path],
        "embeddings": embedding_path,
        "key_format": "fasta_descriptor",  # Match the working test scripts
        "log_iterations": 100,
        "n_draws": 1000,
        "output_files_name": "standard_test_inference",
        "remapping": remapping_path
    }
    
    config_path = os.path.join(output_dir, "standard_inference_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Inference configuration created at {config_path}")
    return config_path

def run_embedding_generation(config_path):
    """Run the embedding generation step."""
    print("\n=== Step 1: Generating embeddings ===")
    start_time = time.time()
    
    cmd = f"python {os.path.join(PLMSOL_DIR, 'generate_embeddings_memory_efficient.py')} --config {config_path}"
    print(f"Running command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Embedding stdout:")
    print(result.stdout)
    
    if result.stderr:
        print("Embedding stderr:")
        print(result.stderr)
        
    if result.returncode != 0:
        print(f"ERROR: Embedding failed with code {result.returncode}")
        sys.exit(1)
        
    end_time = time.time()
    print(f"Embedding completed in {end_time - start_time:.2f} seconds")
    return end_time - start_time

def run_inference(config_path):
    """Run the inference step."""
    print("\n=== Step 2: Running inference ===")
    start_time = time.time()
    
    # Important: Run inference from PLM_Sol root directory
    original_dir = os.getcwd()
    os.chdir(PLMSOL_DIR)
    print(f"Changed working directory to: {os.getcwd()}")
    
    cmd = f"python inference.py --config {config_path}"
    print(f"Running command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Inference stdout:")
    print(result.stdout)
    
    if result.stderr:
        print("Inference stderr:")
        print(result.stderr)
        
    # Return to original directory
    os.chdir(original_dir)
    
    if result.returncode != 0:
        print(f"ERROR: Inference failed with code {result.returncode}")
        sys.exit(1)
        
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds")
    return end_time - start_time

def convert_output_format():
    """Convert the output to benchmark format."""
    print("\n=== Step 3: Converting output format ===")
    start_time = time.time()
    
    # The hardcoded output file is always in the PLM_Sol root directory
    prediction_file = os.path.join(PLMSOL_DIR, "protTrans_prediction_result.csv")
    print(f"Looking for prediction file at: {prediction_file}")
    
    if os.path.exists(prediction_file):
        # Read the prediction file
        try:
            pred_df = pd.read_csv(prediction_file)
            print(f"Read {len(pred_df)} predictions from {prediction_file}")
            
            # Create standardized output
            results = []
            for _, row in pred_df.iterrows():
                results.append({
                    'Accession': row['protein_ID'],
                    'Sequence': row['sequence'],
                    'Predictor': 'PLM_Sol',
                    'SolubilityScore': row['predict_result'],
                    'Probability_Soluble': row['predict_result'],
                    'Probability_Insoluble': 1.0 - row['predict_result']
                })
            
            # Create output DataFrame with required columns
            out_df = pd.DataFrame(results)
            required_columns = ['Accession', 'Sequence', 'Predictor', 'SolubilityScore', 
                               'Probability_Soluble', 'Probability_Insoluble']
            
            # Create output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Save the standardized results
            output_path = os.path.join(OUTPUT_DIR, "random_proteins_benchmark.csv")
            out_df[required_columns].to_csv(output_path, index=False)
            print(f"Results converted and written to {output_path}")
            
            # Also copy the original output for reference
            original_copy = os.path.join(OUTPUT_DIR, "original_prediction_result.csv")
            pred_df.to_csv(original_copy, index=False)
            print(f"Original output copied to {original_copy}")
            
        except Exception as e:
            print(f"ERROR during conversion: {str(e)}")
            return 0
    else:
        print(f"ERROR: Prediction file {prediction_file} not found")
        return 0
        
    end_time = time.time()
    print(f"Output conversion completed in {end_time - start_time:.2f} seconds")
    return end_time - start_time

def main():
    print("=" * 50)
    print("PLM_Sol Standard Test with Random Proteins")
    print("=" * 50)
    print(f"Started at: {time.ctime()}")
    
    start_time = time.time()
    
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create embedding configuration
        embedding_config_path, output_prefix = create_embedding_config(FASTA_PATH, os.path.dirname(__file__))
        
        # Expected paths after embedding generation
        expected_embedding_path = f"{output_prefix}/t5_embeddings/embeddings_file.h5"
        expected_remapping_path = f"{output_prefix}/remapped_sequences_file.fasta"
        model_param_path = os.path.join(PLMSOL_DIR, "model_param/model_param.t7")
        
        # Create inference configuration
        inference_config_path = create_inference_config(
            expected_embedding_path, 
            expected_remapping_path,
            model_param_path,
            os.path.dirname(__file__)
        )
        
        # Run embedding generation
        embedding_time = run_embedding_generation(embedding_config_path)
        
        # Run inference
        inference_time = run_inference(inference_config_path)
        
        # Convert output
        conversion_time = convert_output_format()
        
        # Print summary
        end_time = time.time()
        print("\n=== Test Summary ===")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Embedding time: {embedding_time:.2f} seconds")
        print(f"Inference time: {inference_time:.2f} seconds")
        print(f"Conversion time: {conversion_time:.2f} seconds")
        print(f"Finished at: {time.ctime()}")
        
    except Exception as e:
        print(f"ERROR: Test failed with exception: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()
