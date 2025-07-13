#!/usr/bin/env python
"""
Test script to verify PLM_Sol functionality with random proteins combined sequences.
"""
import os
import sys
import yaml
from pathlib import Path
import argparse
import subprocess

def create_embedding_config(fasta_path, output_dir):
    """Create a bio-embeddings configuration file for random proteins."""
    print("\nCreating random proteins embedding configuration...")
    
    # Get base names without extensions
    fasta_basename = os.path.basename(fasta_path).split('.')[0]
    output_prefix = os.path.join(output_dir, fasta_basename + "_emb")
    
    # Match exactly the structure of the working test_plmsol_functionality.py embedding config
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
        }
    }
    
    config_path = os.path.join(output_dir, "random_embedding_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Embedding configuration created at {config_path}")
    return config_path, output_prefix

def create_inference_config(embedding_path, remapping_path, model_param_path, output_dir):
    """Create an inference configuration file for PLM_Sol."""
    print("\nCreating random proteins inference configuration...")
    
    config = {
        "output_files_name": "random_inference",
        "log_iterations": 100,
        "n_draws": 1000,
        "batch_size": 1,
        "checkpoints_list": [model_param_path],
        "embeddings": embedding_path,
        "remapping": remapping_path,
        "key_format": "fasta_descriptor"
    }
    
    config_path = os.path.join(output_dir, "random_inference_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Inference configuration created at {config_path}")
    return config_path

def run_embedding_generation(config_path):
    """Run the embedding generation step."""
    print("\n=== Step 1: Generating embeddings ===")
    cmd = f"bio_embeddings {config_path} --overwrite"
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
    
    return True

def run_inference(config_path):
    """Run the inference step."""
    print("\n=== Step 2: Running inference ===")
    cmd = f"python inference.py --config {config_path}"
    print(f"Running command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Inference stdout:")
    print(result.stdout)
    
    if result.stderr:
        print("Inference stderr:")
        print(result.stderr)
        
    if result.returncode != 0:
        print(f"ERROR: Inference failed with code {result.returncode}")
        sys.exit(1)
    
    return True

def main():
    print("=" * 50)
    print("PLM_Sol Random Proteins Test")
    print("=" * 50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run PLM_Sol test with random proteins')
    parser.add_argument('--test_dir', type=str, default='plmsol_test',
                        help='Directory containing test files')
    args = parser.parse_args()
    
    # Set up paths
    test_dir = os.path.abspath(args.test_dir)
    fasta_path = os.path.join(test_dir, "random_proteins_combined.fasta")
    model_param_path = "./model_param/model_param.t7"
    
    # Check if random proteins FASTA exists
    if not os.path.exists(fasta_path):
        print(f"❌ random_proteins_combined.fasta not found at {fasta_path}")
        return False
    
    try:
        # Create embedding configuration
        embedding_config_path, output_prefix = create_embedding_config(fasta_path, test_dir)
        
        # Expected paths after embedding generation
        expected_embedding_path = f"{output_prefix}/t5_embeddings/embeddings_file.h5"
        expected_remapping_path = f"{output_prefix}/remapped_sequences_file.fasta"
        
        # Create inference configuration
        inference_config_path = create_inference_config(
            expected_embedding_path, 
            expected_remapping_path,
            model_param_path,
            test_dir
        )
        
        # Run embedding generation
        print("\nRunning embedding generation...")
        embedding_success = run_embedding_generation(embedding_config_path)
        
        if embedding_success:
            # Run inference
            print("\nRunning inference...")
            inference_success = run_inference(inference_config_path)
            
            # Check for output file
            output_file = "protTrans_prediction_result.csv"
            if os.path.exists(output_file):
                print(f"\n✓ SUCCESS! Output file found: {output_file}")
                print(f"File size: {os.path.getsize(output_file)} bytes")
            else:
                print(f"\n❌ ERROR: Output file {output_file} not found")
                return False
        
        print("\n" + "=" * 50)
        print("Test Complete")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
