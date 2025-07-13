#!/usr/bin/env python
"""
Test script to verify PLM_Sol functionality with lacZ sequences.
"""
import os
import sys
import yaml
from pathlib import Path
import argparse

def create_embedding_config(fasta_path, output_dir):
    """Create a bio-embeddings configuration file for lacZ."""
    print("\nCreating lacZ embedding configuration...")
    
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
        }
    }
    
    config_path = os.path.join(output_dir, "lacZ_embedding_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Embedding configuration created at {config_path}")
    return config_path, output_prefix

def create_inference_config(embedding_path, remapping_path, model_param_path, output_dir):
    """Create an inference configuration file for PLM_Sol."""
    print("\nCreating lacZ inference configuration...")
    
    config = {
        "output_files_name": "lacZ_inference",
        "log_iterations": 100,
        "n_draws": 1000,
        "batch_size": 1,
        "checkpoints_list": [model_param_path],
        "embeddings": embedding_path,
        "remapping": remapping_path,
        "key_format": "fasta_descriptor"
    }
    
    config_path = os.path.join(output_dir, "lacZ_inference_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Inference configuration created at {config_path}")
    return config_path

def main():
    print("=" * 50)
    print("PLM_Sol lacZ Test")
    print("=" * 50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run PLM_Sol test with lacZ')
    parser.add_argument('--test_dir', type=str, default='plmsol_test',
                        help='Directory containing test files')
    args = parser.parse_args()
    
    # Set up paths
    test_dir = os.path.abspath(args.test_dir)
    lacZ_fasta = os.path.join(test_dir, "lacZ.fasta")
    model_param_path = "./model_param/model_param.t7"
    
    # Check if lacZ.fasta exists
    if not os.path.exists(lacZ_fasta):
        print(f"❌ lacZ.fasta not found at {lacZ_fasta}")
        return False
    
    try:
        # Create embedding configuration
        embedding_config_path, output_prefix = create_embedding_config(lacZ_fasta, test_dir)
        
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
        
        print("\n" + "=" * 50)
        print("Test Setup Complete")
        print("=" * 50)
        print("\nRun the following commands to test PLM_Sol with lacZ:")
        print(f"\n1. Generate embeddings (with overwrite):")
        print(f"   bio_embeddings {embedding_config_path} --overwrite")
        print(f"\n2. Run inference:")
        print(f"   python inference.py --config {inference_config_path}")
        print("\nThe output should be a file named 'lacZ_inference_prediction_result.csv' with predictions.")
        
    except Exception as e:
        print(f"\n❌ Error during test setup: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
