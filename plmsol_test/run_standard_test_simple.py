#!/usr/bin/env python
"""
Minimal PLM_Sol test script using the exact config format from working examples.
"""
import os
import subprocess
import time
import sys

# Constants
PLMSOL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDING_CONFIG = os.path.join(os.path.dirname(__file__), "standard_embedding_config.yml")
INFERENCE_CONFIG = os.path.join(os.path.dirname(__file__), "standard_inference_config.yml")

def main():
    """Run the complete test with existing config files."""
    print("=" * 50)
    print("PLM_Sol Standard Test with Pre-Created Config Files")
    print("=" * 50)
    
    # Step 1: Generate embeddings
    print("\n=== Step 1: Generating embeddings ===")
    embedding_cmd = f"python {os.path.join(PLMSOL_DIR, 'generate_embeddings_memory_efficient.py')} --config {EMBEDDING_CONFIG}"
    print(f"Running: {embedding_cmd}")
    
    embedding_result = subprocess.run(embedding_cmd, shell=True)
    if embedding_result.returncode != 0:
        print("Embedding generation failed!")
        sys.exit(1)
    
    # Step 2: Run inference from PLM_Sol root directory
    print("\n=== Step 2: Running inference ===")
    current_dir = os.getcwd()
    os.chdir(PLMSOL_DIR)
    print(f"Changed directory to: {os.getcwd()}")
    
    inference_cmd = f"python inference.py --config {INFERENCE_CONFIG}"
    print(f"Running: {inference_cmd}")
    
    inference_result = subprocess.run(inference_cmd, shell=True)
    
    # Return to original directory
    os.chdir(current_dir)
    
    if inference_result.returncode != 0:
        print("Inference failed!")
        sys.exit(1)
    
    # Step 3: Check for output file
    print("\n=== Step 3: Checking for output ===")
    output_file = os.path.join(PLMSOL_DIR, "protTrans_prediction_result.csv")
    
    if os.path.exists(output_file):
        print(f"SUCCESS! Output file found at: {output_file}")
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size} bytes")
        print("\nTo verify contents, run: head -n 10 {output_file}")
    else:
        print(f"ERROR: Output file not found at {output_file}")
        sys.exit(1)
    
    print("\n=== Test completed successfully ===")

if __name__ == "__main__":
    main()
