#!/usr/bin/env python3
"""
PLM_Sol Random Proteins Test with Timing
---------------------------------------
This script runs PLM_Sol on random_proteins_combined.fasta and measures execution time.
It also converts the output to the standard benchmark format.
"""
import os
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path
import datetime

# Configuration
FASTA_PATH = os.path.abspath("random_proteins_combined.fasta")
OUTPUT_DIR = os.path.abspath("random_results")
EMBEDDING_CONFIG = os.path.abspath("random_embedding_config.yml")
INFERENCE_CONFIG = os.path.abspath("random_inference_config.yml")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Record start time
print(f"=== PLM_Sol Random Proteins Test ===")
print(f"Started at: {datetime.datetime.now()}")
start_time = time.time()

# Step 1: Generate embeddings
print(f"\n=== Step 1: Generating embeddings ===")
embed_start = time.time()
embedding_cmd = [
    "python", 
    "../generate_embeddings_memory_efficient.py", 
    "--config", EMBEDDING_CONFIG
]
print(f"Running command: {' '.join(embedding_cmd)}")
embedding_result = subprocess.run(embedding_cmd, capture_output=True, text=True)
embed_end = time.time()

# Print embedding output
print(f"Embedding stdout:")
print(embedding_result.stdout)
if embedding_result.stderr:
    print(f"Embedding stderr:")
    print(embedding_result.stderr)

# Check if embedding succeeded
if embedding_result.returncode != 0:
    print(f"ERROR: Embedding generation failed with code {embedding_result.returncode}")
    sys.exit(1)

print(f"Embedding completed in {embed_end - embed_start:.2f} seconds")

# Step 2: Run inference
print(f"\n=== Step 2: Running inference ===")
inference_start = time.time()

# Important: Change to PLM_Sol directory for inference
plmsol_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(plmsol_dir)
print(f"Changed working directory to: {plmsol_dir}")

inference_cmd = [
    "python", 
    "inference.py", 
    "--config", INFERENCE_CONFIG
]
print(f"Running command: {' '.join(inference_cmd)}")
inference_result = subprocess.run(inference_cmd, capture_output=True, text=True)

# Change back to original directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
inference_end = time.time()

# Print inference output
print(f"Inference stdout:")
print(inference_result.stdout)
if inference_result.stderr:
    print(f"Inference stderr:")
    print(inference_result.stderr)

# Check if inference succeeded
if inference_result.returncode != 0:
    print(f"ERROR: Inference failed with code {inference_result.returncode}")
    sys.exit(1)

print(f"Inference completed in {inference_end - inference_start:.2f} seconds")

# Step 3: Convert output format
print(f"\n=== Step 3: Converting output format ===")
convert_start = time.time()

# The hardcoded output file from PLM_Sol is ALWAYS "protTrans_prediction_result.csv", regardless of config
# This is hardcoded in solver.py's predict_evaluation method
prediction_file = os.path.join(plmsol_dir, "protTrans_prediction_result.csv")
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
        
        # Save the standardized results
        output_path = os.path.join(OUTPUT_DIR, "random_proteins_benchmark.csv")
        out_df[required_columns].to_csv(output_path, index=False)
        print(f"Results converted and written to {output_path}")
        
        # Also copy the original output for reference
        original_copy = os.path.join(OUTPUT_DIR, "original_prediction_result.csv")
        pred_df.to_csv(original_copy, index=False)
        print(f"Original output copied to {original_copy}")
        
    except Exception as e:
        print(f"ERROR: Failed to convert output: {e}")
else:
    print(f"ERROR: Prediction file {prediction_file} not found")

convert_end = time.time()
print(f"Output conversion completed in {convert_end - convert_start:.2f} seconds")

# Calculate total time
end_time = time.time()
total_time = end_time - start_time

# Print summary
print(f"\n=== Test Summary ===")
print(f"Total execution time: {total_time:.2f} seconds")
print(f"Embedding time: {embed_end - embed_start:.2f} seconds")
print(f"Inference time: {inference_end - inference_start:.2f} seconds")
print(f"Conversion time: {convert_end - convert_start:.2f} seconds")
print(f"Finished at: {datetime.datetime.now()}")

# Write timing to file
with open(os.path.join(OUTPUT_DIR, "timing.txt"), "w") as f:
    f.write(f"PLM_Sol Random Proteins Test Timing\n")
    f.write(f"-----------------------------------\n")
    f.write(f"Started:  {datetime.datetime.fromtimestamp(start_time)}\n")
    f.write(f"Finished: {datetime.datetime.fromtimestamp(end_time)}\n")
    f.write(f"Total execution time: {total_time:.2f} seconds\n")
    f.write(f"Embedding time: {embed_end - embed_start:.2f} seconds\n")
    f.write(f"Inference time: {inference_end - inference_start:.2f} seconds\n")
    f.write(f"Conversion time: {convert_end - convert_start:.2f} seconds\n")
