#!/usr/bin/env python3
"""
PLM_Sol Server Wrapper

This wrapper accepts pre-computed embeddings from the persistent embedding server
and passes them to the modified PLM_Sol inference function, eliminating the 30s
startup overhead by skipping embedding generation.

Usage:
    python plmsol_server_wrapper.py --fasta input.fasta --out output.csv --embeddings_file embeddings.json

Note: Model checkpoint is now hardcoded to /home/david_nunn/PLM_Sol/saved_models/model-10.t7
"""

import os
import sys
import json
import argparse
import tempfile
import pandas as pd
from pathlib import Path

# Add PLM_Sol to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import inference
import yaml


def create_inference_config(model_checkpoint, fasta_file, output_file):
    """
    Create minimal inference configuration for PLM_Sol.
    
    Args:
        model_checkpoint: Path to model checkpoint
        fasta_file: Path to input FASTA file
        output_file: Path for output CSV file
        
    Returns:
        Args object with minimal configuration
    """
    
    # Complete configuration for inference (based on working fine-tuning config)
    config = {
        'checkpoint': model_checkpoint,
        'embeddings': fasta_file,  # Will be overridden by server embeddings
        'remapping': fasta_file,   # Will be overridden by server embeddings
        'key_format': 'fasta_descriptor',
        'batch_size': 1,
        'output_files_name': output_file,
        'log_iterations': 100,
        'n_draws': 1000,
        'target': 'sol',
        'unknown_solubility': False,
        'model_type': 'biLSTM_TextCNN',
        'model_parameters': {
            'output_dim': 1,
            'dropout': 0.25,
            'kernel_size': 9
        },
        # CRITICAL: Add missing optimizer_parameters (required by Solver)
        'optimizer': 'Adam',  # Default optimizer
        'optimizer_parameters': {
            'lr': 1.0e-4  # Default learning rate
        },
        # CRITICAL: Add missing embedding_mode (required by inference)
        'embedding_mode': 'lm',  # Language model embeddings
        # CRITICAL: Add missing checkpoints_list (required by inference.py)
        'checkpoints_list': [model_checkpoint],
        # CRITICAL: Add missing training parameters (may be referenced)
        'num_epochs': 1,  # Minimal for inference
        'patience': 1,    # Minimal for inference
        # Additional fields that may be needed
        'experiment_name': 'server_inference',
        'exp_name': 'server_inference'
    }
    
    # Convert to namespace object
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    return Args(**config)


def run_plm_sol_with_server_embeddings(fasta_file, embeddings_file, model_checkpoint, output_file):
    """
    Run PLM_Sol inference using server-provided embeddings.
    
    Args:
        fasta_file: Path to input FASTA file
        embeddings_file: Path to JSON file with embeddings from server
        model_checkpoint: Path to PLM_Sol model checkpoint
        output_file: Path for output CSV file
        
    Returns:
        Path to output CSV file
    """
    
    # Parse embeddings from file
    try:
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        server_embeddings = embeddings_data['embeddings']
        print(f"Loaded {len(server_embeddings)} embeddings from file: {embeddings_file}")
    except Exception as e:
        raise ValueError(f"Failed to parse embeddings file {embeddings_file}: {e}")
    
    # Create inference configuration
    args = create_inference_config(model_checkpoint, fasta_file, output_file)
    
    print(f"Running PLM_Sol inference with server embeddings...")
    print(f"  Model: {model_checkpoint}")
    print(f"  Input: {fasta_file}")
    print(f"  Output: {output_file}")
    print(f"  Embeddings: {len(server_embeddings)} from server")
    
    try:
        # Call modified inference function with server embeddings
        results = inference(args, server_embeddings=server_embeddings)
        
        print(f"PLM_Sol inference completed successfully")
        print(f"Results saved to: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"PLM_Sol inference failed: {e}")
        raise


def main():
    """Main function for server wrapper."""
    parser = argparse.ArgumentParser(description='PLM_Sol Server Wrapper')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--out', required=True, help='Output CSV file')
    parser.add_argument('--embeddings_file', required=True, help='Path to JSON file with server embeddings')

    args = parser.parse_args()
    
    # Hardcoded model checkpoint path
    MODEL_CHECKPOINT = '/home/david_nunn/PLM_Sol/saved_models/model-10.t7'

    # Validate inputs
    if not os.path.exists(args.fasta):
        print(f"Error: FASTA file not found: {args.fasta}")
        return 1

    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"Error: Model checkpoint not found: {MODEL_CHECKPOINT}")
        return 1

    try:
        # Run PLM_Sol with server embeddings
        output_file = run_plm_sol_with_server_embeddings(
            args.fasta,
            args.embeddings_file,
            MODEL_CHECKPOINT,
            args.out
        )

        print(f"Success: PLM_Sol predictions saved to {output_file}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
