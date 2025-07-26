#!/usr/bin/env python3
"""
Direct Inference Script for PLM_Sol
----------------------------------
This script runs PLM_Sol inference using pre-computed embeddings and a specified model checkpoint.
It's optimized for batch processing and produces standardized output.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from datasets.embeddings_dataset import Embeddings_predict_Dataset
from datasets.transforms import predict_ToTensor, Solubility_predict_ToInt
from torchvision.transforms import transforms
from solver import Solver
from models import *
import yaml
import copy


def predict(embeddings_path, remapping_path, config_path, checkpoint_path, output_path, 
           key_format='fasta_descriptor', batch_size=16):
    """Run prediction on pre-computed embeddings using the Solver class"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create args namespace similar to inference.py
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    args_dict = {
        'embeddings': embeddings_path,
        'remapping': remapping_path,
        'key_format': key_format,
        'batch_size': batch_size,
        'log_iterations': -1,  # Disable logging
        'distance_threshold': -1.0,  # Always use denovo predictions
        'model_type': config['model_type'],
        'model_parameters': config.get('model_parameters', {}),
        'optimizer': 'Adam',  # Default optimizer, won't be used for inference
        'optimizer_parameters': {'lr': 0.001},  # Default learning rate
        'checkpoint': checkpoint_path,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Initialize model
    model_class = globals()[config['model_type']]
    model = model_class(**args_dict['model_parameters'])
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=args_dict['device'])
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    
    model = model.to(args_dict['device'])
    model.eval()
    
    # Set up data loading
    transform = transforms.Compose([Solubility_predict_ToInt(), predict_ToTensor()])
    dataset = Embeddings_predict_Dataset(
        embeddings_path, 
        remapping_path,
        key_format=key_format,
        transform=transform
    )
    
    # Create args object for Solver
    solver_args = Args(**args_dict)
    
    # Initialize solver
    solver = Solver(model, solver_args, torch.optim.Adam)
    
    # Run prediction
    predictions, ids, sequences = solver.predict_evaluation(dataset)
    
    # Create output DataFrame
    results = pd.DataFrame({
        'Accession': ids,
        'Sequence': sequences,
        'Predictor': 'PLM_Sol',
        'SolubilityScore': predictions,
        'Probability_Soluble': predictions,
        'Probability_Insoluble': 1 - np.array(predictions)
    })
    
    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Run PLM_Sol inference with pre-computed embeddings')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings file (.h5)')
    parser.add_argument('--remapping', required=True, help='Path to remapping FASTA file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.t7 or .pth)')
    parser.add_argument('--config', required=True, help='Path to model config file (.yml)')
    parser.add_argument('--out', required=True, help='Output CSV file path')
    parser.add_argument('--key_format', default='fasta_descriptor', 
                       choices=['hash', 'fasta_descriptor', 'fasta_descriptor_old'], 
                       help='Key format in the embeddings file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Validate paths
    for path in [args.embeddings, args.remapping, args.checkpoint, args.config]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # Run prediction
    predict(
        embeddings_path=args.embeddings,
        remapping_path=args.remapping,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.out,
        key_format=args.key_format,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
