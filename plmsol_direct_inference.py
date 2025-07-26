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
from torch.utils.data import DataLoader
from datasets.embeddings_dataset import Embeddings_predict_Dataset
from datasets.transforms import predict_ToTensor, Solubility_predict_ToInt
from torchvision.transforms import transforms
from solver import Solver
from models import *
import yaml


def load_model(config_path, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load the model from config and checkpoint"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model_class = globals()[config['model_type']]
    model_params = config.get('model_parameters', {})
    
    # Special handling for biLSTM_TextCNN
    if config['model_type'] == 'biLSTM_TextCNN':
        model_params['embeddings_dim'] = 1024  # T5 embedding dimension
    
    model = model_class(**model_params)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model, config


def predict(embeddings_path, remapping_path, config_path, checkpoint_path, output_path, key_format='id', batch_size=128):
    """Run prediction on pre-computed embeddings"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model, config = load_model(config_path, checkpoint_path, device)
    
    # Set up data loading
    transform = transforms.Compose([Solubility_predict_ToInt(), predict_ToTensor()])
    dataset = Embeddings_predict_Dataset(
        embeddings_path, 
        remapping_path,
        key_format=key_format,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: (torch.stack([item[0] for item in x]), [item[1] for item in x])
    )
    
    # Run inference
    all_predictions = []
    all_sequences = []
    all_ids = []
    
    with torch.no_grad():
        for batch in loader:
            inputs, batch_info = batch
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            # Extract sequence info
            batch_ids = [info['id'] for info in batch_info]
            batch_seqs = [info['sequence'] for info in batch_info]
            
            all_predictions.extend(predictions)
            all_sequences.extend(batch_seqs)
            all_ids.extend(batch_ids)
    
    # Create output DataFrame
    results = pd.DataFrame({
        'Accession': all_ids,
        'Sequence': all_sequences,
        'Predictor': 'PLM_Sol',
        'SolubilityScore': all_predictions,
        'Probability_Soluble': all_predictions,
        'Probability_Insoluble': 1 - np.array(all_predictions)
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
    parser.add_argument('--key_format', default='id', choices=['id', 'hash', 'fasta_descriptor'], 
                       help='Key format in the embeddings file')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    
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
