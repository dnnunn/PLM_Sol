#!/usr/bin/env python3
"""
Direct inference script for PLM_Sol using precomputed embeddings and remapped FASTA.
Allows easy model switching and outputs a standardized CSV for benchmarking or integration.
"""
import argparse
import os
import torch
import yaml
import pandas as pd
from datasets.embeddings_dataset import Embeddings_predict_Dataset
from solver import Solver
from models import *


def parse_args():
    parser = argparse.ArgumentParser(description="PLM_Sol direct inference on precomputed embeddings")
    parser.add_argument('--embeddings', required=True, help='Path to .h5 embeddings file')
    parser.add_argument('--remapping', required=True, help='Path to remapped_sequences_file.fasta')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth or .t7)')
    parser.add_argument('--config', required=True, help='YAML config file for model/eval parameters')
    parser.add_argument('--out', required=True, help='Output CSV file')
    parser.add_argument('--key_format', default='fasta_descriptor', help='Key format for embeddings (must be "fasta_descriptor" to match wrapper/configs)')
    parser.add_argument('--max_length', type=int, default=4000, help='Maximum sequence length (default: 4000)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    return parser.parse_args()


def main():
    args = parse_args()
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with CLI args if provided
    key_format = args.key_format or config.get('key_format', 'id')
    max_length = args.max_length or config.get('max_length', 4000)
    batch_size = args.batch_size or config.get('batch_size', 64)

    # Prepare dataset
    dataset = Embeddings_predict_Dataset(
        embeddings_path=args.embeddings,
        remapped_sequences=args.remapping,
        key_format=key_format,
        max_length=max_length,
        embedding_mode='lm',
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model_type = config.get('model', 'biLSTM_TextCNN')
    model_params = config.get('model_parameters', {})
    model = globals()[model_type](**model_params)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # Run inference
    all_ids = []
    all_sequences = []
    all_scores = []
    all_probs_sol = []
    all_probs_insol = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings, metadata = batch
            outputs = model(embeddings)
            probs = outputs.squeeze().cpu().numpy()
            for i, meta in enumerate(metadata):
                all_ids.append(meta['id'])
                all_sequences.append(meta['sequence'])
                score = float(probs[i])
                all_scores.append(score)
                all_probs_sol.append(score)
                all_probs_insol.append(1.0 - score)

    df = pd.DataFrame({
        'Accession': all_ids,
        'Sequence': all_sequences,
        'Predictor': 'PLM_Sol',
        'SolubilityScore': all_scores,
        'Probability_Soluble': all_probs_sol,
        'Probability_Insoluble': all_probs_insol
    })
    df.to_csv(args.out, index=False)
    print(f"Results saved to: {args.out}")

if __name__ == '__main__':
    main()
