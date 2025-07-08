#!/usr/bin/env python
"""
Memory-efficient script to generate embeddings using ProtTransT5XLU50Embedder.
This script processes sequences one by one to minimize memory usage.
"""

import os
import sys
import h5py
import torch
import numpy as np
from pathlib import Path
from Bio import SeqIO
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Generate embeddings with memory efficiency')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the embedding configuration YAML file')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for embedding generation (default: 1)')
    parser.add_argument('--half_precision', action='store_true',
                        help='Use half precision (fp16) to reduce memory usage')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract paths from config
    sequences_file = config['global']['sequences_file']
    output_prefix = config['global']['prefix']
    
    # Create output directories
    embeddings_dir = os.path.join(output_prefix, 't5_embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Output paths
    embeddings_file = os.path.join(embeddings_dir, 'embeddings_file.h5')
    
    # Check if remapped sequences file exists, otherwise use the original
    remapped_file = os.path.join(output_prefix, 'remapped_sequences_file.fasta')
    if os.path.exists(remapped_file):
        sequences_to_embed = remapped_file
        print(f"Using remapped sequences from {remapped_file}")
    else:
        sequences_to_embed = sequences_file
        print(f"Using original sequences from {sequences_file}")
    
    # Import the embedder here to avoid loading the model until necessary
    print("Importing ProtTransT5XLU50Embedder...")
    try:
        from bio_embeddings.embed import ProtTransT5XLU50Embedder
    except ImportError as e:
        print(f"Error importing ProtTransT5XLU50Embedder: {e}")
        sys.exit(1)
    
    # Create embedder with memory-efficient settings
    print("Creating embedder (this will load the model, which may take time)...")
    embedder = ProtTransT5XLU50Embedder(
        half_precision_model=args.half_precision,
        half_precision=args.half_precision
    )
    
    # Count sequences for progress reporting
    sequence_count = 0
    with open(sequences_to_embed, 'r') as f:
        for line in f:
            if line.startswith('>'):
                sequence_count += 1
    
    print(f"Found {sequence_count} sequences to embed")
    
    # Process sequences and generate embeddings
    print(f"Generating embeddings with batch size {args.batch_size}...")
    
    # Create h5py file for storing embeddings
    with h5py.File(embeddings_file, 'w') as f:
        # Process sequences in batches to save memory
        batch = []
        batch_ids = []
        processed = 0
        
        for record in SeqIO.parse(sequences_to_embed, "fasta"):
            batch.append(str(record.seq))
            batch_ids.append(record.id)
            
            if len(batch) >= args.batch_size or processed + len(batch) == sequence_count:
                # Generate embeddings for the batch
                try:
                    embeddings = embedder.embed_many(batch)
                    
                    # Store embeddings in h5 file
                    for i, (seq_id, embedding) in enumerate(zip(batch_ids, embeddings)):
                        f.create_dataset(seq_id, data=embedding)
                    
                    processed += len(batch)
                    print(f"Processed {processed}/{sequence_count} sequences")
                    
                    # Clear batch
                    batch = []
                    batch_ids = []
                    
                    # Force garbage collection to free memory
                    if args.half_precision:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error generating embeddings for batch: {e}")
                    sys.exit(1)
    
    print(f"Embeddings successfully generated and saved to {embeddings_file}")
    print(f"You can now run inference with:")
    print(f"python inference.py --config {os.path.dirname(args.config)}/test_inference_config.yml")

if __name__ == "__main__":
    main()
