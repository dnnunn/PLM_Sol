#!/usr/bin/env python3
"""
Direct inference script for PLM_Sol that bypasses the normal inference pipeline
and loads the model and embeddings directly.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import h5py
import traceback
import argparse
from models.biLSTM_TextCNN import biLSTM_TextCNN

def load_model(model_path):
    """Load the PLM_Sol model directly"""
    try:
        print(f"Loading model from {model_path}")
        model = biLSTM_TextCNN(embeddings_dim=1024, hidden_dim=512, dropout=0.5, max_len=1000)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint loaded, keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'state_dict'}")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading model from state dict")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Loading direct model state")
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        return model, device
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        return None, None

def load_embeddings(embeddings_file, remapping_file):
    """Load embeddings and mapping from files"""
    try:
        print(f"Loading embeddings from {embeddings_file}")
        
        # Load embeddings
        with h5py.File(embeddings_file, 'r') as f:
            keys = list(f.keys())
            print(f"Found keys in h5 file: {keys}")
            # Assuming the embeddings are stored with protein IDs as keys
            embeddings = {}
            for key in keys:
                embeddings[key] = np.array(f[key])
        
        print(f"Loaded embeddings for {len(embeddings)} proteins")
        
        # Load protein IDs and sequences
        if remapping_file:
            print(f"Loading sequence mapping from {remapping_file}")
            sequences = {}
            with open(remapping_file, 'r') as f:
                header = None
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        header = line[1:].split()[0]  # Remove '>' and take first word as ID
                    elif header:
                        sequences[header] = line
                        header = None
            print(f"Loaded {len(sequences)} sequences")
        else:
            sequences = {key: f"seq_{key}" for key in embeddings}
        
        return embeddings, sequences
        
    except Exception as e:
        print(f"ERROR loading embeddings: {e}")
        traceback.print_exc()
        return None, None

def run_direct_inference(model, embeddings, sequences, device):
    """Run inference directly using the model and embeddings"""
    try:
        print("Running direct inference")
        results = []
        
        with torch.no_grad():
            for protein_id, embedding in embeddings.items():
                if protein_id not in sequences:
                    print(f"Warning: No sequence found for {protein_id}, using placeholder")
                    sequence = f"sequence_{protein_id}"
                else:
                    sequence = sequences[protein_id]
                
                # Convert embedding to tensor and add batch dimension
                tensor = torch.tensor(embedding, dtype=torch.float).to(device)
                tensor = tensor.unsqueeze(0)  # Add batch dimension
                
                # Forward pass
                output = model(tensor)
                
                # Get probability
                probability = torch.sigmoid(output).item()
                
                results.append({
                    'protein_ID': protein_id,
                    'sequence': sequence,
                    'predict_result': probability
                })
                print(f"Protein {protein_id}: probability = {probability:.4f}")
        
        return results
        
    except Exception as e:
        print(f"ERROR during inference: {e}")
        traceback.print_exc()
        return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Direct PLM_Sol inference")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings H5 file")
    parser.add_argument("--model", default="model_param/model_param.t7", help="Path to model file")
    parser.add_argument("--remapping", help="Path to remapped sequences FASTA file")
    parser.add_argument("--output", default="direct_prediction_result.csv", help="Output file path")
    
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model)
    if model is None:
        print("Failed to load model, exiting")
        sys.exit(1)
    
    # Load embeddings
    embeddings, sequences = load_embeddings(args.embeddings, args.remapping)
    if embeddings is None:
        print("Failed to load embeddings, exiting")
        sys.exit(1)
    
    # Run inference
    results = run_direct_inference(model, embeddings, sequences, device)
    if not results:
        print("No results generated, exiting")
        sys.exit(1)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main: {e}")
        traceback.print_exc()
        sys.exit(1)
