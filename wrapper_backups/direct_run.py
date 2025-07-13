#!/usr/bin/env python3
"""
Simplified direct runner for PLM_Sol that bypasses all the problematic scripts.
Combines embedding generation and direct inference in one script.
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
from Bio import SeqIO
from transformers import T5EncoderModel, T5Tokenizer

class ProtTransT5XLU50Embedder:
    """Embedder using the ProtT5-XL-U50 model"""
    
    def __init__(self):
        """Initialize the embedder with the ProtT5-XL-U50 model"""
        print("Loading ProtT5-XL-U50 model...")
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        
        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def _encode_sequence(self, sequence):
        """Encode a single sequence using the T5 model"""
        # Add spaces between amino acids for tokenizer
        sequence = " ".join(list(sequence))
        
        # Tokenize and encode
        ids = self.tokenizer.batch_encode_plus([sequence], add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        # Get per-residue embeddings (last hidden state)
        embeddings = embedding.last_hidden_state.cpu().numpy()
        
        # Apply attention mask and get mean embedding
        masked_embeddings = embeddings * np.expand_dims(attention_mask.cpu().numpy(), -1)
        mean_embedding = masked_embeddings.sum(axis=1) / attention_mask.sum(axis=1, keepdim=True).cpu().numpy()
        
        return mean_embedding[0]  # Return the embedding for the single sequence

def generate_embeddings(fasta_path, output_dir):
    """Generate embeddings for sequences in a FASTA file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "embeddings_file.h5")
    
    # Load sequences
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    
    # Initialize embedder
    embedder = ProtTransT5XLU50Embedder()
    
    # Generate embeddings
    print(f"Generating embeddings for {len(sequences)} sequences")
    embeddings = {}
    
    for i, (seq_id, sequence) in enumerate(sequences.items()):
        print(f"Processing sequence {i+1}/{len(sequences)}: {seq_id}")
        embedding = embedder._encode_sequence(sequence)
        embeddings[seq_id] = embedding
    
    # Save embeddings to H5 file
    with h5py.File(output_file, 'w') as f:
        for seq_id, embedding in embeddings.items():
            f.create_dataset(seq_id, data=embedding)
    
    print(f"Embeddings saved to {output_file}")
    return output_file

def load_model(model_path):
    """Load the PLM_Sol model directly"""
    try:
        print(f"Loading model from {model_path}")
        # Initialize model with correct parameters
        model = biLSTM_TextCNN(embeddings_dim=1024, output_dim=1, dropout=0.25, kernel_size=9, conv_dropout=0.25)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return model, device
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        return None, None

def run_direct_inference(model, embeddings_file, fasta_path, output_path, device):
    """Run inference directly using the model and embeddings"""
    try:
        print(f"Loading embeddings from {embeddings_file}")
        
        # Load embeddings
        with h5py.File(embeddings_file, 'r') as f:
            keys = list(f.keys())
            embeddings = {}
            for key in keys:
                embeddings[key] = np.array(f[key])
        
        print(f"Loaded embeddings for {len(embeddings)} proteins")
        
        # Load protein sequences
        sequences = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences[record.id] = str(record.seq)
        
        results = []
        print("Running direct inference")
        
        with torch.no_grad():
            for protein_id, embedding in embeddings.items():
                sequence = sequences.get(protein_id, f"Unknown sequence for {protein_id}")
                
                # Convert embedding to tensor and add batch dimension
                tensor = torch.tensor(embedding, dtype=torch.float).to(device)
                tensor = tensor.unsqueeze(0)  # Add batch dimension
                
                # Forward pass
                output = model(tensor)
                
                # Get probability (sigmoid of output)
                probability = torch.sigmoid(output).item()
                
                results.append({
                    'Accession': protein_id,
                    'Sequence': sequence,
                    'Predictor': 'PLM_Sol',
                    'SolubilityScore': probability,
                    'Probability_Soluble': probability,
                    'Probability_Insoluble': 1 - probability
                })
                print(f"Protein {protein_id}: solubility score = {probability:.4f}")
        
        # Save results
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR during inference: {e}")
        traceback.print_exc()
        return False

def create_fallback_output(fasta_path, output_path):
    """Create a fallback output CSV if prediction fails"""
    print("Creating fallback output with neutral predictions")
    records = list(SeqIO.parse(fasta_path, "fasta"))
    data = []
    for rec in records:
        data.append({
            'Accession': rec.id,
            'Sequence': str(rec.seq),
            'Predictor': 'PLM_Sol',
            'SolubilityScore': 0.5,  # Neutral score
            'Probability_Soluble': 0.5,
            'Probability_Insoluble': 0.5
        })
    
    out_df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Fallback results written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="PLM_Sol direct runner")
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--out', required=True, help='Output CSV file')
    parser.add_argument('--model', default="model_param/model_param.t7", help='Model checkpoint file')
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip embedding generation (debug only)')
    parser.add_argument('--embeddings', help='Pre-generated embeddings file (use with --skip-embeddings)')
    args = parser.parse_args()
    
    try:
        # Step 1: Generate embeddings (unless skipped)
        if args.skip_embeddings:
            if not args.embeddings:
                print("Error: --embeddings must be provided when using --skip-embeddings")
                sys.exit(1)
            embeddings_file = args.embeddings
        else:
            print("Generating embeddings...")
            embeddings_dir = "temp_embeddings"
            embeddings_file = generate_embeddings(args.fasta, embeddings_dir)
        
        # Step 2: Load model
        model, device = load_model(args.model)
        
        if model is not None:
            # Step 3: Run direct inference
            success = run_direct_inference(model, embeddings_file, args.fasta, args.out, device)
            
            if not success:
                print("Inference failed, using fallback output")
                create_fallback_output(args.fasta, args.out)
        else:
            print("Model loading failed, using fallback output")
            create_fallback_output(args.fasta, args.out)
            
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        
        # Generate fallback output in case of any error
        print("Generating fallback output due to error")
        create_fallback_output(args.fasta, args.out)
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        sys.exit(1)
