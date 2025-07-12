#!/usr/bin/env python3
"""
PLM_Sol Batch Predictor Wrapper
Standardizes output for benchmarking solubility predictors.

Usage:
  python plmsol_predict_wrapper.py --fasta <input_fasta> --out <output_csv>

Outputs CSV with columns:
  Accession, Sequence, Predictor, SolubilityScore, Probability_Soluble, Probability_Insoluble
"""
import argparse
import os
import subprocess
import tempfile
import shutil
import pandas as pd
from Bio import SeqIO
import yaml

# Helper to write a config YAML for embedding
EMBED_CONFIG_TEMPLATE = {
    'global': {
        'sequences_file': '',  # to be filled
        'prefix': ''           # to be filled
    }
}

# Helper to write a config YAML for inference
INFER_CONFIG_TEMPLATE = {
    'global': {
        'model_config': './configs/inference_Sol_biLSTM_TextCNN.yml',
        'embeddings_file': '',   # to be filled
        'remapping': '',        # to be filled
        'output_file': ''       # to be filled
    }
}

def fasta_to_remapped(fasta_path, remapped_path):
    # PLM_Sol expects remapped_sequences_file.fasta in FASTA format
    shutil.copy(fasta_path, remapped_path)

def run_embeddings(config_path):
    cmd = [
        'python', 'generate_embeddings_memory_efficient.py',
        '--config', config_path
    ]
    subprocess.run(cmd, check=True)

def run_inference(config_path):
    cmd = [
        'python', 'inference.py',
        '--config', config_path
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Batch PLM_Sol predictor wrapper")
    parser.add_argument('--fasta', '-f', required=True, help='Input FASTA file')
    parser.add_argument('--out', '-o', required=True, help='Output CSV file')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Prepare embedding config
        embed_config = EMBED_CONFIG_TEMPLATE.copy()
        embed_config['global']['sequences_file'] = os.path.abspath(args.fasta)
        embed_config['global']['prefix'] = tmpdir
        embed_config_path = os.path.join(tmpdir, 'embed_config.yml')
        with open(embed_config_path, 'w') as f:
            yaml.safe_dump(embed_config, f)

        # Step 2: Run embedding
        run_embeddings(embed_config_path)
        embeddings_file = os.path.join(tmpdir, 't5_embeddings', 'embeddings_file.h5')
        remapped_fasta = os.path.join(tmpdir, 'remapped_sequences_file.fasta')
        fasta_to_remapped(args.fasta, remapped_fasta)

        # Step 3: Prepare inference config
        infer_config = INFER_CONFIG_TEMPLATE.copy()
        infer_config['global']['embeddings_file'] = embeddings_file
        infer_config['global']['remapping'] = remapped_fasta
        infer_config['global']['output_file'] = os.path.join(tmpdir, 'plmsol_predictions.csv')
        infer_config_path = os.path.join(tmpdir, 'infer_config.yml')
        with open(infer_config_path, 'w') as f:
            yaml.safe_dump(infer_config, f)

        # Step 4: Run inference
        run_inference(infer_config_path)

        # Step 5: Parse predictions and write standardized CSV
        pred_df = pd.read_csv(infer_config['global']['output_file'])
        seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(args.fasta, "fasta")}
        pred_df['Predictor'] = 'PLM_Sol'
        pred_df['Sequence'] = pred_df['Accession'].map(seqs)
        # Assume prediction column is 'SolubilityScore' or similar
        if 'SolubilityScore' not in pred_df.columns:
            # Try to infer from available columns
            if 'probability' in pred_df.columns:
                pred_df['SolubilityScore'] = pred_df['probability']
            elif 'pred_label' in pred_df.columns:
                pred_df['SolubilityScore'] = pred_df['pred_label'].map(lambda x: 1 if x == 1 or str(x).lower() == 'soluble' else 0)
            else:
                pred_df['SolubilityScore'] = 0.0
        pred_df['Probability_Soluble'] = pred_df['SolubilityScore']
        pred_df['Probability_Insoluble'] = 1 - pred_df['SolubilityScore']
        # Standardize columns
        pred_df.rename(columns={'name': 'Accession'}, inplace=True)
        out_df = pred_df[['Accession', 'Sequence', 'Predictor', 'SolubilityScore', 'Probability_Soluble', 'Probability_Insoluble']]
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Results written to {args.out}")

if __name__ == '__main__':
    main()
