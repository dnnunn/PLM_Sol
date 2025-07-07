#!/usr/bin/env python
# Test script to verify PLM_Sol functionality on the VM

import os
import sys
import tempfile
from pathlib import Path
import shutil
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO

def test_imports():
    """Test if all required dependencies are installed."""
    print("Testing imports...")
    try:
        import h5py
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        from Bio import SeqIO
        print("✓ Basic dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Error importing basic dependencies: {e}")
        return False
    
    try:
        # Try importing PLM_Sol specific modules
        from models.biLSTM_TextCNN import biLSTM_TextCNN
        from datasets.embeddings_dataset import Embeddings_predict_Dataset
        from datasets.transforms import Solubility_predict_ToInt, predict_ToTensor
        from torchvision.transforms import transforms
        from utils.general import AMINO_ACIDS
        from solver import Solver
        print("✓ PLM_Sol specific modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Error importing PLM_Sol specific modules: {e}")
        return False

def create_test_fasta(output_dir):
    """Create a test FASTA file with a few protein sequences."""
    print("Creating test FASTA file...")
    test_sequences = [
        ("test_protein_1", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"),
        ("test_protein_2", "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSWGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"),
        ("test_protein_3", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK")
    ]
    
    fasta_path = os.path.join(output_dir, "test_dataset.fasta")
    with open(fasta_path, "w") as f:
        for idx, (name, seq) in enumerate(test_sequences):
            f.write(f">{name}\n{seq}\n")
    
    print(f"✓ Test FASTA file created at {fasta_path}")
    return fasta_path

def create_embedding_config(fasta_path, output_dir):
    """Create a bio-embeddings configuration file for ProtT5."""
    print("Creating embedding configuration...")
    
    # Get base names without extensions
    fasta_basename = os.path.basename(fasta_path).split('.')[0]
    output_prefix = os.path.join(output_dir, fasta_basename + "_emb")
    
    config = {
        "global": {
            "sequences_file": fasta_path,
            "prefix": output_prefix
        },
        "t5_embeddings": {
            "type": "embed",
            "protocol": "prottrans_t5_xl_u50",
            "half_precision_model": True,
            "half_precision": True
        },
        "annotations_from_t5": {
            "type": "extract",
            "protocol": "la_prott5",
            "depends_on": "t5_embeddings"
        }
    }
    
    config_path = os.path.join(output_dir, "test_embedding_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Embedding configuration created at {config_path}")
    return config_path, output_prefix

def test_embedding_generation(config_path):
    """Test if bio-embeddings can generate embeddings."""
    print("\nTesting embedding generation...")
    print("This step may take some time as it downloads the ProtT5 model if not already cached.")
    print("Command to run: bio_embeddings test_embedding_config.yml")
    print("\nNote: You should manually run this command in the VM to test embedding generation.")
    print("If successful, it will create embeddings in the specified output directory.")
    return True  # We can't actually run this here, but provide instructions

def create_inference_config(embedding_path, remapping_path, model_param_path, output_dir):
    """Create an inference configuration file for PLM_Sol."""
    print("Creating inference configuration...")
    
    config = {
        "output_files_name": "test_inference",
        "log_iterations": 100,
        "n_draws": 1000,
        "batch_size": 1,
        "checkpoints_list": [model_param_path],
        "embeddings": embedding_path,
        "remapping": remapping_path,
        "key_format": "fasta_descriptor"
    }
    
    config_path = os.path.join(output_dir, "test_inference_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Inference configuration created at {config_path}")
    return config_path

def test_model_prediction(inference_config_path):
    """Test if the model can load and make predictions."""
    print("\nTesting model prediction...")
    print(f"Command to run: python inference.py --config {inference_config_path}")
    print("\nNote: You should manually run this command in the VM to test model prediction.")
    print("If successful, it will create a CSV file with predictions.")
    return True  # We can't actually run this here, but provide instructions

def main():
    print("=" * 50)
    print("PLM_Sol Functionality Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install the required dependencies.")
        return False
    
    # Create a temporary directory for test files
    test_dir = os.path.join(os.getcwd(), "plmsol_test")
    os.makedirs(test_dir, exist_ok=True)
    print(f"Created test directory at {test_dir}")
    
    try:
        # Create test FASTA file
        fasta_path = create_test_fasta(test_dir)
        
        # Create embedding configuration
        embedding_config_path, output_prefix = create_embedding_config(fasta_path, test_dir)
        
        # Test embedding generation (instructions only)
        test_embedding_generation(embedding_config_path)
        
        # Expected paths after embedding generation
        expected_embedding_path = f"{output_prefix}/t5_embeddings/embeddings_file.h5"
        expected_remapping_path = f"{output_prefix}/remapped_sequences_file.fasta"
        model_param_path = "./model_param/model_param.t7"
        
        # Create inference configuration
        inference_config_path = create_inference_config(
            expected_embedding_path, 
            expected_remapping_path,
            model_param_path,
            test_dir
        )
        
        # Test model prediction (instructions only)
        test_model_prediction(inference_config_path)
        
        print("\n" + "=" * 50)
        print("Test Setup Complete")
        print("=" * 50)
        print("\nTo complete the test, run the following commands on the VM:")
        print(f"1. cd to the PLM_Sol directory")
        print(f"2. conda activate PLM_Sol (or create environment if needed)")
        print(f"3. bio_embeddings {os.path.join(test_dir, 'test_embedding_config.yml')}")
        print(f"4. python inference.py --config {inference_config_path}")
        print("\nIf successful, you should see a file named 'protTrans_prediction_result.csv' with predictions.")
        
    except Exception as e:
        print(f"\n❌ Error during test setup: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
