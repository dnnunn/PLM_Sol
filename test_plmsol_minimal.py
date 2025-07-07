#!/usr/bin/env python
"""
Minimal test script for PLM_Sol functionality without instantiating large models.
"""
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path

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

def check_model_files():
    """Check if model checkpoint files exist."""
    print("\nChecking model files...")
    
    # Check for model_param.t7
    model_param_path = os.path.join(os.getcwd(), "model_param", "model_param.t7")
    if os.path.exists(model_param_path):
        print(f"✓ Found model checkpoint at {model_param_path}")
        
        # Check file size
        size_mb = os.path.getsize(model_param_path) / (1024 * 1024)
        print(f"  - File size: {size_mb:.2f} MB")
        
        # Try to load the model parameters
        try:
            params = torch.load(model_param_path, map_location=torch.device('cpu'))
            print(f"✓ Successfully loaded model parameters")
            print(f"  - Parameter type: {type(params)}")
            if hasattr(params, 'keys'):
                print(f"  - Contains keys: {', '.join(list(params.keys())[:5])}...")
            return True
        except Exception as e:
            print(f"✗ Error loading model parameters: {e}")
            return False
    else:
        print(f"✗ Model checkpoint not found at {model_param_path}")
        return False

def check_embedder():
    """Check if the required embedder is available."""
    print("\nChecking for ProtTransT5XLU50Embedder...")
    try:
        from bio_embeddings.embed import ProtTransT5XLU50Embedder
        print("✓ ProtTransT5XLU50Embedder class is available")
        return True
    except ImportError as e:
        print(f"✗ Error importing ProtTransT5XLU50Embedder: {e}")
        return False

def create_test_fasta(output_dir):
    """Create a test FASTA file with a few protein sequences."""
    print("\nCreating test FASTA file...")
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
    print("\nCreating embedding configuration...")
    
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
        }
    }
    
    config_path = os.path.join(output_dir, "test_embedding_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Embedding configuration created at {config_path}")
    return config_path, output_prefix

def create_inference_config(embedding_path, remapping_path, model_param_path, output_dir):
    """Create an inference configuration file for PLM_Sol."""
    print("\nCreating inference configuration...")
    
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

def main():
    print("=" * 50)
    print("PLM_Sol Minimal Functionality Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ Import test failed. Please install the required dependencies.")
        return False
    
    # Check model files
    model_files_ok = check_model_files()
    if not model_files_ok:
        print("\n❌ Model files check failed. Please ensure the model checkpoint is available.")
        return False
    
    # Check embedder
    embedder_ok = check_embedder()
    if not embedder_ok:
        print("\n❌ Embedder check failed. Please ensure bio_embeddings is properly installed.")
        return False
    
    # Create a directory for test files
    test_dir = os.path.join(os.getcwd(), "plmsol_test")
    os.makedirs(test_dir, exist_ok=True)
    print(f"\nCreated test directory at {test_dir}")
    
    try:
        # Create test FASTA file
        fasta_path = create_test_fasta(test_dir)
        
        # Create embedding configuration
        embedding_config_path, output_prefix = create_embedding_config(fasta_path, test_dir)
        
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
        
        print("\n" + "=" * 50)
        print("Test Setup Complete")
        print("=" * 50)
        print("\nTo complete the test, run the following commands:")
        print(f"1. Generate embeddings (this will download the model if not cached):")
        print(f"   bio_embeddings {embedding_config_path}")
        print(f"\n2. Run inference:")
        print(f"   python inference.py --config {inference_config_path}")
        print("\nIf successful, you should see a file named 'test_inference_prediction_result.csv' with predictions.")
        
    except Exception as e:
        print(f"\n❌ Error during test setup: {e}")
    
    return True

if __name__ == "__main__":
    main()
