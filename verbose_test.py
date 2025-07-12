#!/usr/bin/env python
# Verbose test script to verify PLM_Sol functionality

import os
import sys
import traceback
import torch
import numpy as np
import yaml

def print_section(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def test_environment():
    print_section("Environment Information")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

def test_imports():
    print_section("Testing Imports")
    
    # Basic imports
    modules = {
        "numpy": "np",
        "pandas": "pd",
        "torch": "torch",
        "yaml": "yaml",
        "h5py": "h5py",
        "Bio": "Bio"
    }
    
    for module_name, alias in modules.items():
        try:
            module = __import__(module_name)
            if hasattr(module, "__version__"):
                print(f"✅ {module_name} {module.__version__}")
            else:
                print(f"✅ {module_name} (version unknown)")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
    
    # PLM_Sol specific imports
    print("\nTesting PLM_Sol specific imports:")
    try:
        from models.biLSTM_TextCNN import biLSTM_TextCNN
        print("✅ models.biLSTM_TextCNN")
    except ImportError as e:
        print(f"❌ models.biLSTM_TextCNN: {e}")
    
    try:
        from datasets.embeddings_dataset import Embeddings_predict_Dataset
        print("✅ datasets.embeddings_dataset")
    except ImportError as e:
        print(f"❌ datasets.embeddings_dataset: {e}")
    
    try:
        from bio_embeddings.embed import ProtTransT5XLU50Embedder
        print("✅ bio_embeddings.embed.ProtTransT5XLU50Embedder")
    except ImportError as e:
        print(f"❌ bio_embeddings.embed.ProtTransT5XLU50Embedder: {e}")

def check_model_files():
    print_section("Checking Model Files")
    
    # Check model_param directory
    model_param_dir = "model_param"
    if os.path.exists(model_param_dir):
        print(f"✅ {model_param_dir} directory exists")
        model_files = os.listdir(model_param_dir)
        print(f"Files in {model_param_dir}: {model_files}")
        
        model_param_file = os.path.join(model_param_dir, "model_param.t7")
        if os.path.exists(model_param_file):
            print(f"✅ {model_param_file} exists ({os.path.getsize(model_param_file) / (1024*1024):.2f} MB)")
        else:
            print(f"❌ {model_param_file} not found")
    else:
        print(f"❌ {model_param_dir} directory not found")
    
    # Check checkpoints directory
    checkpoints_dir = "checkpoints"
    if os.path.exists(checkpoints_dir):
        print(f"✅ {checkpoints_dir} directory exists")
        checkpoint_files = os.listdir(checkpoints_dir)
        print(f"Files in {checkpoints_dir}: {checkpoint_files}")
        
        for file in checkpoint_files:
            if file.endswith('.pth'):
                file_path = os.path.join(checkpoints_dir, file)
                print(f"✅ {file_path} exists ({os.path.getsize(file_path) / (1024*1024):.2f} MB)")
    else:
        print(f"❌ {checkpoints_dir} directory not found")

def test_model_loading():
    print_section("Testing Model Loading")
    
    try:
        from models.biLSTM_TextCNN import biLSTM_TextCNN
        
        # Create a simple model instance
        print("Creating model instance...")
        model = biLSTM_TextCNN(
            embedding_dim=1024,
            hidden_dim=512,
            num_layers=2,
            dropout=0.5,
            kernel_num=100,
            kernel_sizes=[3, 4, 5],
            class_num=2
        )
        print("✅ Model instance created successfully")
        
        # Try loading checkpoint
        checkpoint_path = os.path.join("checkpoints", "model_param.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                print("✅ Checkpoint loaded successfully")
                print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dictionary'}")
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                traceback.print_exc()
        else:
            print(f"❌ Checkpoint file {checkpoint_path} not found")
    
    except Exception as e:
        print(f"❌ Error in model loading test: {e}")
        traceback.print_exc()

def main():
    print_section("PLM_Sol Verbose Test")
    
    try:
        test_environment()
        test_imports()
        check_model_files()
        test_model_loading()
        
        print_section("Test Summary")
        print("All tests completed. Check the output above for any errors.")
        print("If all tests passed, PLM_Sol should be ready for use.")
        print("\nNext steps:")
        print("1. Create a FASTA file with protein sequences")
        print("2. Generate embeddings using bio_embeddings")
        print("3. Run inference using the model")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
