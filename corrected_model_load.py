#!/usr/bin/env python3
"""
Corrected test script to directly load the PLM_Sol model
"""
import os
import sys
import torch
import traceback
from models import *
from models.biLSTM_TextCNN import biLSTM_TextCNN

def test_model_loading():
    """Test loading the PLM_Sol model directly"""
    try:
        print("Testing model loading for PLM_Sol")
        print(f"Current working directory: {os.getcwd()}")
        
        # Try to find the model file
        model_path = os.path.join("model_param", "model_param.t7")
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            return False
        
        print(f"Found model file at {model_path}")
        print(f"Model file size: {os.path.getsize(model_path)} bytes")
        
        # Check the train arguments to get parameters
        train_args_path = os.path.join("model_param", "train_arguments.yml")
        if os.path.exists(train_args_path):
            import yaml
            print(f"Loading train arguments from {train_args_path}")
            with open(train_args_path, 'r') as f:
                train_args = yaml.safe_load(f)
                print(f"Train arguments: {train_args}")
        
        # Print the model class signature to see what parameters it accepts
        import inspect
        print("Model signature:")
        print(inspect.signature(biLSTM_TextCNN.__init__))
        
        # Try different parameter combinations
        try:
            print("Trying to initialize model with common parameters...")
            # Common pattern is embeddings_dim, num_filters, kernel_sizes, dropout
            model = biLSTM_TextCNN(embeddings_dim=1024, dropout=0.5)
            print("Model initialized with: embeddings_dim, dropout")
        except Exception as e:
            print(f"Failed with: {e}")
            try:
                model = biLSTM_TextCNN(1024)  # Just try with embeddings dimension
                print("Model initialized with just embeddings_dim")
            except Exception as e:
                print(f"Failed with: {e}")
                try:
                    # Try with no parameters (default constructor)
                    model = biLSTM_TextCNN()
                    print("Model initialized with no parameters")
                except Exception as e:
                    print(f"Failed with: {e}")
                    print("Could not initialize model with any parameter combination")
                    return False
        
        print("Successfully created model instance")
        
        print(f"Loading checkpoint from {model_path}...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint loaded, type: {type(checkpoint)}")
        
        # Print checkpoint keys if it's a dict
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {checkpoint.keys()}")
            if 'model_state_dict' in checkpoint:
                print("Loading model from state dict")
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                print("Loading from state_dict key")
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("Loading direct model state")
                model.load_state_dict(checkpoint)
        else:
            print("Loading direct model state")
            model.load_state_dict(checkpoint)
            
        print("Model loaded successfully!")
        model.eval()
        print("Model is in eval mode")
        return True
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print(f"Model loading {'succeeded' if success else 'failed'}")
    sys.exit(0 if success else 1)
