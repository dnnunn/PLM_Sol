#!/usr/bin/env python3
"""
Test script to directly load the PLM_Sol model
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
        
        # Try to load model
        print("Creating model instance...")
        model = biLSTM_TextCNN(embeddings_dim=1024, hidden_dim=512, dropout=0.5, max_len=1000)
        print("Model instance created")
        
        print(f"Loading checkpoint from {model_path}...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint loaded, type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading model from state dict")
            model.load_state_dict(checkpoint['model_state_dict'])
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
