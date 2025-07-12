#!/usr/bin/env python
import os
import torch
import traceback
from models.biLSTM_TextCNN import biLSTM_TextCNN

try:
    print("Creating model instance with correct parameters...")
    model = biLSTM_TextCNN(
        embeddings_dim=1024,  # Correct parameter name is embeddings_dim, not embedding_dim
        output_dim=1,
        dropout=0.25,
        kernel_size=9,
        conv_dropout=0.25
    )
    print("✅ Model instance created successfully")
    
    # Try loading checkpoint
    checkpoint_path = os.path.join("checkpoints", "model_param.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            print("✅ Checkpoint loaded successfully")
            print(f"Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
                
                # Try to load state dict into model
                try:
                    model.load_state_dict(checkpoint)
                    print("✅ Model state loaded successfully")
                except Exception as e:
                    print(f"❌ Failed to load state dict: {e}")
            else:
                print("Checkpoint is not a dictionary, might be a direct state dict")
                try:
                    model.load_state_dict(checkpoint)
                    print("✅ Model state loaded successfully")
                except Exception as e:
                    print(f"❌ Failed to load state dict: {e}")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            traceback.print_exc()
    else:
        print(f"❌ Checkpoint file {checkpoint_path} not found")
        
except Exception as e:
    print(f"❌ Error in model loading test: {e}")
    traceback.print_exc()
