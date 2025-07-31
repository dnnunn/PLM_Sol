#!/usr/bin/env python3
import h5py
import numpy as np
import sys

def check_embeddings(h5_path):
    with h5py.File(h5_path, 'r') as f:
        print("Keys in the H5 file:", list(f.keys()))
        
        # Check the first few keys and their shapes
        print("\nFirst 5 keys and their shapes:")
        for i, key in enumerate(f.keys()):
            if i >= 5:
                break
            print(f"{key}: {f[key].shape}")
            
        # Print a sample embedding
        first_key = next(iter(f.keys()))
        print(f"\nSample embedding for {first_key}:")
        print(f"Shape: {f[first_key].shape}")
        print(f"First 5 values: {f[first_key][:5, 0]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_embeddings.py <path_to_embeddings.h5>")
        sys.exit(1)
    check_embeddings(sys.argv[1])
