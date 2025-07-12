#!/usr/bin/env python
import sys

try:
    from bio_embeddings.embed import ProtTransT5XLU50Embedder
    print("✓ ProtTransT5XLU50Embedder class exists")
    
    # List available embedders without instantiating them
    from bio_embeddings.embed import available_embedders
    print("\nAvailable embedders:")
    for name in available_embedders:
        print(f"- {name}")
        
except ImportError as e:
    print(f"✗ Error importing ProtTransT5XLU50Embedder: {e}")

print("\nDone checking embedders.")
