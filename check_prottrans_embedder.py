#!/usr/bin/env python
import sys
import importlib

print(f"Python version: {sys.version}")

try:
    import bio_embeddings
    print(f"bio_embeddings version: {bio_embeddings.__version__}")
    
    # Try to import specific embedders
    print("\nChecking for ProtTransT5XLU50Embedder:")
    try:
        from bio_embeddings.embed import ProtTransT5XLU50Embedder
        print("✓ Successfully imported ProtTransT5XLU50Embedder")
        
        # Try to instantiate the embedder
        try:
            embedder = ProtTransT5XLU50Embedder()
            print("✓ Successfully instantiated ProtTransT5XLU50Embedder")
        except Exception as e:
            print(f"✗ Error instantiating ProtTransT5XLU50Embedder: {e}")
    except ImportError as e:
        print(f"✗ Error importing ProtTransT5XLU50Embedder: {e}")
    
    # Check for alternative embedders
    print("\nChecking for other available embedders:")
    try:
        from bio_embeddings.embed import available_embedders
        print("Available embedders:")
        for embedder_name in available_embedders:
            print(f"- {embedder_name}")
    except ImportError as e:
        print(f"Error importing available_embedders: {e}")
        
        # Try to list embedders manually
        print("\nAttempting to list embedders manually:")
        from bio_embeddings import embed
        for attr in dir(embed):
            if "Embedder" in attr and not attr.startswith("_"):
                print(f"- {attr}")
    
except ImportError as e:
    print(f"Error importing bio_embeddings: {e}")

print("\nDone checking embedders.")
