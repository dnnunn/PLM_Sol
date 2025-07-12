#!/usr/bin/env python
import sys
import pkgutil
import importlib

print(f"Python version: {sys.version}")

try:
    import bio_embeddings
    
    # Try to get version
    try:
        print(f"bio_embeddings version: {bio_embeddings.__version__}")
    except AttributeError:
        print("bio_embeddings version not available")
    
    # List all submodules
    print("\nAvailable bio_embeddings submodules:")
    for _, name, ispkg in pkgutil.iter_modules(bio_embeddings.__path__, bio_embeddings.__name__ + '.'):
        print(f"- {name}")
    
    # Check embed module specifically
    print("\nChecking bio_embeddings.embed module:")
    try:
        from bio_embeddings import embed
        print("Successfully imported bio_embeddings.embed")
        
        # List all attributes in embed module
        print("\nAttributes in bio_embeddings.embed:")
        for attr in dir(embed):
            if not attr.startswith('_'):
                print(f"- {attr}")
                
        # Check for specific embedder classes
        embedder_names = [
            "ProtTransT5XLU50Embedder",
            "ProtTransT5Embedder",
            "ProtT5Embedder",
            "T5Embedder",
            "ESM1bEmbedder",
            "ESM2Embedder",
            "ProtTransBertBFDEmbedder"
        ]
        
        print("\nChecking for specific embedder classes:")
        for name in embedder_names:
            try:
                cls = getattr(embed, name, None)
                if cls:
                    print(f"✅ Found {name}")
                else:
                    print(f"❌ {name} not found")
            except Exception as e:
                print(f"❌ Error checking {name}: {e}")
                
    except ImportError as e:
        print(f"Error importing bio_embeddings.embed: {e}")
    
except ImportError as e:
    print(f"Error importing bio_embeddings: {e}")
