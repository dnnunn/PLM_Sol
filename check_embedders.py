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
    
    except ImportError as e:
        print(f"Error importing bio_embeddings.embed: {e}")
    
    # Check protocols
    print("\nChecking available protocols:")
    try:
        from bio_embeddings.utilities import get_available_protocols
        protocols = get_available_protocols()
        print(f"Available protocols: {protocols}")
    except Exception as e:
        print(f"Error getting protocols: {e}")
        
except ImportError as e:
    print(f"Error importing bio_embeddings: {e}")
