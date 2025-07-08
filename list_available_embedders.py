#!/usr/bin/env python
"""
Script to list all available embedders in bio_embeddings and their memory requirements.
"""

import os
import sys
import importlib
import inspect
import psutil
import gc
import time
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

def main():
    print("Checking available embedders in bio_embeddings...")
    
    # Try to import bio_embeddings
    try:
        import bio_embeddings
        print(f"bio_embeddings version: {bio_embeddings.__version__ if hasattr(bio_embeddings, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"Error importing bio_embeddings: {e}")
        return
    
    # Try to import the embed module
    try:
        from bio_embeddings import embed
        print("Successfully imported bio_embeddings.embed")
    except ImportError as e:
        print(f"Error importing bio_embeddings.embed: {e}")
        return
    
    # Get all potential embedder classes
    embedder_classes = []
    for name in dir(embed):
        if name.endswith('Embedder') and name != 'EmbedderInterface':
            embedder_classes.append(name)
    
    print(f"\nFound {len(embedder_classes)} potential embedder classes:")
    for i, name in enumerate(sorted(embedder_classes)):
        print(f"{i+1}. {name}")
    
    # Create a test sequence
    test_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    
    print("\nTesting memory requirements for each embedder...")
    print("=" * 80)
    print(f"{'Embedder Name':<40} | {'Memory Before (MB)':<20} | {'Memory After (MB)':<20} | {'Embedding Shape':<20}")
    print("=" * 80)
    
    # Test each embedder
    for name in sorted(embedder_classes):
        print(f"Testing {name}...", end="", flush=True)
        
        try:
            # Force garbage collection
            gc.collect()
            
            # Get memory usage before loading
            memory_before = get_memory_usage()
            
            # Try to import the embedder class
            embedder_class = getattr(embed, name)
            
            # Check if it's a class and can be instantiated
            if not inspect.isclass(embedder_class):
                print(f"\r{name:<40} | {'N/A':<20} | {'Not a class':<20} | {'N/A':<20}")
                continue
            
            # Try to instantiate with reduced parameters if possible
            try:
                # Try with half precision first
                embedder = embedder_class(half_precision_model=True, half_precision=True)
            except TypeError:
                try:
                    # If half precision not supported, try default constructor
                    embedder = embedder_class()
                except Exception as e:
                    print(f"\r{name:<40} | {memory_before:<20.2f} | {'Error: ' + str(e)[:30]:<20} | {'N/A':<20}")
                    continue
            
            # Get memory after loading
            memory_after = get_memory_usage()
            
            # Try to embed the test sequence
            try:
                start_time = time.time()
                embedding = embedder.embed(test_sequence)
                end_time = time.time()
                
                # Print results
                print(f"\r{name:<40} | {memory_before:<20.2f} | {memory_after:<20.2f} | {str(embedding.shape):<20} | {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"\r{name:<40} | {memory_before:<20.2f} | {memory_after:<20.2f} | {'Error: ' + str(e)[:30]:<20}")
            
            # Force garbage collection
            del embedder
            gc.collect()
            
        except Exception as e:
            print(f"\r{name:<40} | {'Error: ' + str(e)[:50]}")
    
    print("\nRecommendation:")
    print("1. For best results but high memory usage: ProtTransT5XLU50Embedder")
    print("2. For medium memory usage: ESM1bEmbedder")
    print("3. For low memory usage: Word2VecEmbedder or FastTextEmbedder")
    print("\nTo use a specific embedder, modify your config file to use the appropriate protocol.")

if __name__ == "__main__":
    main()
