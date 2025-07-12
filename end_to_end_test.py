#!/usr/bin/env python
import os
import sys
import torch
import yaml
from pathlib import Path

# Create test directory
test_dir = Path("./plmsol_test")
test_dir.mkdir(exist_ok=True)
print(f"Created test directory: {test_dir}")

# Create a simple test FASTA file
fasta_path = test_dir / "test.fasta"
with open(fasta_path, "w") as f:
    f.write(">test_protein\n")
    f.write("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK\n")
print(f"Created test FASTA file: {fasta_path}")

# Create embedding config
embed_config_path = test_dir / "embedding_config.yml"
embed_config = {
    "global": {
        "sequences_file": str(fasta_path),
        "prefix": str(test_dir / "embeddings")
    },
    "embeddings": {
        "type": "embed",
        "protocol": "word2vec",  # Using word2vec since ProtTransT5XLU50 is not available
        "options": {}
    }
}

with open(embed_config_path, "w") as f:
    yaml.dump(embed_config, f)
print(f"Created embedding config: {embed_config_path}")

print("\nTo complete the test, run:")
print(f"1. bio_embeddings {embed_config_path}")
print("2. After embeddings are generated, run inference.py with the generated embeddings")
