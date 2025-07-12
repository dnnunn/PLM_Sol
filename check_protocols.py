#!/usr/bin/env python
from bio_embeddings.utilities import get_available_protocols

print("Available embedding protocols:")
protocols = get_available_protocols()
for protocol in protocols:
    print(f"- {protocol}")
