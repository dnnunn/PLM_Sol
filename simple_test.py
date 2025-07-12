print("Hello, world!")

import sys
print(f"Python version: {sys.version}")

try:
    import numpy
    print(f"✅ numpy {numpy.__version__}")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import torch
    print(f"✅ torch {torch.__version__}")
except ImportError as e:
    print(f"❌ torch import failed: {e}")

try:
    import bio_embeddings
    print(f"✅ bio_embeddings imported")
except ImportError as e:
    print(f"❌ bio_embeddings import failed: {e}")

try:
    import matplotlib
    print(f"✅ matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ matplotlib import failed: {e}")
