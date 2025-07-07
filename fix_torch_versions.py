#!/usr/bin/env python
import sys
import subprocess
import os

def run_command(cmd):
    print(f"Running: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(process.stdout)
    if process.returncode != 0:
        print(f"Error: {process.stderr}")
        return False
    return True

def main():
    print("Fixing PyTorch and torchvision compatibility issues")
    
    # First uninstall current versions
    print("\n1. Uninstalling current PyTorch packages...")
    run_command("pip uninstall -y torch torchvision torchaudio pytorch-lightning pytorch-pretrained-bert pytorch-transformers")
    
    # Install compatible versions
    print("\n2. Installing compatible PyTorch with CUDA support...")
    # Using PyTorch 1.9.1 with CUDA support and compatible torchvision
    run_command("pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html")
    
    # Install other PyTorch-related packages
    print("\n3. Installing other PyTorch-related packages...")
    run_command("pip install pytorch-lightning==1.8.6 pytorch-pretrained-bert==0.6.2 pytorch-transformers==1.1.0 torchmetrics==1.2.1")
    
    # Verify installation
    print("\n4. Verifying installation...")
    run_command("python -c \"import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')\"")
    run_command("python -c \"import torchvision; print('torchvision version:', torchvision.__version__)\"")
    
    # Reinstall bio_embeddings
    print("\n5. Reinstalling bio_embeddings...")
    run_command("pip install --upgrade bio-embeddings==0.2.2")
    run_command("pip install --upgrade bio-embeddings[all]==0.2.2")
    
    print("\nSetup complete. Please try importing bio_embeddings again.")

if __name__ == "__main__":
    main()
