#!/usr/bin/env python
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(process.stdout)
    if process.returncode != 0:
        print(f"Error: {process.stderr}")
        return False
    return True

def main():
    print("Installing missing dependencies...")
    
    # Install all required dependencies
    dependencies = [
        "PyYAML",
        "pyaml",
        "tensorboard",
        "deepblast",  # This might be needed for bio_embeddings align module
    ]
    
    for dep in dependencies:
        print(f"\nInstalling {dep}...")
        run_command(f"pip install {dep}")
    
    # Check if they worked
    try:
        import yaml
        print(f"✓ Successfully imported yaml module")
    except ImportError as e:
        print(f"✗ Failed to import yaml: {e}")
        
    try:
        import pyaml
        print(f"✓ Successfully imported pyaml module")
    except ImportError as e:
        print(f"✗ Failed to import pyaml: {e}")
        
    try:
        import tensorboard
        print(f"✓ Successfully imported tensorboard module")
    except ImportError as e:
        print(f"✗ Failed to import tensorboard: {e}")
    
    print("\nDependencies installation complete.")

if __name__ == "__main__":
    main()
