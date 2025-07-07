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
    
    # Install PyYAML instead of pyaml (more common and compatible)
    run_command("pip install PyYAML")
    
    # Check if it worked
    try:
        import yaml
        print(f"✓ Successfully imported yaml module")
    except ImportError as e:
        print(f"✗ Failed to import yaml: {e}")
    
    print("\nDependencies installation complete.")

if __name__ == "__main__":
    main()
