import traceback
import sys

try:
    print("Starting test...")
    
    import os
    import time
    import yaml
    
    print("Basic imports successful")
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Date and time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for model files
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    print(f"Looking for checkpoint directory: {checkpoint_dir}")
    
    if os.path.exists(checkpoint_dir):
        print(f"✅ Checkpoint directory found")
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        print(f"Found {len(checkpoint_files)} checkpoint files")
    else:
        print(f"❌ Checkpoint directory not found")
    
    print("Test completed successfully")
    
except Exception as e:
    print("Error occurred:")
    traceback.print_exc()
