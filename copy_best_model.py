#!/usr/bin/env python3
"""
Copy Best Model Checkpoint to Standard Location
-----------------------------------------------

Copies the best performing model checkpoint from the fine-tuning experiment
to the standard location (/home/david_nunn/PLM_Sol/saved_models/model-10.t7)
so that all existing integration scripts work without path changes.

Usage:
  python copy_best_model.py --experiment finetune_combined_high_1_5sigma_20250801_114226
  python copy_best_model.py --auto  # Auto-detect latest experiment

This ensures compatibility with:
- PeptideFrontEnd integration scripts
- PLM_Sol wrapper scripts
- Existing inference configurations
"""

import argparse
import shutil
import os
from pathlib import Path
import glob
import yaml

def find_latest_experiment(base_dir="/home/david_nunn/PLM_Sol"):
    """Find the most recent fine-tuning experiment directory"""
    outputs_dir = Path(base_dir) / "outputs"
    if not outputs_dir.exists():
        print(f"❌ Outputs directory not found: {outputs_dir}")
        return None
    
    # Look for experiment directories
    experiment_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith('finetune_')]
    
    if not experiment_dirs:
        print(f"❌ No experiment directories found in {outputs_dir}")
        return None
    
    # Sort by modification time (most recent first)
    experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest = experiment_dirs[0]
    
    print(f"📁 Latest experiment found: {latest.name}")
    return latest

def find_best_model_checkpoint(experiment_dir):
    """Find the best model checkpoint in the experiment directory"""
    experiment_path = Path(experiment_dir)
    
    # Look for model checkpoints in the models subdirectory first
    models_subdir = experiment_path / "models"
    
    model_patterns = [
        "model-*.t7",
        "model_*.t7", 
        "*.t7",
        "best_model.t7",
        "model.t7"
    ]
    
    model_files = []
    
    # First check the models subdirectory
    if models_subdir.exists():
        for pattern in model_patterns:
            model_files.extend(models_subdir.glob(pattern))
    
    # If no models found in subdirectory, check main experiment directory
    if not model_files:
        for pattern in model_patterns:
            model_files.extend(experiment_path.glob(pattern))
    
    if not model_files:
        print(f"❌ No model checkpoints found in {experiment_path}")
        return None
    
    # If multiple files, prefer the highest numbered model (best performance)
    if len(model_files) == 1:
        best_model = model_files[0]
    else:
        # Prefer files with "best" in name first
        best_candidates = [f for f in model_files if 'best' in f.name.lower()]
        if best_candidates:
            best_model = best_candidates[0]
        else:
            # Sort by model number (highest first) - model-43.t7 should be preferred
            import re
            def extract_model_number(filename):
                match = re.search(r'model-?(\d+)', filename.name)
                return int(match.group(1)) if match else 0
            
            model_files.sort(key=extract_model_number, reverse=True)
            best_model = model_files[0]
    
    print(f"🏆 Best model checkpoint: {best_model.name}")
    print(f"📊 File size: {best_model.stat().st_size / (1024*1024):.1f} MB")
    print(f"📅 Modified: {best_model.stat().st_mtime}")
    
    return best_model

def copy_model_to_standard_location(source_model, base_dir="/home/david_nunn/PLM_Sol"):
    """Copy the model to the standard saved_models location"""
    saved_models_dir = Path(base_dir) / "saved_models"
    saved_models_dir.mkdir(exist_ok=True)
    
    target_path = saved_models_dir / "model-10.t7"
    
    # Backup existing model if it exists
    if target_path.exists():
        backup_path = saved_models_dir / "model-10.t7.backup"
        print(f"📦 Backing up existing model to: {backup_path}")
        shutil.copy2(target_path, backup_path)
    
    # Copy the new model
    print(f"📋 Copying {source_model} to {target_path}")
    shutil.copy2(source_model, target_path)
    
    # Verify the copy
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"✅ Model successfully copied to: {target_path}")
        print(f"📊 Final size: {target_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    else:
        print(f"❌ Failed to copy model to: {target_path}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Copy best model checkpoint to standard location")
    parser.add_argument('--experiment', help='Specific experiment directory name')
    parser.add_argument('--auto', action='store_true', help='Auto-detect latest experiment')
    parser.add_argument('--base_dir', default='/home/david_nunn/PLM_Sol', help='PLM_Sol base directory')
    
    args = parser.parse_args()
    
    print("🔧 PLM_Sol Model Checkpoint Copy Utility")
    print("=" * 50)
    
    # Determine experiment directory
    if args.experiment:
        experiment_dir = Path(args.base_dir) / "outputs" / args.experiment
        if not experiment_dir.exists():
            print(f"❌ Experiment directory not found: {experiment_dir}")
            return False
        print(f"🎯 Using specified experiment: {args.experiment}")
    elif args.auto:
        experiment_dir = find_latest_experiment(args.base_dir)
        if not experiment_dir:
            return False
    else:
        print("❌ Please specify --experiment <name> or --auto")
        return False
    
    # Find best model checkpoint
    best_model = find_best_model_checkpoint(experiment_dir)
    if not best_model:
        return False
    
    # Copy to standard location
    success = copy_model_to_standard_location(best_model, args.base_dir)
    
    if success:
        print(f"\n🎉 Model checkpoint ready for integration!")
        print(f"📁 Standard location: {args.base_dir}/saved_models/model-10.t7")
        print(f"🚀 Ready to test PeptideFrontEnd integration")
        return True
    else:
        print(f"\n❌ Failed to copy model checkpoint")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
