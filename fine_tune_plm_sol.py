#!/usr/bin/env python3
"""
Fine-tune PLM_Sol on specialized datasets for PeptideFrontEnd GA workflows.
This script handles the complete fine-tuning pipeline:
1. Generate embeddings for our fine-tuning datasets
2. Create remapped FASTA files
3. Fine-tune the PLM_Sol model
4. Evaluate performance improvements
"""

import os
import sys
import subprocess
import yaml
import argparse
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime

class PLMSolFineTuner:
    def __init__(self, base_dir="/home/david_nunn/PLM_Sol"):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "fine_tuning_datasets"
        self.embeddings_dir = self.base_dir / "fine_tuning_embeddings"
        self.outputs_dir = self.base_dir / "fine_tuning_outputs"
        
        # Create necessary directories
        self.embeddings_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        
        print(f"🔧 PLM_Sol Fine-Tuner initialized")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Datasets: {self.datasets_dir}")
        print(f"🧬 Embeddings: {self.embeddings_dir}")
        print(f"📈 Outputs: {self.outputs_dir}")

    def generate_embeddings(self, dataset_name, force_regenerate=False):
        """Generate T5 embeddings for a specific dataset using bio_embeddings"""
        print(f"\n🧬 Generating embeddings for {dataset_name}...")
        
        dataset_dir = self.datasets_dir / dataset_name
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} not found in {self.datasets_dir}")
        
        embedding_output_dir = self.embeddings_dir / dataset_name
        embedding_output_dir.mkdir(exist_ok=True)
        
        # Process train, val, test splits
        for split in ['train', 'val', 'test']:
            fasta_file = dataset_dir / f"{split}.fasta"
            if not fasta_file.exists():
                print(f"⚠️  Warning: {fasta_file} not found, skipping {split}")
                continue
            
            split_output_dir = embedding_output_dir / f"{split}_emb"
            embeddings_file = split_output_dir / "t5_embeddings" / "embeddings_file.h5"
            remapped_file = split_output_dir / "remapped_sequences_file.fasta"
            
            # Skip if embeddings already exist and not forcing regeneration
            if embeddings_file.exists() and remapped_file.exists() and not force_regenerate:
                print(f"✅ Embeddings for {dataset_name}/{split} already exist, skipping")
                continue
            
            print(f"🔄 Processing {dataset_name}/{split}...")
            
            # Create bio_embeddings config for this split
            config_file = self.create_embedding_config(fasta_file, split_output_dir)
            
            # Run bio_embeddings
            try:
                cmd = [
                    "conda", "run", "-n", "PLM_Sol",
                    "bio_embeddings", str(config_file), "--overwrite"
                ]
                
                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
                
                if result.returncode != 0:
                    print(f"❌ Error generating embeddings for {dataset_name}/{split}:")
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")
                    continue
                
                print(f"✅ Successfully generated embeddings for {dataset_name}/{split}")
                
            except Exception as e:
                print(f"❌ Exception generating embeddings for {dataset_name}/{split}: {e}")
                continue
        
        print(f"🎉 Embedding generation completed for {dataset_name}")

    def create_embedding_config(self, fasta_file, output_dir):
        """Create bio_embeddings config file for T5 embeddings using PLM_Sol format"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            'global': {
                'sequences_file': str(fasta_file),
                'prefix': str(output_dir)
            },
            't5_embeddings': {
                'type': 'embed',
                'key_format': 'fasta_descriptor', 
                'protocol': 'prottrans_t5_xl_u50',
                'half_precision_model': True,
                'half_precision': True
            }
        }
        
        config_file = output_dir / "embedding_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_file

    def create_fine_tuning_config(self, dataset_name, experiment_name=None):
        """Create training config for fine-tuning"""
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"finetune_{dataset_name}_{timestamp}"
        
        embedding_base = self.embeddings_dir / dataset_name
        
        config = {
            'experiment_name': f'PLM_Sol_FineTuning_{dataset_name}',
            'num_epochs': 50,  # Reduced for fine-tuning
            'batch_size': 64,  # Smaller batch size for fine-tuning
            'log_iterations': 10,
            'patience': 10,  # Early stopping patience
            'optimizer_parameters': {
                'lr': 1.0e-4  # Lower learning rate for fine-tuning
            },
            'target': 'sol',
            'unknown_solubility': False,
            'key_format': 'fasta_descriptor', # Use correct parser for '>id description soluble-label' format
            'exp_name': experiment_name,
            
            # Paths to our fine-tuning embeddings
            'train_embeddings': str(embedding_base / 'train_emb' / 't5_embeddings' / 'embeddings_file.h5'),
            'val_embeddings': str(embedding_base / 'val_emb' / 't5_embeddings' / 'embeddings_file.h5'),
            'test_embeddings': str(embedding_base / 'test_emb' / 't5_embeddings' / 'embeddings_file.h5'),
            'train_remapping': str(embedding_base / 'train_emb' / 'remapped_sequences_file.fasta'),
            'val_remapping': str(embedding_base / 'val_emb' / 'remapped_sequences_file.fasta'),
            'test_remapping': str(embedding_base / 'test_emb' / 'remapped_sequences_file.fasta'),
            
            # Model parameters (using pre-trained checkpoint)
            'model_type': 'FFN',
            'model_parameters': {
                'output_dim': 1,
                'hidden_dim': 32,
                'n_hidden_layers': 0,
                'dropout': 0.25
            },
            
            # Use pre-trained checkpoint for fine-tuning
            'checkpoint': './checkpoints/FFN_checkpoint.pt',
            'eval_on_test': True
        }
        
        config_file = self.outputs_dir / f"{experiment_name}_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_file, experiment_name

    def run_fine_tuning(self, dataset_name, experiment_name=None):
        """Run the fine-tuning process"""
        print(f"\n🚀 Starting fine-tuning for {dataset_name}...")
        
        # Create training config
        config_file, exp_name = self.create_fine_tuning_config(dataset_name, experiment_name)
        print(f"📋 Config created: {config_file}")
        
        # Run training
        try:
            cmd = [
                "conda", "run", "-n", "PLM_Sol",
                "python", "train.py", 
                "--config", str(config_file),
                "--key_format", "fasta_descriptor" # Explicitly set key_format to override train.py default
            ]
            
            print(f"🏃 Running fine-tuning: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.base_dir, text=True)
            
            if result.returncode == 0:
                print(f"✅ Fine-tuning completed successfully for {dataset_name}")
                return exp_name
            else:
                print(f"❌ Fine-tuning failed for {dataset_name}")
                return None
                
        except Exception as e:
            print(f"❌ Exception during fine-tuning: {e}")
            return None

    def evaluate_fine_tuned_model(self, experiment_name, dataset_name):
        """Evaluate the fine-tuned model performance"""
        print(f"\n📊 Evaluating fine-tuned model: {experiment_name}")
        
        # Check if model outputs exist
        model_dir = self.base_dir / "outputs" / experiment_name
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return None
        
        # Look for evaluation results
        eval_files = list(model_dir.glob("*evaluation*.txt"))
        log_files = list(model_dir.glob("run.log"))
        
        results = {}
        
        # Parse log file for training metrics
        if log_files:
            log_file = log_files[0]
            print(f"📄 Parsing training log: {log_file}")
            results['training_log'] = str(log_file)
            
            # Extract final metrics from log
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-50:]):  # Check last 50 lines
                        if 'test acc:' in line:
                            results['final_test_accuracy'] = line.strip()
                            break
            except Exception as e:
                print(f"⚠️  Could not parse log file: {e}")
        
        # Parse evaluation files
        for eval_file in eval_files:
            print(f"📊 Found evaluation file: {eval_file}")
            results[eval_file.name] = str(eval_file)
        
        return results

    def run_complete_fine_tuning_pipeline(self, dataset_names=None, force_regenerate_embeddings=False):
        """Run the complete fine-tuning pipeline for specified datasets"""
        if dataset_names is None:
            # Default to all available datasets
            dataset_names = [d.name for d in self.datasets_dir.iterdir() if d.is_dir()]
        
        print(f"🎯 Starting complete fine-tuning pipeline for datasets: {dataset_names}")
        
        results = {}
        
        for dataset_name in dataset_names:
            print(f"\n{'='*60}")
            print(f"🎯 Processing dataset: {dataset_name}")
            print(f"{'='*60}")
            
            try:
                # Step 1: Generate embeddings
                self.generate_embeddings(dataset_name, force_regenerate_embeddings)
                
                # Step 2: Run fine-tuning
                experiment_name = self.run_fine_tuning(dataset_name)
                
                if experiment_name:
                    # Step 3: Evaluate results
                    eval_results = self.evaluate_fine_tuned_model(experiment_name, dataset_name)
                    
                    results[dataset_name] = {
                        'experiment_name': experiment_name,
                        'status': 'completed',
                        'evaluation': eval_results
                    }
                else:
                    results[dataset_name] = {
                        'status': 'failed',
                        'error': 'Fine-tuning failed'
                    }
                    
            except Exception as e:
                print(f"❌ Error processing {dataset_name}: {e}")
                results[dataset_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Save summary results
        self.save_pipeline_results(results)
        
        return results

    def save_pipeline_results(self, results):
        """Save pipeline results summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.outputs_dir / f"fine_tuning_summary_{timestamp}.yml"
        
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        print(f"\n📋 Pipeline results saved to: {results_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("🎉 FINE-TUNING PIPELINE SUMMARY")
        print(f"{'='*60}")
        
        for dataset_name, result in results.items():
            status = result.get('status', 'unknown')
            print(f"📊 {dataset_name}: {status.upper()}")
            
            if status == 'completed':
                exp_name = result.get('experiment_name', 'N/A')
                print(f"   └── Experiment: {exp_name}")
                
                eval_info = result.get('evaluation', {})
                if 'final_test_accuracy' in eval_info:
                    print(f"   └── Final accuracy: {eval_info['final_test_accuracy']}")
            
            elif status in ['failed', 'error']:
                error = result.get('error', 'Unknown error')
                print(f"   └── Error: {error}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PLM_Sol on specialized datasets")
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='List of dataset names to fine-tune on')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force regeneration of embeddings even if they exist')
    return parser.parse_args()

if __name__ == "__main__":
    args = main()
    # Initialize fine-tuner
    fine_tuner = PLMSolFineTuner()
    
    # Run complete pipeline with force regenerate flag
    results = fine_tuner.run_complete_fine_tuning_pipeline(
        args.datasets, 
        args.force_regenerate
    )
    
    print(f"\n🎉 Fine-tuning pipeline completed!")
    print(f"📊 Processed {len(results)} datasets")
    
    # Count successes
    completed = sum(1 for r in results.values() if r.get('status') == 'completed')
    print(f"✅ Successful: {completed}/{len(results)}")
