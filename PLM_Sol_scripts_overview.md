# PLM_Sol Scripts Overview

This document provides a comprehensive analysis of the PLM_Sol directory, including its structure, Python scripts, their purposes, dependencies, and data flows.

## Table of Contents
1. [Directory Structure](#directory-structure)
2. [Core Scripts](#core-scripts)
3. [Model Training and Fine-tuning](#model-training-and-fine-tuning)
4. [Inference and Prediction](#inference-and-prediction)
5. [Data Processing Utilities](#data-processing-utilities)
6. [Model Architectures](#model-architectures)
7. [Dataset Handling](#dataset-handling)
8. [Dependencies](#dependencies)
9. [Data Flow](#data-flow)
10. [Integration with PeptideFrontEnd](#integration-with-peptidefrontend)
11. [Maintenance and Cleanup](#maintenance-and-cleanup)

## Directory Structure

```
PLM_Sol/
├── checkpoints/                 # Saved model checkpoints
├── configs/                     # Configuration files for different models and tasks
│   ├── SOL_biLSTM_TextCNN.yml   # Configuration for biLSTM+TextCNN model
│   └── baseline_eval_config.yml # Baseline evaluation configuration
├── datasets/                    # Dataset handling code
│   ├── __init__.py
│   ├── embeddings_dataset.py    # Dataset class for handling embeddings
│   └── transforms.py            # Data transformation utilities
├── docs/                        # Documentation
├── embedding_dataset/           # Pre-computed embeddings
├── fasta_files/                 # Input FASTA files
├── fine_tuning_datasets/        # Datasets for model fine-tuning
│   ├── combined_high/           # Combined high-quality datasets
│   ├── high_proline/            # High proline content dataset
│   ├── high_rk/                 # High arginine/lysine dataset
│   └── high_wyfl/               # High tryptophan/tyrosine/phenylalanine/leucine dataset
├── fine_tuning_embeddings/      # Embeddings for fine-tuning
│   └── [dataset_name]/          # Per-dataset embedding directories
│       ├── test_emb/
│       ├── train_emb/
│       └── val_emb/
├── models/                      # Model architectures
│   ├── __init__.py
│   ├── biLSTM_TextCNN.py       # BiLSTM + TextCNN model
│   ├── ffn.py                  # Feed-forward network
│   └── light_attention.py      # Light attention mechanism
├── scripts/                    # Utility scripts
│   ├── convert_fasta_headers_high_proline.py
│   └── merge_embeddings_and_fastas.py
├── utils/                      # Helper functions
│   └── general.py              # General utility functions
├── create_expanded_datasets.py  # Dataset expansion utility
├── create_fine_tuning_datasets.py  # Dataset preparation for fine-tuning
├── evaluate_baseline.py        # Baseline model evaluation
├── fine_tune_plm_sol.py        # Model fine-tuning script
├── fix_fasta_headers.py        # FASTA file processing utilities
├── fix_fasta_headers_v2.py     # Updated FASTA header processor
├── inference.py                # Main inference script
├── install_dependencies.py     # Dependency installation
├── plmsol_filtered_wrapper_v2.py # Filtered prediction wrapper
├── plmsol_predict_wrapper.py   # Main prediction wrapper
├── run_bulk_prediction.py      # Bulk prediction utility
├── solver.py                   # Training optimization logic
└── train.py                    # Main training script
```

## Core Scripts

### `train.py`
- **Purpose**: Main script for training the PLM_Sol model.
- **Key Features**:
  - Handles model initialization
  - Manages training loop
  - Implements checkpointing
  - Logs training metrics
- **Dependencies**:
  - PyTorch
  - numpy
  - tqdm
  - Custom model architectures from `models/`
- **Usage**:
  ```bash
  python train.py --config configs/SOL_biLSTM_TextCNN.yml
  ```

### `inference.py`
- **Purpose**: Performs inference using a trained PLM_Sol model.
- **Key Features**:
  - Loads trained models
  - Processes input sequences
  - Generates solubility predictions
- **Dependencies**:
  - PyTorch
  - bio-embeddings (for sequence embedding)
  - numpy
- **Usage**:
  ```bash
  python inference.py --config configs/baseline_eval_config.yml
  ```

### `solver.py`
- **Purpose**: Implements the training and optimization logic.
- **Key Features**:
  - Optimizer configuration
  - Learning rate scheduling
  - Loss calculation
  - Gradient updates
- **Relationships**:
  - Called by `train.py` and `fine_tune_plm_sol.py`
  - Uses models from `models/`
  - Interacts with datasets from `datasets/`

## Model Training and Fine-tuning

### `fine_tune_plm_sol.py`
- **Purpose**: Fine-tunes the base PLM_Sol model on custom datasets.
- **Key Features**:
  - Loads pre-trained weights
  - Applies fine-tuning with custom datasets
  - Saves fine-tuned models
- **Dependencies**:
  - PyTorch
  - bio-embeddings
  - Custom data loaders
- **Usage**:
  ```bash
  python fine_tune_plm_sol.py --dataset_path /path/to/dataset --model_checkpoint /path/to/checkpoint
  ```

### `create_fine_tuning_datasets.py`
- **Purpose**: Prepares datasets for fine-tuning.
- **Key Features**:
  - Processes raw sequence data
  - Generates training/validation splits
  - Creates embedding files
- **Relationships**:
  - Creates datasets used by `fine_tune_plm_sol.py`
  - Outputs to `fine_tuning_datasets/`

### `evaluate_baseline.py`
- **Purpose**: Evaluates baseline model performance.
- **Key Features**:
  - Loads test datasets
  - Computes evaluation metrics
  - Generates performance reports
- **Usage**:
  ```bash
  python evaluate_baseline.py --config configs/baseline_eval_config.yml
  ```

## Inference and Prediction

### `plmsol_predict_wrapper.py`
- **Purpose**: Wrapper script for integrating PLM_Sol with external systems.
- **Key Features**:
  - Standardizes input/output formats
  - Handles sequence embedding
  - Manages temporary files
  - Returns predictions in a consistent format
- **Usage**:
  ```bash
  python plmsol_predict_wrapper.py --fasta input.fasta --out predictions.csv
  ```

### `plmsol_filtered_wrapper_v2.py`
- **Purpose**: Advanced wrapper with filtering capabilities.
- **Key Features**:
  - Filters sequences by length or other criteria
  - Handles batch processing
  - Provides more control over prediction parameters
- **Relationships**:
  - Extends functionality of `plmsol_predict_wrapper.py`
  - Used for specialized prediction tasks

### `run_bulk_prediction.py`
- **Purpose**: Runs predictions on large sequence datasets.
- **Key Features**:
  - Processes FASTA files
  - Handles batching
  - Saves results to CSV
  - Supports model checkpoint selection
- **Usage**:
  ```bash
  python run_bulk_prediction.py --input large_dataset.fasta --output predictions.csv
  ```

## Data Processing Utilities

### `fix_fasta_headers.py` and `fix_fasta_headers_v2.py`
- **Purpose**: Processes FASTA file headers for compatibility.
- **Key Features**:
  - Standardizes header format
  - Handles special characters
  - Ensures unique sequence IDs
- **Improvements in v2**:
  - Better handling of complex headers
  - More robust error handling
  - Support for custom ID generation

### `scripts/merge_embeddings_and_fastas.py`
- **Purpose**: Combines sequence embeddings with FASTA files.
- **Key Features**:
  - Matches sequences to their embeddings
  - Handles large datasets efficiently
  - Maintains sequence-embedding correspondence
- **Usage**:
  ```bash
  python scripts/merge_embeddings_and_fastas.py --fasta input.fasta --embeddings embeddings.h5 --output merged.h5
  ```

## Model Architectures

### `models/biLSTM_TextCNN.py`
- **Purpose**: Implements a biLSTM + TextCNN hybrid architecture.
- **Key Features**:
  - Processes sequences with both LSTM and CNN
  - Captures both local and global sequence patterns
  - Configurable architecture parameters

### `models/ffn.py`
- **Purpose**: Implements a feed-forward neural network.
- **Key Features**:
  - Simple fully-connected architecture
  - Configurable hidden layers
  - Used as a baseline model

### `models/light_attention.py`
- **Purpose**: Implements a lightweight attention mechanism.
- **Key Features**:
  - Improves model interpretability
  - Highlights important sequence regions
  - Can be combined with other architectures

## Dataset Handling

### `datasets/embeddings_dataset.py`
- **Purpose**: Handles loading and processing of sequence embeddings.
- **Key Features**:
  - Supports multiple embedding formats
  - Implements data augmentation
  - Handles batching and shuffling

### `datasets/transforms.py`
- **Purpose**: Implements data transformations.
- **Key Features**:
  - Sequence padding
  - Data normalization
  - One-hot encoding

## Dependencies

### Core Dependencies
- Python 3.7+
- PyTorch 1.8.0+
- bio-embeddings 0.1.5+
- numpy 1.19.0+
- pandas 1.2.0+
- tqdm 4.50.0+
- PyYAML 5.3.1+
- scikit-learn 0.24.0+

### Environment
- Requires specific conda environment (PLM_Sol)
- GPU acceleration recommended for training
- CUDA 11.1+ for GPU support

## Data Flow

1. **Input**:
   - FASTA files containing protein sequences
   - Configuration files for model parameters
   - Pre-computed embeddings (optional)

2. **Processing**:
   - Sequence embedding generation (if not pre-computed)
   - Model inference or training
   - Post-processing of results
   - Evaluation against ground truth (for training/validation)

3. **Output**:
   - Solubility predictions (CSV format)
   - Trained model checkpoints (.t7 files)
   - Log files and metrics
   - Visualization plots

## Integration with PeptideFrontEnd

The PLM_Sol predictor is integrated into the PeptideFrontEnd through wrapper scripts that:
1. Handle environment isolation
2. Standardize input/output formats
3. Manage temporary files
4. Provide consistent error handling

The main integration points are:
- `plmsol_predict_wrapper.py` for prediction
- `run_bulk_prediction.py` for batch processing
- Configuration files in `configs/` for model parameters

## Maintenance and Cleanup

### Known Issues
- Some scripts have hardcoded paths that may need updating
- Redundant functionality between similar scripts
- Inconsistent error handling across components

### Recommendations
1. **Code Organization**:
   - Consolidate duplicate functionality
   - Standardize file naming and locations
   - Move hardcoded paths to configuration

2. **Documentation**:
   - Add comprehensive docstrings
   - Document expected file formats
   - Create usage examples

3. **Testing**:
   - Add unit tests for all components
   - Implement integration tests
   - Add performance benchmarks

4. **Error Handling**:
   - Standardize error messages
   - Add input validation
   - Improve error recovery

5. **Performance**:
   - Profile and optimize critical paths
   - Add support for distributed training
   - Implement better checkpointing

6. **Dependencies**:
   - Pin dependency versions
   - Document environment setup
   - Add dependency checking

7. **Future Work**:
   - Support for additional model architectures
   - Integration with more embedding methods
   - Enhanced visualization capabilities
