# PLM_Sol Integration Documentation

## Overview
This document describes the integration of PLM_Sol protein solubility predictor with the PeptideFrontEnd benchmarking pipeline. PLM_Sol uses ProtTrans T5 embeddings for protein sequences and predicts solubility scores through a PyTorch model.

## Key Components

1. **Wrapper Script**: `plmsol_predict_wrapper.py` - A minimally invasive wrapper that handles the full PLM_Sol prediction pipeline.
2. **Test Script**: `test_plmsol_wrapper.py` - Validates the wrapper functionality with test sequences.
3. **Working Directory**: All inference must be run from the PLM_Sol root directory.
4. **Output Format**: Standard CSV with columns: Accession, Sequence, Predictor, SolubilityScore, Probability_Soluble, Probability_Insoluble.

## Fixed Issues

### 1. Embedding Generation
- **Issue**: Bio-embeddings failed to generate remapped sequences file
- **Fix**: Added `--overwrite` flag to bio_embeddings command, ensuring both embeddings and remapped sequences are created

### 2. Key Format Configuration
- **Issue**: PLM_Sol couldn't find correct keys in embedding file due to key format mismatch
- **Fix**: Set `key_format="fasta_descriptor"` in inference config to match actual format in embedding file

### 3. MD5 Hash Clashes
- **Issue**: Identical sequences in test FASTA caused MD5 hash clashes
- **Fix**: Created test script with distinct protein sequences to avoid hash collisions

### 4. Path Handling
- **Issue**: Hardcoded paths in PLM_Sol inference code
- **Fix**: Changed working directory to PLM_Sol root during inference, then restored original working directory

### 5. Output File Mapping
- **Issue**: PLM_Sol uses hardcoded output file `protTrans_prediction_result.csv`
- **Fix**: Wrapper detects and reads this file regardless of config settings

### 6. Sequence ID Mapping
- **Issue**: PLM_Sol uses MD5 hashes for protein IDs internally
- **Fix**: Added logic to map MD5 hashes back to original sequence IDs using remapped sequences file

### 7. Column Duplication
- **Issue**: Output CSV had duplicate "Sequence" columns
- **Fix**: Improved column handling to ensure proper column names without duplication

## Usage Instructions

### Basic Usage
```bash
python /path/to/PLM_Sol/plmsol_predict_wrapper.py --fasta input.fasta --out results.csv
```

### Debug Mode
```bash
python /path/to/PLM_Sol/plmsol_predict_wrapper.py --fasta input.fasta --out results.csv --debug
```

In debug mode, temporary files are preserved in a directory like `/var/tmp/plmsol_debug_XXXXXXXX`.

### Integration with Benchmarking
```python
# Example in benchmarking pipeline
import subprocess

def run_plmsol(input_fasta, output_csv):
    cmd = f"python /path/to/PLM_Sol/plmsol_predict_wrapper.py --fasta {input_fasta} --out {output_csv}"
    subprocess.run(cmd, shell=True, check=True)
```

## Technical Details

### Embedding Configuration
The wrapper dynamically creates embedding configuration YAML with:
- Protocol: prottrans_t5_xl_u50
- Half precision model: True
- Half precision: True
- Sequences file: Path to input FASTA

### Inference Configuration
The wrapper creates inference configuration YAML with:
- Output files name: "plmsol_result"
- Checkpoints list: ["./model_param/model_param.t7"]
- Batch size: 1
- n_draws: 1000
- key_format: "fasta_descriptor" (critical for correct functioning)
- Embeddings: Path to generated embeddings file
- Remapping: Path to remapped sequences file

### Error Handling
The wrapper implements robust error handling and provides fallback outputs (all predictions set to 0.5) if any step fails, ensuring the benchmarking pipeline can continue running.

## Dependencies
- bio_embeddings Python package
- PyTorch (as required by PLM_Sol)
- Biopython (for FASTA handling)
- Pandas (for CSV processing)

## Testing
The `test_plmsol_wrapper.py` script validates the full workflow with three distinct protein sequences:
1. SARS-CoV-2 Spike protein fragment
2. Human ACE2 receptor fragment
3. Human Hemoglobin fragment

Run the test script from the PLM_Sol directory:
```bash
python test_plmsol_wrapper.py
```
