# PLM_Sol Integration Guide

This document outlines how to run PLM_Sol for peptide solubility prediction and integrate it with other components of the PeptideFusion project.

## Key Components

- **Embedding Generation**: First step that converts protein sequences to embeddings
- **Inference**: Second step that uses the embeddings to predict solubility
- **Output Conversion**: Transforms PLM_Sol output format to benchmark format

## Critical Requirements

1. **Working Directory Handling**:
   - Embedding generation can run from any directory with proper config paths
   - Inference **MUST** be run from the PLM_Sol root directory due to hardcoded output path

2. **Configuration Parameters**:
   - `key_format` in inference config **MUST** match the format of keys in embedding file
     - Use `key_format: id` for protein IDs (e.g., `ace_inhibitory_1`)
     - Use `key_format: hash` only if embedding keys are actual hash values
   - Output path is always hardcoded as `protTrans_prediction_result.csv` in PLM_Sol root
   - `output_files_name` parameter doesn't affect the actual output path

3. **File Path Structure**:
   - Model parameter: `/path/to/PLM_Sol/model_param/model_param.t7`
   - Embedding output: `/path/to/PLM_Sol/plmsol_test/random_emb/t5_embeddings/embeddings_file.h5`
   - Inference output: `/path/to/PLM_Sol/protTrans_prediction_result.csv` (hardcoded)

## Running Tests

The `run_random_test.py` script in `plmsol_test/` demonstrates the complete workflow:

1. Generates embeddings for test proteins
2. Runs inference on the embeddings
3. Converts output to benchmark format

```bash
cd ~/PLM_Sol/plmsol_test
python run_random_test.py
```

## Troubleshooting

- **No output file**: Check working directory (must be PLM_Sol root) and key_format
- **Embeddings issues**: Verify embedding file exists and has expected content
- **Model issues**: Ensure model_param.t7 exists in expected location

## Integration Notes

PLM_Sol has specific requirements that differ from other predictors:
1. Two-step process (embedding then inference)
2. Fixed hardcoded output file name
3. Working directory sensitivity
4. Key format must match embedding file keys

These factors must be considered when integrating PLM_Sol with other systems.
