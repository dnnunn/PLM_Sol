#!/bin/bash
# Clean Dataset Benchmarking Script
# Runs all 4 predictors on 16646 clean sequences

echo "🧬 CLEAN DATASET BENCHMARKING"
echo "Sequences: 16646 (all processable by every predictor)"
echo "Dataset: clean_benchmark_dataset/clean_sequences.fasta"
echo ""

# Create output directory
OUTPUT_DIR="clean_benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Run SWI
echo "🔬 Running SWI..."
cd ~/SWI
conda activate SWI
python swi_predict_wrapper.py --fasta ~/solubility_data_screening/clean_benchmark_dataset/clean_sequences.fasta --out ~/solubility_data_screening/$OUTPUT_DIR/swi_clean_results.csv
echo "✅ SWI complete"
echo ""

# Run DSResSol  
echo "🔬 Running DSResSol..."
cd ~/DSResSol
conda activate DsResSol
python dsressol_predict_wrapper_fixed.py --fasta ~/solubility_data_screening/clean_benchmark_dataset/clean_sequences.fasta --out ~/solubility_data_screening/$OUTPUT_DIR/dsressol_clean_results.csv
echo "✅ DSResSol complete"
echo ""

# Run PLM_Sol
echo "🔬 Running PLM_Sol..."
cd ~/solubility_data_screening
conda activate PLM_Sol
python plmsol_filtered_wrapper_v2.py --fasta clean_benchmark_dataset/clean_sequences.fasta --out $OUTPUT_DIR/plmsol_clean_results.csv
echo "✅ PLM_Sol complete"
echo ""

# Run DeepSoluE
echo "🔬 Running DeepSoluE..."
cd ~/solubility_data_screening
conda activate DeepSoluE
python deepsol_predict_wrapper.py --fasta clean_benchmark_dataset/clean_sequences.fasta --out $OUTPUT_DIR/deepsol_clean_results.csv
echo "✅ DeepSoluE complete"
echo ""

echo "🎉 All predictors complete!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Transfer results with:"
echo "gcloud compute scp --recurse peptide-engine-california-july-17:~/solubility_data_screening/$OUTPUT_DIR ~/Desktop/Apps/PeptideFusionProject/solubility_data_screening/ --zone=us-west4-a"
