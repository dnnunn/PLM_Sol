# PLM_Sol & PeptideFrontEnd Integration Changelog

## 2025-07-25
- Major project cleanup: removed obsolete/duplicate files, Legacy directory, and test artifacts.
- Confirmed `clean_sequences.fasta` as the definitive benchmarking FASTA; deleted `all_sequences.fasta` to avoid confusion.
- Synchronized all FASTA files between VM and local; established `clean_sequences.fasta` as the only benchmarking input.
- Created new embedding and evaluation YAMLs for benchmarking (`embedding_clean_benchmark.yml`, `eval_clean_benchmark.yml`).
- **All YAML configs now use full absolute VM paths (`/home/david_nunn/PLM_Sol/...`) for compatibility; all workflow documentation updated accordingly.**
- Documented workflow for generating T5 embeddings and running inference with both baseline (`model_param.pth`) and improved (`model-10.t7`) models.
- Clarified correct usage of `bio_embeddings` CLI for embedding generation.
- Confirmed one-to-one correspondence between FASTA entries and result CSV rows in benchmarking.
- Established best practices for file transfer (gcloud scp) and project synchronization.
- Outlined requirements for comprehensive documentation and integration guides to support future work and avoid repeated issues.

## 2025-07-25 (Benchmarking & Integration Update)
- **Standardized PLM_Sol Benchmarking Output:**
    - All benchmarking runs now use `plmsol_filtered_wrapper_v2.py` to guarantee output CSVs (e.g., `plmsol_clean_results.csv`) contain every input sequence, even if some are filtered for length.
    - The wrapper delegates to the original PLM_Sol prediction script and merges results, assigning default scores to filtered-out sequences.
    - This ensures compatibility with benchmarking pipelines and downstream analysis.
- **Dual-Model Evaluation (Baseline vs Optimized):**
    - Documented workflow for running PLM_Sol with both the baseline model (`model_param.pth`) and the new optimized model (`model-10.t7`).
    - Output files for both runs are standardized, enabling direct comparison and detailed improvement analysis.
- **Forward-Looking PeptideFrontEnd Integration:**
    - The standardized output format and wrapper logic will be reused for integration with PeptideFrontEnd and the genetic algorithm workflow.
    - PeptideFrontEnd will consume the same CSV format for solubility predictions, ensuring seamless pipeline handoff.
    - Future GA runs can use the optimized PLM_Sol model and wrapper to evaluate candidate peptides, with guaranteed output structure regardless of input or model errors.
- **Best Practices:**
    - Always run the wrapper script for both baseline and optimized models to generate comparable results.
    - Maintain this output contract for any future integrations or model updates.

---

## [Older entries can be added here as needed]
