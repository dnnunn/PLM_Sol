# PLM_Sol & PeptideFrontEnd Integration Changelog

## 2025-07-25
- Major project cleanup: removed obsolete/duplicate files, Legacy directory, and test artifacts.
- Confirmed `clean_sequences.fasta` as the definitive benchmarking FASTA; deleted `all_sequences.fasta` to avoid confusion.
- Synchronized all FASTA files between VM and local; established `clean_sequences.fasta` as the only benchmarking input.
- Created new embedding and evaluation YAMLs for benchmarking (`embedding_clean_benchmark.yml`, `eval_clean_benchmark.yml`).
- Documented workflow for generating T5 embeddings and running inference with both baseline (`model_param.pth`) and improved (`model-10.t7`) models.
- Clarified correct usage of `bio_embeddings` CLI for embedding generation.
- Confirmed one-to-one correspondence between FASTA entries and result CSV rows in benchmarking.
- Established best practices for file transfer (gcloud scp) and project synchronization.
- Outlined requirements for comprehensive documentation and integration guides to support future work and avoid repeated issues.

---

## [Older entries can be added here as needed]
