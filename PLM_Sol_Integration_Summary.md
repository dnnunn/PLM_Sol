# PLM_Sol Persistent Embedding Server Integration & Benchmarking

## Overview
This document summarizes the steps, design decisions, and performance benchmarking for integrating PLM_Sol with a persistent embedding server, as well as direct (non-server) benchmarking. It provides recommendations for batch processing and outlines best practices for integration with the PeptideFrontEnd genetic algorithm (GA) workflow.

---

## 1. Problem Diagnosis & Solution Design
- **Root Cause:** Slow inference due to repeated loading of large PLM models (e.g., T5) for each batch.
- **Solution:** Implement a persistent embedding server to keep the model loaded in memory, serving embeddings via HTTP API.

---

## 2. Implementation Steps

### A. Environment & Dependencies
- Ensured all dependencies (bio_embeddings, Flask, etc.) were installed and up to date.

### B. Persistent Embedding Server
- Created `plm_embedding_server.py`:
  - Loads the T5 model once at startup.
  - Exposes `/embeddings` endpoint for batch embedding requests (POST JSON: `{ "sequences": [ ... ] }`).
  - Returns embeddings as JSON.
  - Health and stats endpoints for monitoring.

### C. Client Integration
- Updated client workflow (`plmsol_end_to_end_predict.py`):
  - Parses FASTA, POSTs sequences to the server.
  - Saves returned embeddings as JSON.
  - Runs PLM_Sol inference with server-generated embeddings.
  - Outputs predictions to `solubility_predictor_results.csv`.

### D. Direct (Non-Server) Benchmarking
- Created `plmsol_direct_predict.py`:
  - Runs the original PLM_Sol pipeline (embedding + inference) using `plmsol_predict_wrapper.py`.
  - Used for apples-to-apples performance comparison with the server approach.

### E. Output Standardization & Diagnostics
- Standardized output filenames to avoid confusion with legacy files.
- Added detailed logging and debug output to both server and client scripts.
- Verified correctness and performance with both small and large FASTA files.

---

## 3. Performance Benchmarking

### A. Small Batch Example (10 sequences)
- **Server method:** ~21s (no model reload per batch)
- **Direct method:** ~17–23s (includes model load, embedding, inference)
- **Note:** Progress bar only shows embedding time, not model load overhead.

### B. Large Batch Example (500 sequences, 10 batches of 50)
- **Direct method:** ~12 min (model loaded 10x, 1 per batch)
- **Server method:** ~10.2 min (model loaded once, reused for all batches)
- **Savings:** ~15% faster for server method; savings increase as number of batches/generations grows.

---

## 4. Batch Size Recommendations for GA Workflow
- **Optimal batch size:** 50–100 sequences per POST (empirically balances GPU utilization and memory usage).
- **Larger batches:** Fewer requests, better GPU utilization, but higher memory use.
- **Smaller batches:** More requests, less memory, but more overhead.
- **Best practice:** Start with batch size 50; test 64 or 100 if memory allows. Avoid >128 unless you have ample GPU RAM.

---

## 5. Best Practices & Next Steps
- Use the persistent embedding server for all GA/production workflows.
- Monitor GPU memory and adjust batch size as needed.
- Integrate server workflow into the PeptideFrontEnd GA loop for maximum speedup.
- For repeated or interactive workflows, the persistent server eliminates repeated model load time, compounding time savings.

---

## 6. References & Supporting Scripts
- `plm_embedding_server.py`: Persistent embedding server
- `plmsol_end_to_end_predict.py`: Client script for server-based workflow
- `plmsol_direct_predict.py`: Direct benchmarking script
- `plmsol_predict_wrapper.py`: Batch inference wrapper

---

**For further details, see in-code comments and debug logs.**
