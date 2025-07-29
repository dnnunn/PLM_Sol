# PLM_Sol Script Reference & Directory Structure

This document summarizes the purpose, usage, and status of all major scripts in the `PLM_Sol` directory. It also documents the new organization: only production, server-based, and parallelized scripts remain in the main directory. Benchmarking, fine-tuning, and troubleshooting scripts are moved to dedicated subfolders.

---

## Production/Active Scripts (Root Directory)

| Script                        | Purpose/When to Use                                 | Inputs/Args                                    |
|-------------------------------|-----------------------------------------------------|------------------------------------------------|
| `plm_embedding_server.py`     | Persistent embedding server for PLM_Sol             | --port, --model_path                           |
| `plmsol_filtered_wrapper_v2.py`| Main batch inference; accepts server embeddings     | --fasta, --out, --model_checkpoint, --server_embeddings_file |
| `plmsol_server_wrapper.py`    | Simple wrapper for server-based inference           | --fasta, --out, --embeddings_file              |


## Moved to `benchmarking_and_finetuning/` (Legacy/For Reference)

| Script                        | Purpose/When to Use                                 | Inputs/Args                                    |
|-------------------------------|-----------------------------------------------------|------------------------------------------------|
| `plmsol_predict_wrapper.py`   | Minimal batch wrapper for benchmarking only         | --fasta, --out, --model_checkpoint             |
| `plmsol_end_to_end_predict.py`| End-to-end test: embedding server + inference       | --fasta, --out, --embeddings_server_url        |
| `plmsol_direct_predict.py`    | Direct inference, no server (legacy)                | --fasta, --out                                 |


## Moved to `troubleshooting/` (For Debugging/Development)

- Scripts for investigating embedding or inference separately, or for debugging integration issues, are placed here. Each script includes a note at the top explaining its purpose.


## Fine-tuning/Training

- All scripts related to model fine-tuning, dataset creation, or training are in `fine_tuning/` or `fine_tuning_datasets/`.


---

## Notes
- **Only use scripts in the root directory for production inference.**
- Legacy and benchmarking scripts are preserved in subfolders for reference or possible future use.
- Any script not using server-side embedding and parallel batch inference is either removed or moved out of the main directory.
- Each script now includes an inline comment at the top explaining its status and correct usage.
