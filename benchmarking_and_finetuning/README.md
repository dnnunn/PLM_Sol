# Benchmarking & Fine-Tuning Scripts

This folder contains legacy and reference scripts for benchmarking PLM_Sol performance, running direct inference (without server), and end-to-end tests that are not used in production workflows.

## Scripts
- `plmsol_predict_wrapper.py`: Minimal batch wrapper for benchmarking.
- `plmsol_end_to_end_predict.py`: End-to-end test script (embedding server + inference).
- `plmsol_direct_predict.py`: Direct inference, no server (legacy).

**Do not use these scripts for production or GA workflows.**

Refer to `../PLM_Sol_Script_Reference.md` for a complete directory map and current workflow.
