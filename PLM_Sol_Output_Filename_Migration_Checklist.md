# PLM_Sol Output Filename Migration Checklist

**Objective:**
Migrate all relevant scripts, predictors, and test/debug tools from using `protTrans_prediction_result.csv` to `solubility_predictor_results.csv` as the standard output file for PLM_Sol predictions.

This checklist documents all files and code locations that must be updated for a clean and robust migration, ensuring no legacy or vestigial references remain. Use this as a record for future audits or if a rollback is required.

---

## 1. Wrapper and Core Pipeline
- [x] `PLM_Sol/plmsol_predict_wrapper.py`
  - [x] Config: output_files_name set to `solubility_predictor_results.csv`
  - [x] Output file check and return logic updated
  - [x] Cleans up both old and new output files before run

## 2. Predictors and Server-Based Wrappers
- [ ] `PeptideFrontEnd/genetic_algorithm/plm_sol_predictor_enhanced.py`
  - [ ] Any hardcoded references to old output filename
  - [ ] Output parsing logic (ensure reading new file)
- [ ] `PeptideFrontEnd/genetic_algorithm/plm_sol_predictor_server.py`
  - [ ] Output file name/path for wrapper calls and parsing
- [ ] `PeptideFrontEnd/genetic_algorithm/plm_sol_predictor_server_debug.py`
  - [ ] Output file name/path for wrapper calls and parsing
- [ ] `PeptideFrontEnd/genetic_algorithm/plm_sol_factory.py`
  - [ ] Any references to output filename (direct or via factory)

## 3. Test, Benchmark, and Integration Scripts
- [ ] `PLM_Sol/run_bulk_prediction.py`
  - [ ] Output file name/path for wrapper calls and parsing
- [ ] `PLM_Sol/run_clean_benchmark.sh`
  - [ ] Any hardcoded output filename for PLM_Sol
- [ ] `PeptideFrontEnd/genetic_algorithm/test_deap_integration.py`
  - [ ] Output parsing logic
- [ ] `PeptideFrontEnd/genetic_algorithm/example_deap_optimization.py`
  - [ ] Output parsing logic

## 4. Fitness Evaluator and Downstream Consumers
- [ ] `PeptideFrontEnd/genetic_algorithm/fitness_evaluator.py`
  - [ ] Any logic that expects the old output filename
  - [ ] Output parsing logic

## 5. Documentation and Reports
- [ ] Any markdown files, READMEs, or integration reports referencing the old output filename (search for `protTrans_prediction_result.csv`)

---

## Migration Steps
1. Update all code to use `solubility_predictor_results.csv` as the output filename for PLM_Sol predictions.
2. Search for and remove or update any remaining references to `protTrans_prediction_result.csv`.
3. Test all predictors, wrappers, and integration scripts to confirm correct operation and output parsing.
4. Update documentation to reflect the new output filename.
5. Archive this checklist as a record of the migration.

---

**Note:**
If any issues arise or a rollback is required, use this checklist to identify all affected locations.

---

_Last updated: 2025-07-28_
