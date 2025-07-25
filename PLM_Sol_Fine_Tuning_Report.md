# PLM_Sol Fine-Tuning & Integration Report

## 1. Project Objective

The primary goal was to fine-tune the `PLM_Sol` solubility prediction model on a specialized dataset enriched with proline, arginine, lysine, and other key amino acids relevant to the PeptideFrontEnd genetic algorithm workflows. The objective was to improve prediction accuracy for these specific peptide sequences.

## 2. Initial State & Challenges

The project began with a series of significant technical hurdles:

- **Broken Prediction Scripts:** The initial scripts for both single and batch predictions were non-functional, using incorrect wrappers, arguments, and parsing logic.
- **Environment and Dependency Issues:** The fine-tuning environment had numerous dependency conflicts and configuration errors, preventing the pipeline from running.
- **Incorrect Data Handling:** The fine-tuning pipeline failed repeatedly due to incorrect FASTA header formats and a misunderstanding of how the model loaded solubility labels.
- **Architecture Mismatches:** Evaluation scripts failed due to incorrect model architecture definitions, requiring manual code inspection to identify the correct model class (`biLSTM_TextCNN`).

## 3. Key Accomplishments & Fixes

Through a systematic debugging process, we successfully resolved all major issues:

- **Production-Ready Predictor:**
  - Re-engineered the batch prediction workflow to use a robust wrapper (`plmsol_predict_wrapper.py`).
  - Implemented sequence deduplication and caching, achieving a performance of **~0.2 seconds per unique sequence** and near-instantaneous results for duplicates, making it suitable for high-throughput GA workflows.

- **Successful Fine-Tuning Pipeline:**
  - Corrected all data-related errors by fixing FASTA header generation and ensuring the `key_format` in the configuration was set to `fasta_descriptor`.
  - Resolved all script errors in the `PLM_Sol` library, including fixing an off-by-one error in the data loader (`embeddings_dataset.py`) and removing unsupported arguments from the `Solver` class.
  - The full pipeline, from data preparation to model training, now runs to completion without errors.

## 4. Performance Evaluation

We established a baseline and evaluated the fine-tuned model on our specialized test set (`combined_high_1_5sigma`).

| Model              | Test Accuracy | Test Average Accuracy |
| :----------------- | :-----------: | :-------------------: |
| **Baseline Model** |   **66.5%**   |       **67.5%**       |
| **Fine-tuned Model** |    60.9%    |         62.2%         |

## 5. Diagnosis: Overfitting

The evaluation revealed that the fine-tuned model performed worse than the baseline. A detailed analysis of the training logs (`run.log`) confirmed the cause: **classic overfitting**.

- The model's accuracy on the training data consistently improved.
- The model's accuracy on the validation (test) data fluctuated and did not show sustained improvement, indicating that the model was memorizing the training data rather than generalizing.

## 6. Next Steps & Recommendations

The current fine-tuned model should **not** be deployed. The immediate next step is to address the overfitting problem:

1.  **Implement Early Stopping:** Modify the training loop to monitor validation loss and stop training when performance no longer improves. This is the highest priority.
2.  **Hyperparameter Tuning:** Experiment with a lower learning rate, increased dropout, or other regularization techniques to help the model generalize better.
3.  **Expand the Dataset:** If performance does not improve, consider using the `1σ` dataset, which is larger and may provide a more robust training signal.

This project has successfully laid the groundwork for effective fine-tuning. With these adjustments, we are well-positioned to produce a high-performing, specialized solubility predictor.
