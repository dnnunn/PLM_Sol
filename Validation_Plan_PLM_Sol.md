# PLM_Sol: Functional Validation Plan on VM

**Goal:**
Run PLM_Sol and generate solubility predictions for your proteins on your VM.

---

## Steps

1. **Install Dependencies**
   - Use the provided environment files:
     ```bash
     conda env create -f env.yml
     conda activate PLM_Sol
     pip install -r requirements.txt
     pip install bio-embeddings[all]
     ```

2. **Generate Embeddings**
   - Use bio-embeddings to generate `.h5` embedding files for your protein sequences:
     ```bash
     cd embedding_dataset
     # Adjust the config file paths as needed
     bio_embeddings embedding_protT5.yml
     ```

3. **Run Predictions**
   - Run the inference script:
     ```bash
     python inference.py --config ./configs/inference_Sol_biLSTM_TextCNN.yml
     ```

4. **Post-processing**
   - Use the provided notebook (`PLM_Sol_csv.ipynb`) to merge original and predicted CSV files if needed.

5. **Troubleshooting**
   - Resolve any errors related to dependencies or input formatting.
   - Refer to the README for additional guidance.

---

## Deliverable
PLM_Sol running on your VM and producing solubility predictions for your test proteins.
