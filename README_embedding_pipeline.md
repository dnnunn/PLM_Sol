# PLM_Sol Embedding Generation Pipeline: YAML Config Template

## Correct YAML Structure for bio_embeddings (T5)

For every new dataset (train/val/test), use the following template for the YAML config:

```yaml
global:
  sequences_file: <path to your fasta file>
  prefix: <output prefix for embedding files>

t5_embeddings:
  type: embed
  protocol: prottrans_t5_xl_u50
  half_precision_model: True
  half_precision: True
```

- `sequences_file`: Path to the input FASTA file (e.g., `fine_tuning_datasets/high_proline/train.fixed.fasta`)
- `prefix`: Path and prefix for output embedding files (e.g., `fine_tuning_embeddings/high_proline/train_emb/train_emb`)

## Example for a train split:
```yaml
global:
  sequences_file: fine_tuning_datasets/high_proline/train.fixed.fasta
  prefix: fine_tuning_embeddings/high_proline/train_emb/train_emb

t5_embeddings:
  type: embed
  protocol: prottrans_t5_xl_u50
  half_precision_model: True
  half_precision: True
```

## Usage
Run embedding generation for each split:

```bash
bio_embeddings <your_config>.yml --overwrite
```

## Notes
- Do **not** use the PLM_Sol fine-tuning config as a template for embedding generation. The required keys and structure are different.
- If you add annotation extraction, add a section like:
  ```yaml
  annotations_from_t5:
    type: extract
    protocol: la_prott5
    depends_on: t5_embeddings
  ```
  (But this is not needed for standard fine-tuning.)

---

**Always use this template for new datasets to avoid compatibility errors.**
