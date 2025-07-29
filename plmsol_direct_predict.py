#!/usr/bin/env python3
"""
Direct PLM_Sol prediction script (no embedding server).

Runs the original PLM_Sol inference pipeline on a FASTA file, including embedding generation.

Usage:
    python plmsol_direct_predict.py --fasta INPUT.fasta --out OUTPUT.csv

Note: Model checkpoint is hardcoded as in the server wrapper.
"""
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Direct PLM_Sol prediction (no embedding server)")
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--out', required=True, help='Output CSV file')
    args = parser.parse_args()

    fasta_path = os.path.abspath(args.fasta)
    out_path = os.path.abspath(args.out)
    # Hardcoded model checkpoint (same as server wrapper)
    MODEL_CHECKPOINT = '/home/david_nunn/PLM_Sol/saved_models/model-10.t7'

    print(f"[1/2] Running PLM_Sol direct inference (includes embedding generation)...")
    result = subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), 'plmsol_predict_wrapper.py'),
        '--fasta', fasta_path,
        '--out', out_path,
        '--model_checkpoint', MODEL_CHECKPOINT
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"ERROR: PLM_Sol direct inference failed.")
        sys.exit(1)
    print(f"[2/2] Output written to: {out_path}")
    if os.path.exists(out_path):
        print(f"Success: Predictions written to {out_path}")
    else:
        print(f"ERROR: Output CSV not found: {out_path}")
        sys.exit(1)

if __name__ == '__main__':
    main()
