#!/usr/bin/env python3
"""
End-to-end PLM_Sol prediction script for GA workflow integration.

1. POSTs input FASTA to persistent embedding server
2. Saves returned embeddings as JSON
3. Calls plmsol_server_wrapper.py for inference
4. Prints timing and verifies output

Usage:
    python plmsol_end_to_end_predict.py --fasta INPUT.fasta --out OUTPUT.csv --embeddings_server_url http://localhost:5000/get_embeddings

Note: Model checkpoint is hardcoded in plmsol_server_wrapper.py
"""
import argparse
import subprocess
import requests
import time
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="End-to-end PLM_Sol prediction (embedding server + inference)")
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--out', required=True, help='Output CSV file')
    parser.add_argument('--embeddings_server_url', required=True, help='Embedding server endpoint URL (e.g. http://localhost:5000/get_embeddings)')
    parser.add_argument('--tmp_embeddings', default=None, help='Temp embeddings JSON path (optional)')
    args = parser.parse_args()

    fasta_path = os.path.abspath(args.fasta)
    out_path = os.path.abspath(args.out)
    embeddings_json = args.tmp_embeddings or os.path.splitext(out_path)[0] + '_embeddings.json'

    print(f"[1/3] Requesting embeddings from server for: {fasta_path}")
    t0 = time.time()
    with open(fasta_path, 'rb') as f:
        response = requests.post(args.embeddings_server_url, files={'file': f})
    if response.status_code != 200:
        print(f"ERROR: Embedding server returned status {response.status_code}")
        sys.exit(1)
    with open(embeddings_json, 'wb') as out_f:
        out_f.write(response.content)
    t1 = time.time()
    print(f"  Embeddings JSON saved to: {embeddings_json} ({t1-t0:.2f}s)")

    print(f"[2/3] Running PLM_Sol inference with server embeddings...")
    t2 = time.time()
    result = subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), 'plmsol_server_wrapper.py'),
        '--fasta', fasta_path,
        '--embeddings_file', embeddings_json,
        '--out', out_path
    ], capture_output=True, text=True)
    t3 = time.time()
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"ERROR: PLM_Sol inference failed.")
        sys.exit(1)
    print(f"  Inference completed ({t3-t2:.2f}s)")

    print(f"[3/3] Total end-to-end time: {t3-t0:.2f}s")
    if os.path.exists(out_path):
        print(f"Success: Predictions written to {out_path}")
    else:
        print(f"ERROR: Output CSV not found: {out_path}")
        sys.exit(1)

if __name__ == '__main__':
    main()
