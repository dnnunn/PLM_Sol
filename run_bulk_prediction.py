#!/usr/bin/env python3
"""
Standalone Bulk Prediction Script for PLM_Sol
---------------------------------------------

This script allows for running PLM_Sol predictions on a large FASTA file and generating a persistent CSV output. 
It is designed for model evaluation and can be used with either the default base model or a specific, user-provided model checkpoint.

This script does NOT interfere with the main PeptideFrontEnd genetic algorithm workflow.

Usage:

# To run with the default base model:
python run_bulk_prediction.py --fasta /path/to/your/dataset.fasta --out /path/to/your/results_base_model.csv

# To run with a new, improved model:
python run_bulk_prediction.py --fasta /path/to/your/dataset.fasta --out /path/to/your/results_new_model.csv --model_checkpoint /path/to/your/model-10.t7

"""
import argparse
import os
import subprocess
import sys

def main():
    """Main function to orchestrate the bulk prediction process."""
    parser = argparse.ArgumentParser(description="Run bulk PLM_Sol predictions for model evaluation.")
    parser.add_argument('--fasta', '-f', required=True, help='Path to the input FASTA file for bulk prediction.')
    parser.add_argument('--out', '-o', required=True, help='Path to the output CSV file for storing results.')
    parser.add_argument('--model_checkpoint', '-m', help='(Optional) Path to a specific model checkpoint file (.t7). If not provided, the default base model will be used.')
    args = parser.parse_args()

    # The PLM_Sol wrapper script must be run from the PLM_Sol root directory.
    plmsol_root = os.path.dirname(os.path.abspath(__file__))
    wrapper_script = os.path.join(plmsol_root, 'plmsol_predict_wrapper.py')

    if not os.path.exists(wrapper_script):
        print(f"Error: Wrapper script not found at {wrapper_script}", file=sys.stderr)
        sys.exit(1)

    # Construct the command to call the wrapper script
    # We assume this script is run from an environment that has 'conda' available.
    cmd = [
        "conda", "run", "-n", "PLM_Sol", # Assumes the standard conda environment name
        "python", wrapper_script,
        "--fasta", os.path.abspath(args.fasta),
        "--out", os.path.abspath(args.out)
    ]

    # If a specific model checkpoint is provided, add it to the command.
    # The wrapper script knows to use its default if this is not provided.
    if args.model_checkpoint:
        cmd.extend(["--model_checkpoint", os.path.abspath(args.model_checkpoint)])

    print(f"Executing command:\n{' '.join(cmd)}\n")

    try:
        # Execute the command from the PLM_Sol root directory
        result = subprocess.run(
            cmd,
            cwd=plmsol_root,
            check=True, # Raises an exception for non-zero exit codes
            capture_output=True,
            text=True
        )
        print("Prediction successful.")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"\nResults have been saved to: {os.path.abspath(args.out)}")

    except subprocess.CalledProcessError as e:
        print(f"Error: Bulk prediction failed with exit code {e.returncode}.", file=sys.stderr)
        print("STDOUT:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("STDERR:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
