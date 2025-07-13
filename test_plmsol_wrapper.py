#!/usr/bin/env python3
"""
Test script for PLM_Sol wrapper
-------------------------------

This script creates a small test FASTA file and runs the wrapper
to validate its functionality.
"""
import os
import subprocess
import tempfile
from pathlib import Path

def main():
    print("Testing PLM_Sol wrapper")
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small test FASTA with DIFFERENT sequences (to avoid MD5 clash)
        test_fasta = os.path.join(tmpdir, "test_seqs.fasta")
        with open(test_fasta, "w") as f:
            # Sequence 1 - Spike protein fragment
            f.write(">seq1 Spike protein fragment\n")
            f.write("MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVY\n")
            # Sequence 2 - Different protein fragment (ACE2)
            f.write(">seq2 ACE2 fragment\n")
            f.write("MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNL\n")
            # Sequence 3 - Another protein fragment (Hemoglobin)
            f.write(">seq3 Hemoglobin fragment\n")
            f.write("MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSEL\n")
        
        # Output path for results
        results_csv = os.path.join(tmpdir, "results.csv")
        
        # Get the directory of this test script - should be PLM_Sol root
        plmsol_root = os.path.dirname(os.path.abspath(__file__))
        wrapper_path = os.path.join(plmsol_root, "plmsol_predict_wrapper.py")
        
        print(f"Test FASTA: {test_fasta}")
        print(f"Results will be saved to: {results_csv}")
        print(f"Using wrapper at: {wrapper_path}")
        
        # Run the wrapper script
        cmd = [
            "python", wrapper_path,
            "--fasta", test_fasta,
            "--out", results_csv,
            "--debug"
        ]
        
        try:
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            
            # Print output
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            
            # Check if output was created
            if os.path.exists(results_csv):
                print(f"SUCCESS: Output file created at {results_csv}")
                
                # Read and display first few lines of the output
                with open(results_csv, 'r') as f:
                    lines = f.readlines()[:5]
                    print("First few lines of output:")
                    for line in lines:
                        print(line.strip())
                
                return True
            else:
                print(f"ERROR: Output file not created at {results_csv}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Error running wrapper: {e}")
            if e.stdout:
                print(f"STDOUT:\n{e.stdout}")
            if e.stderr:
                print(f"STDERR:\n{e.stderr}")
            return False

if __name__ == "__main__":
    success = main()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
