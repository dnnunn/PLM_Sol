#!/usr/bin/env python
"""
Debug script to test PLM_Sol file writing functionality
"""
import os
import pandas as pd
import sys

def test_csv_writing():
    """Test basic CSV writing in the current directory"""
    print("=== PLM_Sol CSV Writing Test ===")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Pandas version: {pd.__version__}")
    
    # Create simple test dataframe
    df = pd.DataFrame({
        'protein_ID': ['test1', 'test2'],
        'sequence': ['ACGT', 'AAAA'],
        'predict_result': [0.5, 0.7]
    })
    
    # Define output path
    output_path = 'test_prediction_result.csv'
    print(f"Attempting to write CSV to: {os.path.abspath(output_path)}")
    
    # Try to write the file
    try:
        df.to_csv(output_path, index=False)
        print(f"✓ Success! File written to {os.path.abspath(output_path)}")
        if os.path.exists(output_path):
            print(f"✓ File exists with size: {os.path.getsize(output_path)} bytes")
    except Exception as e:
        print(f"✗ Error writing file: {str(e)}")

    # Try writing the hardcoded filename
    try:
        hardcoded_path = 'protTrans_prediction_result.csv'
        df.to_csv(hardcoded_path, index=False)
        print(f"✓ Success! Hardcoded file written to {os.path.abspath(hardcoded_path)}")
        if os.path.exists(hardcoded_path):
            print(f"✓ Hardcoded file exists with size: {os.path.getsize(hardcoded_path)} bytes")
    except Exception as e:
        print(f"✗ Error writing hardcoded file: {str(e)}")
        
    print("\n=== Directory Contents After Test ===")
    for filename in os.listdir('.'):
        if filename.endswith('.csv'):
            print(f"- {filename}: {os.path.getsize(filename)} bytes")
    
if __name__ == "__main__":
    test_csv_writing()
