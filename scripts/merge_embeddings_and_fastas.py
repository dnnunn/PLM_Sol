import os
import h5py
from Bio import SeqIO
import argparse

# Helper to merge FASTA files
def merge_fastas(fasta_paths, output_path):
    seen = set()
    with open(output_path, 'w') as out:
        for fasta in fasta_paths:
            for record in SeqIO.parse(fasta, 'fasta'):
                if record.id not in seen:
                    SeqIO.write(record, out, 'fasta')
                    seen.add(record.id)

def merge_h5s(h5_paths, fasta_paths, output_path):
    # Collect all sequence IDs in merged FASTA order
    all_ids = []
    for fasta in fasta_paths:
        for record in SeqIO.parse(fasta, 'fasta'):
            all_ids.append(record.id)
    # Map from ID to (file, key)
    id_to_filekey = {}
    for h5_path, fasta in zip(h5_paths, fasta_paths):
        with h5py.File(h5_path, 'r') as h5:
            for record in SeqIO.parse(fasta, 'fasta'):
                if record.id in h5:
                    id_to_filekey[record.id] = (h5_path, record.id)
                else:
                    # Try to find matching key (sometimes IDs are hashes)
                    for key in h5.keys():
                        if key.startswith(record.id):
                            id_to_filekey[record.id] = (h5_path, key)
                            break
    # Write merged h5
    with h5py.File(output_path, 'w') as out_h5:
        for seq_id in all_ids:
            h5_path, key = id_to_filekey[seq_id]
            with h5py.File(h5_path, 'r') as in_h5:
                in_h5.copy(key, out_h5)

def main():
    parser = argparse.ArgumentParser(description='Merge FASTA and embedding HDF5 files for PLM_Sol evaluation.')
    parser.add_argument('--fasta', nargs='+', required=True, help='Input FASTA files (train, test, val)')
    parser.add_argument('--h5', nargs='+', required=True, help='Input HDF5 embedding files (train, test, val)')
    parser.add_argument('--out_fasta', required=True, help='Output merged FASTA')
    parser.add_argument('--out_h5', required=True, help='Output merged HDF5 embeddings')
    args = parser.parse_args()

    print(f"Merging FASTA files: {args.fasta} -> {args.out_fasta}")
    merge_fastas(args.fasta, args.out_fasta)
    print(f"Merging HDF5 embedding files: {args.h5} -> {args.out_h5}")
    merge_h5s(args.h5, args.fasta, args.out_h5)
    print("Done.")

if __name__ == '__main__':
    main()
