"""
Server Embeddings Dataset for PLM_Sol

This dataset class accepts pre-computed embeddings from the persistent embedding server
instead of loading them from H5 files, eliminating the 30s startup overhead.

Based on Embeddings_predict_Dataset but modified to use server-provided embeddings.
"""

from typing import Tuple, List, Dict
import torch
from Bio import SeqIO
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.general import AMINO_ACIDS


class ServerEmbeddings_predict_Dataset(Dataset):
    """
    Dataset that uses pre-computed embeddings from the persistent embedding server.
    
    This eliminates the need to load embeddings from H5 files, providing significant
    speedup when embeddings are already computed by the server.
    """
    
    def __init__(self, 
                 server_embeddings: List[List[float]], 
                 remapped_sequences: str,
                 key_format: str = 'hash',
                 max_length: int = float('inf'),
                 embedding_mode: str = 'lm',
                 transform=lambda x: x) -> None:
        """
        Initialize dataset with server-provided embeddings.
        
        Args:
            server_embeddings: List of embeddings from the persistent embedding server
            remapped_sequences: Path to remapped FASTA file with sequence metadata
            key_format: Format of keys in the FASTA file
            max_length: Maximum sequence length to include
            embedding_mode: Type of embeddings ('lm' for language model)
            transform: Transform to apply to embeddings
        """
        super().__init__()
        self.transform = transform
        self.embedding_mode = embedding_mode
        self.server_embeddings = server_embeddings
        self.solubility_metadata_list = []
        self.one_hot_enc = []
        
        # Parse sequences from remapped FASTA file
        embedding_index = 0
        for record in SeqIO.parse(open(remapped_sequences), 'fasta'):
            if key_format == 'hash':
                id = str(record.id)
            elif key_format == 'fasta_descriptor':
                id = str(record.description.split(' ')[0]).replace('.','_').replace('/','_')
            elif key_format == 'fasta_descriptor_old':
                id = str(record.description)
            else:
                raise Exception('Unknown key_format: ', key_format)
            
            if len(record.seq) <= max_length:
                # Handle one-hot encoding if needed
                if self.embedding_mode == 'onehot':
                    amino_acid_ids = []
                    for char in record.seq:
                        amino_acid_ids.append(AMINO_ACIDS[char])
                    one_hot_enc = F.one_hot(torch.tensor(amino_acid_ids), num_classes=len(AMINO_ACIDS))
                    self.one_hot_enc.append(one_hot_enc)
                
                # Calculate amino acid frequencies
                frequencies = torch.zeros(25)
                for i, aa in enumerate(AMINO_ACIDS):
                    frequencies[i] = str(record.seq).count(aa)
                frequencies /= len(record.seq)
                
                # Store metadata with embedding index
                metadata = {
                    'id': id,
                    'sequence': str(record.seq),
                    'length': len(record.seq),
                    'frequencies': frequencies,
                    'embedding_index': embedding_index  # Index into server_embeddings
                }
                
                self.solubility_metadata_list.append({'metadata': metadata})
                embedding_index += 1
        
        # Validate that we have the right number of embeddings
        if len(self.server_embeddings) != len(self.solubility_metadata_list):
            raise ValueError(f"Mismatch: {len(self.server_embeddings)} embeddings but {len(self.solubility_metadata_list)} sequences")
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        """
        Retrieve single sample from the dataset.
        
        Args:
            index: Index of sample to retrieve
            
        Returns:
            embedding: Tensor with pre-computed embedding from server
            metadata: Sequence metadata dictionary
        """
        solubility_metadata = self.solubility_metadata_list[index]
        
        if self.embedding_mode == 'lm':
            # Use pre-computed embedding from server
            embedding_index = solubility_metadata['metadata']['embedding_index']
            embedding = torch.tensor(self.server_embeddings[embedding_index], dtype=torch.float32)
        elif self.embedding_mode == 'onehot':
            embedding = self.one_hot_enc[index]
        else:
            raise Exception(f'embedding_mode {self.embedding_mode} not supported with server embeddings')
        
        # Apply transform
        embedding = self.transform(embedding)
        
        return embedding, solubility_metadata['metadata']
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.solubility_metadata_list)
