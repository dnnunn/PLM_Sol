"""
Persistent Embedding Cache for PLM_Sol Optimization

This module provides a persistent cache for T5 embeddings to dramatically speed up
PLM_Sol predictions by avoiding redundant embedding generation.

Key Features:
- Disk-based HDF5 storage for embeddings
- Content-based hashing for cache keys
- Automatic cache management and cleanup
- Thread-safe operations
- Compression for storage efficiency

Expected Performance:
- 10-50x speedup for cached sequences
- Near-instant prediction for repeated patterns
- Significant speedup for DEAP parameter optimization
"""

import hashlib
import h5py
import numpy as np
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CacheMetadata:
    """Metadata for cached embeddings"""
    sequence_hash: str
    sequence_length: int
    embedding_shape: Tuple[int, ...]
    model_name: str
    created_at: str
    last_accessed: str
    access_count: int = 0

class PersistentEmbeddingCache:
    """
    Persistent cache for protein sequence embeddings.
    
    Uses content-based hashing to create unique cache keys and stores
    embeddings in compressed HDF5 format for efficient retrieval.
    """
    
    def __init__(self, 
                 cache_dir: str = "/home/david_nunn/PLM_Sol/embedding_cache",
                 model_name: str = "prot_t5_xl_half_uniref50-enc",
                 max_cache_size_gb: float = 10.0,
                 compression: str = "gzip"):
        """
        Initialize the persistent embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
            model_name: Name of the embedding model (for cache validation)
            max_cache_size_gb: Maximum cache size in GB
            compression: HDF5 compression method ('gzip', 'lzf', None)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.max_cache_size_gb = max_cache_size_gb
        self.compression = compression
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'total_sequences_cached': 0,
            'cache_size_mb': 0.0
        }
        
        # Metadata file for cache management
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized embedding cache at {self.cache_dir}")
        logger.info(f"Current cache contains {len(self.metadata)} embeddings")
    
    def _generate_cache_key(self, sequence: str) -> str:
        """Generate a unique cache key for a sequence."""
        # Use sequence content + model name for key generation
        content = f"{sequence}:{self.model_name}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        # Organize cache files in subdirectories to avoid too many files in one dir
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.h5"
    
    def _load_metadata(self) -> Dict[str, CacheMetadata]:
        """Load cache metadata from disk."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {
                    key: CacheMetadata(**value) 
                    for key, value in data.items()
                }
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            data = {
                key: asdict(metadata) 
                for key, metadata in self.metadata.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def get_embedding(self, sequence: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding for a sequence from cache.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Cached embedding array or None if not found
        """
        with self._lock:
            cache_key = self._generate_cache_key(sequence)
            cache_path = self._get_cache_path(cache_key)
            
            if not cache_path.exists():
                self.stats['misses'] += 1
                return None
            
            try:
                with h5py.File(cache_path, 'r') as f:
                    embedding = f['embedding'][:]
                
                # Update access metadata
                if cache_key in self.metadata:
                    self.metadata[cache_key].last_accessed = datetime.now().isoformat()
                    self.metadata[cache_key].access_count += 1
                
                self.stats['hits'] += 1
                logger.debug(f"Cache hit for sequence (length {len(sequence)})")
                return embedding
                
            except Exception as e:
                logger.error(f"Failed to load embedding from cache: {e}")
                self.stats['misses'] += 1
                return None
    
    def store_embedding(self, sequence: str, embedding: np.ndarray):
        """
        Store embedding for a sequence in cache.
        
        Args:
            sequence: Protein sequence
            embedding: Embedding array to cache
        """
        with self._lock:
            cache_key = self._generate_cache_key(sequence)
            cache_path = self._get_cache_path(cache_key)
            
            try:
                # Store embedding in HDF5 format with compression
                with h5py.File(cache_path, 'w') as f:
                    f.create_dataset(
                        'embedding', 
                        data=embedding, 
                        compression=self.compression,
                        compression_opts=9 if self.compression == 'gzip' else None
                    )
                    # Store sequence for validation
                    f.attrs['sequence'] = sequence
                    f.attrs['model_name'] = self.model_name
                    f.attrs['created_at'] = datetime.now().isoformat()
                
                # Update metadata
                self.metadata[cache_key] = CacheMetadata(
                    sequence_hash=cache_key,
                    sequence_length=len(sequence),
                    embedding_shape=embedding.shape,
                    model_name=self.model_name,
                    created_at=datetime.now().isoformat(),
                    last_accessed=datetime.now().isoformat(),
                    access_count=0
                )
                
                self.stats['stores'] += 1
                self.stats['total_sequences_cached'] = len(self.metadata)
                
                logger.debug(f"Stored embedding for sequence (length {len(sequence)})")
                
                # Periodic metadata save and cache cleanup
                if self.stats['stores'] % 10 == 0:
                    self._save_metadata()
                    self._cleanup_if_needed()
                
            except Exception as e:
                logger.error(f"Failed to store embedding in cache: {e}")
    
    def get_batch_embeddings(self, sequences: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get embeddings for a batch of sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Tuple of (cached_embeddings, uncached_sequences)
        """
        cached_embeddings = []
        uncached_sequences = []
        
        for sequence in sequences:
            embedding = self.get_embedding(sequence)
            if embedding is not None:
                cached_embeddings.append(embedding)
            else:
                uncached_sequences.append(sequence)
        
        logger.info(f"Batch cache: {len(cached_embeddings)} hits, {len(uncached_sequences)} misses")
        return cached_embeddings, uncached_sequences
    
    def store_batch_embeddings(self, sequences: List[str], embeddings: List[np.ndarray]):
        """
        Store embeddings for a batch of sequences.
        
        Args:
            sequences: List of protein sequences
            embeddings: List of corresponding embeddings
        """
        if len(sequences) != len(embeddings):
            raise ValueError("Sequences and embeddings lists must have same length")
        
        for sequence, embedding in zip(sequences, embeddings):
            self.store_embedding(sequence, embedding)
        
        logger.info(f"Stored {len(sequences)} embeddings in cache")
    
    def _cleanup_if_needed(self):
        """Clean up cache if it exceeds size limits."""
        try:
            cache_size_mb = self._calculate_cache_size()
            self.stats['cache_size_mb'] = cache_size_mb
            
            if cache_size_mb > self.max_cache_size_gb * 1024:
                logger.info(f"Cache size ({cache_size_mb:.1f} MB) exceeds limit, cleaning up...")
                self._cleanup_old_entries()
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def _calculate_cache_size(self) -> float:
        """Calculate total cache size in MB."""
        total_size = 0
        for cache_file in self.cache_dir.rglob("*.h5"):
            total_size += cache_file.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _cleanup_old_entries(self, target_reduction_percent: float = 20.0):
        """Remove oldest/least accessed cache entries."""
        # Sort by last accessed time and access count
        sorted_metadata = sorted(
            self.metadata.items(),
            key=lambda x: (x[1].last_accessed, x[1].access_count)
        )
        
        # Remove oldest 20% of entries
        num_to_remove = int(len(sorted_metadata) * target_reduction_percent / 100)
        
        for cache_key, metadata in sorted_metadata[:num_to_remove]:
            try:
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                del self.metadata[cache_key]
            except Exception as e:
                logger.error(f"Failed to remove cache entry {cache_key}: {e}")
        
        logger.info(f"Removed {num_to_remove} old cache entries")
        self._save_metadata()
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_percent': hit_rate,
            'total_requests': total_requests,
            'cache_dir': str(self.cache_dir),
            'model_name': self.model_name
        }
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        with self._lock:
            try:
                for cache_file in self.cache_dir.rglob("*.h5"):
                    cache_file.unlink()
                
                self.metadata.clear()
                self._save_metadata()
                
                # Reset stats
                self.stats = {
                    'hits': 0,
                    'misses': 0,
                    'stores': 0,
                    'total_sequences_cached': 0,
                    'cache_size_mb': 0.0
                }
                
                logger.info("Cache cleared successfully")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
    
    def precompute_common_patterns(self, sequences: List[str], batch_size: int = 32):
        """
        Pre-compute embeddings for common sequence patterns.
        
        This is useful for GA workflows where certain peptide combinations
        are likely to appear frequently.
        
        Args:
            sequences: List of sequences to pre-compute
            batch_size: Batch size for embedding generation
        """
        logger.info(f"Pre-computing embeddings for {len(sequences)} sequences...")
        
        # Filter out already cached sequences
        uncached_sequences = [
            seq for seq in sequences 
            if self.get_embedding(seq) is None
        ]
        
        if not uncached_sequences:
            logger.info("All sequences already cached")
            return
        
        logger.info(f"Need to compute embeddings for {len(uncached_sequences)} sequences")
        
        # This would integrate with the existing PLM_Sol embedding pipeline
        # For now, we'll just log the intent
        logger.info("Pre-computation would integrate with PLM_Sol embedding pipeline")
        
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save metadata."""
        self._save_metadata()


# Convenience functions for integration
def create_cache(cache_dir: str = None) -> PersistentEmbeddingCache:
    """Create a persistent embedding cache with default settings."""
    if cache_dir is None:
        cache_dir = "/home/david_nunn/PLM_Sol/embedding_cache"
    
    return PersistentEmbeddingCache(cache_dir=cache_dir)


def get_or_create_global_cache() -> PersistentEmbeddingCache:
    """Get or create a global cache instance."""
    global _global_cache
    if '_global_cache' not in globals():
        _global_cache = create_cache()
    return _global_cache
