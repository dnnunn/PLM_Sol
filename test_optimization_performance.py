#!/usr/bin/env python3
"""
Performance Test for Optimized PLM_Sol Predictor

This script tests the performance improvements from the persistent embedding cache
and other optimizations, specifically for DEAP experimentation scenarios.

Test Scenarios:
1. Cold start (no cache) vs. warm cache performance
2. Batch processing efficiency
3. Duplicate sequence handling (common in GA)
4. Memory and disk usage
5. Cache hit rates over time

Expected Results:
- 10-50x speedup for cached sequences
- Near-instant prediction for duplicates
- Efficient batch processing
"""

import time
import logging
import random
import statistics
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import our optimized predictor
from optimized_plm_sol_predictor import OptimizedPLMSolPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizationTester:
    """Test suite for PLM_Sol optimization performance."""
    
    def __init__(self):
        self.predictor = None
        self.test_sequences = []
        self.results = {}
        
    def setup_test_sequences(self):
        """Generate test sequences that simulate GA/DEAP scenarios."""
        logger.info("Setting up test sequences...")
        
        # Common bioactive peptides for fusion proteins
        peptides = [
            "VPPIPP",  # VPP
            "IPPIPP",  # IPP  
            "HLPLP",   # HLPLP
            "RPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL",  # Longer peptide
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"  # Full sequence
        ]
        
        # Generate fusion-like sequences of varying lengths
        self.test_sequences = []
        
        # Short sequences (100-200 AA) - common in early GA generations
        for i in range(20):
            seq = self._generate_fusion_sequence(peptides, target_length=150)
            self.test_sequences.append(seq)
        
        # Medium sequences (200-300 AA) - typical GA targets
        for i in range(30):
            seq = self._generate_fusion_sequence(peptides, target_length=250)
            self.test_sequences.append(seq)
        
        # Long sequences (300-400 AA) - final GA products
        for i in range(20):
            seq = self._generate_fusion_sequence(peptides, target_length=350)
            self.test_sequences.append(seq)
        
        # Add some duplicates (common in GA populations)
        duplicates = random.choices(self.test_sequences, k=30)
        self.test_sequences.extend(duplicates)
        
        logger.info(f"Generated {len(self.test_sequences)} test sequences")
        logger.info(f"Unique sequences: {len(set(self.test_sequences))}")
        
    def _generate_fusion_sequence(self, peptides: List[str], target_length: int) -> str:
        """Generate a fusion protein sequence."""
        sequence = "M"  # Start with methionine
        
        while len(sequence) < target_length:
            # Add peptide
            peptide = random.choice(peptides)
            
            # Add linker (3 AA)
            linker = "".join(random.choices("ADEFGHIKLMNQRSTVWY", k=3))
            
            # Check if we can add both
            if len(sequence) + len(linker) + len(peptide) > target_length:
                break
                
            sequence += linker + peptide
        
        return sequence
    
    def test_cold_vs_warm_performance(self):
        """Test performance difference between cold start and warm cache."""
        logger.info("Testing cold vs warm cache performance...")
        
        # Initialize predictor with fresh cache
        self.predictor = OptimizedPLMSolPredictor()
        self.predictor.clear_caches()
        
        # Test subset for timing
        test_subset = self.test_sequences[:10]
        
        # Cold start test
        logger.info("Running cold start test...")
        start_time = time.time()
        cold_results = self.predictor.predict_batch(test_subset)
        cold_time = time.time() - start_time
        
        # Warm cache test (same sequences)
        logger.info("Running warm cache test...")
        start_time = time.time()
        warm_results = self.predictor.predict_batch(test_subset)
        warm_time = time.time() - start_time
        
        # Calculate speedup
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        
        self.results['cold_vs_warm'] = {
            'cold_time': cold_time,
            'warm_time': warm_time,
            'speedup': speedup,
            'sequences_tested': len(test_subset)
        }
        
        logger.info(f"Cold start: {cold_time:.2f}s ({cold_time/len(test_subset):.3f}s per sequence)")
        logger.info(f"Warm cache: {warm_time:.2f}s ({warm_time/len(test_subset):.3f}s per sequence)")
        logger.info(f"Speedup: {speedup:.1f}x")
        
    def test_batch_efficiency(self):
        """Test batch processing efficiency with different batch sizes."""
        logger.info("Testing batch processing efficiency...")
        
        batch_sizes = [1, 5, 10, 20, 50]
        batch_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Use fresh sequences for each test
            test_sequences = self.test_sequences[:batch_size]
            
            start_time = time.time()
            results = self.predictor.predict_batch(test_sequences)
            batch_time = time.time() - start_time
            
            time_per_sequence = batch_time / batch_size if batch_size > 0 else 0
            
            batch_results[batch_size] = {
                'total_time': batch_time,
                'time_per_sequence': time_per_sequence,
                'sequences': batch_size
            }
            
            logger.info(f"Batch {batch_size}: {batch_time:.2f}s total, {time_per_sequence:.3f}s per sequence")
        
        self.results['batch_efficiency'] = batch_results
        
    def test_duplicate_handling(self):
        """Test performance with high duplicate rates (GA scenario)."""
        logger.info("Testing duplicate sequence handling...")
        
        # Create test with many duplicates
        unique_sequences = self.test_sequences[:10]
        
        # Create batches with increasing duplicate rates
        duplicate_rates = [0.0, 0.2, 0.5, 0.8, 0.9]
        duplicate_results = {}
        
        for dup_rate in duplicate_rates:
            logger.info(f"Testing duplicate rate: {dup_rate:.1%}")
            
            # Create batch with specified duplicate rate
            batch_size = 50
            num_unique = int(batch_size * (1 - dup_rate))
            num_duplicates = batch_size - num_unique
            
            test_batch = unique_sequences[:num_unique]
            
            # Add duplicates
            if num_duplicates > 0:
                duplicates = random.choices(test_batch, k=num_duplicates)
                test_batch.extend(duplicates)
            
            # Shuffle to simulate realistic GA population
            random.shuffle(test_batch)
            
            start_time = time.time()
            results = self.predictor.predict_batch(test_batch)
            batch_time = time.time() - start_time
            
            duplicate_results[dup_rate] = {
                'total_time': batch_time,
                'time_per_sequence': batch_time / batch_size,
                'unique_sequences': num_unique,
                'duplicate_sequences': num_duplicates,
                'batch_size': batch_size
            }
            
            logger.info(f"Duplicate rate {dup_rate:.1%}: {batch_time:.2f}s for {batch_size} sequences")
        
        self.results['duplicate_handling'] = duplicate_results
        
    def test_cache_performance_over_time(self):
        """Test cache hit rates and performance over multiple generations."""
        logger.info("Testing cache performance over time...")
        
        generation_results = []
        
        # Simulate 10 GA generations
        for generation in range(10):
            logger.info(f"Simulating generation {generation + 1}")
            
            # Each generation has some new sequences and some from previous generations
            if generation == 0:
                # First generation - all new
                gen_sequences = self.test_sequences[:50]
            else:
                # Mix of new and previous sequences
                new_sequences = self.test_sequences[generation*20:(generation+1)*20]
                previous_sequences = random.choices(
                    self.test_sequences[:generation*20], 
                    k=30
                )
                gen_sequences = new_sequences + previous_sequences
                random.shuffle(gen_sequences)
            
            start_time = time.time()
            results = self.predictor.predict_batch(gen_sequences)
            gen_time = time.time() - start_time
            
            # Get cache statistics
            stats = self.predictor.get_performance_stats()
            
            generation_results.append({
                'generation': generation + 1,
                'time': gen_time,
                'sequences': len(gen_sequences),
                'cache_hit_rate': stats['predictor_stats']['cache_hit_rate_percent'],
                'total_time_saved': stats['predictor_stats']['time_saved_minutes']
            })
            
            logger.info(f"Generation {generation + 1}: {gen_time:.2f}s, "
                       f"cache hit rate: {stats['predictor_stats']['cache_hit_rate_percent']:.1f}%")
        
        self.results['cache_over_time'] = generation_results
        
    def test_memory_usage(self):
        """Test memory and disk usage of the cache system."""
        logger.info("Testing memory and disk usage...")
        
        # Get initial cache stats
        initial_stats = self.predictor.get_performance_stats()
        
        # Process a large batch to build up cache
        large_batch = self.test_sequences
        self.predictor.predict_batch(large_batch)
        
        # Get final cache stats
        final_stats = self.predictor.get_performance_stats()
        
        self.results['memory_usage'] = {
            'initial_cache_size_mb': initial_stats['embedding_cache_stats']['cache_size_mb'],
            'final_cache_size_mb': final_stats['embedding_cache_stats']['cache_size_mb'],
            'sequences_cached': final_stats['embedding_cache_stats']['total_sequences_cached'],
            'cache_hit_rate': final_stats['predictor_stats']['cache_hit_rate_percent']
        }
        
        logger.info(f"Cache size: {final_stats['embedding_cache_stats']['cache_size_mb']:.1f} MB")
        logger.info(f"Sequences cached: {final_stats['embedding_cache_stats']['total_sequences_cached']}")
        
    def generate_performance_report(self):
        """Generate a comprehensive performance report."""
        logger.info("Generating performance report...")
        
        report = {
            'test_summary': {
                'total_sequences_tested': len(self.test_sequences),
                'unique_sequences': len(set(self.test_sequences)),
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'performance_results': self.results,
            'final_predictor_stats': self.predictor.get_performance_stats() if self.predictor else {}
        }
        
        # Save report
        import json
        report_file = "/tmp/plm_sol_optimization_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to: {report_file}")
        
        # Print summary
        self._print_summary()
        
        return report
    
    def _print_summary(self):
        """Print a summary of test results."""
        print("\n" + "="*60)
        print("PLM_SOL OPTIMIZATION PERFORMANCE SUMMARY")
        print("="*60)
        
        if 'cold_vs_warm' in self.results:
            cvw = self.results['cold_vs_warm']
            print(f"Cold vs Warm Cache:")
            print(f"  Speedup: {cvw['speedup']:.1f}x")
            print(f"  Cold: {cvw['cold_time']:.2f}s, Warm: {cvw['warm_time']:.2f}s")
        
        if 'batch_efficiency' in self.results:
            be = self.results['batch_efficiency']
            best_batch = min(be.items(), key=lambda x: x[1]['time_per_sequence'])
            print(f"Batch Efficiency:")
            print(f"  Optimal batch size: {best_batch[0]} sequences")
            print(f"  Best time per sequence: {best_batch[1]['time_per_sequence']:.3f}s")
        
        if 'duplicate_handling' in self.results:
            dh = self.results['duplicate_handling']
            high_dup = dh.get(0.9, {})
            low_dup = dh.get(0.0, {})
            if high_dup and low_dup:
                speedup = low_dup['total_time'] / high_dup['total_time']
                print(f"Duplicate Handling:")
                print(f"  90% duplicates vs 0% duplicates speedup: {speedup:.1f}x")
        
        if 'cache_over_time' in self.results:
            cot = self.results['cache_over_time']
            final_gen = cot[-1] if cot else {}
            print(f"Cache Performance Over Time:")
            print(f"  Final cache hit rate: {final_gen.get('cache_hit_rate', 0):.1f}%")
            print(f"  Total time saved: {final_gen.get('total_time_saved', 0):.1f} minutes")
        
        if self.predictor:
            final_stats = self.predictor.get_performance_stats()
            print(f"Final Statistics:")
            print(f"  Total predictions: {final_stats['predictor_stats']['total_predictions']}")
            print(f"  Cache hit rate: {final_stats['predictor_stats']['cache_hit_rate_percent']:.1f}%")
            print(f"  Cache size: {final_stats['embedding_cache_stats']['cache_size_mb']:.1f} MB")
        
        print("="*60)
    
    def run_all_tests(self):
        """Run all performance tests."""
        logger.info("Starting comprehensive optimization performance tests...")
        
        try:
            self.setup_test_sequences()
            self.test_cold_vs_warm_performance()
            self.test_batch_efficiency()
            self.test_duplicate_handling()
            self.test_cache_performance_over_time()
            self.test_memory_usage()
            
            return self.generate_performance_report()
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise


def main():
    """Run the optimization performance tests."""
    tester = OptimizationTester()
    report = tester.run_all_tests()
    
    print("\nOptimization testing complete!")
    print("Check the performance report for detailed results.")
    
    return report


if __name__ == "__main__":
    main()
