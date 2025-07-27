#!/usr/bin/env python3
"""
Simple Integration Test for PLM_Sol Optimizations

This script provides a quick test to verify that the optimization components
are working correctly before running the full performance test suite.

Tests:
1. Cache initialization
2. Basic prediction functionality
3. Cache hit/miss behavior
4. Integration with PeptideFrontEnd interface
"""

import sys
import os
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cache_initialization():
    """Test that the persistent embedding cache initializes correctly."""
    logger.info("Testing cache initialization...")
    
    try:
        from persistent_embedding_cache import PersistentEmbeddingCache
        
        # Create cache with test directory
        test_cache_dir = "/tmp/plm_sol_test_cache"
        cache = PersistentEmbeddingCache(cache_dir=test_cache_dir)
        
        # Test basic cache operations
        test_sequence = "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
        
        # Should be cache miss initially
        embedding = cache.get_embedding(test_sequence)
        assert embedding is None, "Expected cache miss for new sequence"
        
        # Store a dummy embedding
        import numpy as np
        dummy_embedding = np.random.rand(100, 1024)  # Typical T5 embedding shape
        cache.store_embedding(test_sequence, dummy_embedding)
        
        # Should be cache hit now
        retrieved_embedding = cache.get_embedding(test_sequence)
        assert retrieved_embedding is not None, "Expected cache hit after storing"
        assert np.array_equal(dummy_embedding, retrieved_embedding), "Retrieved embedding should match stored"
        
        # Test cache stats
        stats = cache.get_cache_stats()
        assert stats['hits'] == 1, f"Expected 1 hit, got {stats['hits']}"
        assert stats['misses'] == 1, f"Expected 1 miss, got {stats['misses']}"
        
        logger.info("✅ Cache initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache initialization test failed: {e}")
        return False

def test_optimized_predictor():
    """Test that the optimized predictor can be instantiated."""
    logger.info("Testing optimized predictor initialization...")
    
    try:
        from optimized_plm_sol_predictor import OptimizedPLMSolPredictor
        
        # Create predictor with test settings
        predictor = OptimizedPLMSolPredictor(
            cache_dir="/tmp/plm_sol_test_cache",
            conda_env="PLM_Sol",  # This might not exist in test environment
            model_path="/home/david_nunn/PLM_Sol/saved_models/model-10.t7"  # This might not exist
        )
        
        # Test basic functionality (without actually running PLM_Sol)
        stats = predictor.get_performance_stats()
        assert 'predictor_stats' in stats, "Expected predictor stats"
        assert 'embedding_cache_stats' in stats, "Expected cache stats"
        
        logger.info("✅ Optimized predictor test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimized predictor test failed: {e}")
        return False

def test_frontend_integration():
    """Test integration with PeptideFrontEnd interface."""
    logger.info("Testing PeptideFrontEnd integration...")
    
    try:
        # Add PeptideFrontEnd path
        frontend_path = "/Users/davidnunn/Desktop/Apps/PeptideFusionProject/PeptideFrontEnd"
        if frontend_path not in sys.path:
            sys.path.insert(0, frontend_path)
        
        from genetic_algorithm.plm_sol_predictor_optimized import OptimizedPLMSolPredictorFactory
        
        # Create predictor using factory
        predictor = OptimizedPLMSolPredictorFactory.create_predictor("testing")
        
        # Test that it has the expected interface
        assert hasattr(predictor, 'predict_batch'), "Expected predict_batch method"
        assert hasattr(predictor, 'predict_single'), "Expected predict_single method"
        assert hasattr(predictor, 'get_performance_stats'), "Expected get_performance_stats method"
        
        # Test with fallback mode (since PLM_Sol environment might not be available)
        test_sequences = [
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL",
            "MKALIVLGLVLLGAALERPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
        ]
        
        # This should work even in fallback mode
        results = predictor.predict_batch(test_sequences)
        
        assert len(results) == len(test_sequences), "Expected result for each sequence"
        
        for result in results:
            assert 'sequence' in result, "Expected sequence in result"
            assert 'solubility' in result, "Expected solubility in result"
            assert 'prediction' in result, "Expected prediction in result"
            assert 'source' in result, "Expected source in result"
        
        logger.info("✅ PeptideFrontEnd integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ PeptideFrontEnd integration test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    logger.info("Testing performance monitoring...")
    
    try:
        from genetic_algorithm.plm_sol_predictor_optimized import PerformanceMonitor, OptimizedPLMSolPredictorFactory
        
        predictor = OptimizedPLMSolPredictorFactory.create_predictor("testing")
        monitor = PerformanceMonitor(predictor)
        
        # Simulate some batch predictions
        monitor.record_batch_prediction(batch_size=10, prediction_time=5.0)
        monitor.record_batch_prediction(batch_size=20, prediction_time=8.0)
        
        # Get session summary
        summary = monitor.get_session_summary()
        
        assert summary['total_batches'] == 2, "Expected 2 batches recorded"
        assert summary['total_sequences'] == 30, "Expected 30 total sequences"
        
        logger.info("✅ Performance monitoring test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests."""
    logger.info("Starting PLM_Sol optimization integration tests...")
    
    tests = [
        ("Cache Initialization", test_cache_initialization),
        ("Optimized Predictor", test_optimized_predictor),
        ("PeptideFrontEnd Integration", test_frontend_integration),
        ("Performance Monitoring", test_performance_monitoring)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if passed_test:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Optimizations are ready for use.")
    else:
        logger.warning("⚠️  Some tests failed. Check logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
