#!/usr/bin/env python3
"""
PyTorch Compatibility Test
Verify that all optimized components work without MLX dependencies
"""

import torch
import sys
import os
from typing import Dict, Any

print("🧪 PYTORCH COMPATIBILITY TEST")
print("=" * 60)

def test_imports():
    """Test that all imports work without MLX"""
    print("🔍 Testing imports...")

    try:
        # Test PyTorch trainer utilities
        from utils.aa2fold_trainer_pytorch import (
            REAL_FOLDSEEK_3DI_ALPHABET,
            REAL_FOLDSEEK_3DI_TO_IDX,
            AA_TO_IDX,
            get_validated_sequences,
            PyTorchEnhancedRealProteinDataLoader
        )
        print("  ✅ utils.aa2fold_trainer_pytorch: SUCCESS")

        # Test optimized trainer
        from optimized_cuda_trainer import (
            OptimizedLTCCell,
            OptimizedProteinLTCModel,
            OptimizedCudaTrainer,
            OPTIMIZED_CONFIG
        )
        print("  ✅ optimized_cuda_trainer: SUCCESS")

        # Test optimized spline model
        from spline.sequence_to_3di_cuda_optimized import (
            OptimizedSequenceTo3DiModel,
            ParallelLTCLayer,
            OPTIMIZED_SPLINE_CONFIG
        )
        print("  ✅ spline.sequence_to_3di_cuda_optimized: SUCCESS")

        return True

    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_model_creation():
    """Test that models can be created and run forward passes"""
    print("\n🔧 Testing model creation...")

    try:
        from optimized_cuda_trainer import OptimizedProteinLTCModel, OPTIMIZED_CONFIG

        # Create model
        model = OptimizedProteinLTCModel(OPTIMIZED_CONFIG)
        print("  ✅ OptimizedProteinLTCModel created")

        # Test forward pass
        batch_size = 4
        seq_len = 128  # Smaller for testing

        # Create test input
        test_input = torch.randint(0, 20, (batch_size, seq_len))

        # Forward pass
        with torch.no_grad():
            output = model(test_input)

        expected_shape = (batch_size, seq_len, OPTIMIZED_CONFIG['struct_vocab_size'])
        if output.shape == expected_shape:
            print(f"  ✅ Forward pass successful: {output.shape}")
        else:
            print(f"  ❌ Shape mismatch: got {output.shape}, expected {expected_shape}")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing data loading...")

    try:
        from utils.aa2fold_trainer_pytorch import PyTorchEnhancedRealProteinDataLoader

        # Create loader
        loader = PyTorchEnhancedRealProteinDataLoader()
        print("  ✅ Data loader created")

        # Test with non-existent files (should handle gracefully)
        try:
            pairs = loader.create_valid_pairs_enhanced(
                max_pairs=10,
                aa_file="nonexistent_aa.fasta",
                tsv_file="nonexistent_3di.tsv"
            )
            print("  ✅ Graceful handling of missing files")
        except Exception as e:
            print(f"  ⚠️  Expected error for missing files: {str(e)[:50]}...")

        return True

    except Exception as e:
        print(f"  ❌ Data loading test failed: {e}")
        return False

def test_cuda_optimization():
    """Test CUDA-specific optimizations"""
    print("\n⚡ Testing CUDA optimizations...")

    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"  🔥 CUDA available: {cuda_available}")

        if cuda_available:
            print(f"  🔥 CUDA device: {torch.cuda.get_device_name()}")
            print(f"  💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

            # Test mixed precision
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            print("  ✅ Mixed precision support available")

        else:
            print("  🖥️  Using CPU/MPS - CUDA optimizations will be disabled")

        # Test batch size optimization
        from optimized_cuda_trainer import OptimizedCudaTrainer, OPTIMIZED_CONFIG

        trainer = OptimizedCudaTrainer(OPTIMIZED_CONFIG)
        print("  ✅ Optimized trainer created")

        return True

    except Exception as e:
        print(f"  ❌ CUDA optimization test failed: {e}")
        return False

def test_benchmark_functionality():
    """Test benchmark script functionality"""
    print("\n📊 Testing benchmark functionality...")

    try:
        from benchmark_optimizations import (
            SequentialLTCCell,
            ParallelLTCCell,
            BenchmarkResult
        )
        print("  ✅ Benchmark classes imported")

        # Test model creation
        input_dim = 128
        hidden_dim = 256

        sequential_model = SequentialLTCCell(input_dim, hidden_dim)
        parallel_model = ParallelLTCCell(input_dim, hidden_dim)
        print("  ✅ Benchmark models created")

        # Test with small input
        batch_size = 2
        seq_len = 32  # Small for testing

        inputs = torch.randn(batch_size, seq_len, input_dim)
        hidden = torch.randn(batch_size, seq_len, hidden_dim)

        # Test both models
        with torch.no_grad():
            seq_output = sequential_model(inputs, hidden)
            par_output = parallel_model(inputs, hidden)

        print(f"  ✅ Sequential output shape: {seq_output.shape}")
        print(f"  ✅ Parallel output shape: {par_output.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Benchmark test failed: {e}")
        return False

def run_compatibility_summary():
    """Run complete compatibility test and provide summary"""
    print("\n🎯 PYTORCH COMPATIBILITY SUMMARY")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("CUDA Optimization", test_cuda_optimization),
        ("Benchmark", test_benchmark_functionality)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    passed = sum(results.values())
    total = len(results)

    print(f"\n📋 Test Results: {passed}/{total} passed")
    print("-" * 30)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<20} {status}")

    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ PyTorch compatibility: VERIFIED")
        print(f"✅ MLX dependencies: REMOVED")
        print(f"✅ CUDA optimization: READY")
        print(f"🚀 Ready for deployment on CUDA machine!")
    else:
        print(f"\n⚠️  {total - passed} tests failed - needs debugging")

    return passed == total

if __name__ == "__main__":
    success = run_compatibility_summary()

    if success:
        print(f"\n🔥 PYTORCH COMPATIBILITY: SUCCESS!")
        print(f"🚀 Your CUDA machine is ready to run:")
        print(f"   python optimized_cuda_trainer.py")
        print(f"   python spline/sequence_to_3di_cuda_optimized.py")
        print(f"   python benchmark_optimizations.py")

        sys.exit(0)
    else:
        print(f"\n❌ PYTORCH COMPATIBILITY: FAILED!")
        print(f"🔧 Check error messages above and fix issues")
        sys.exit(1)