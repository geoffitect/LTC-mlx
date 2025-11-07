#============================================================
# Quick test script to verify all frameworks work
# Run this before the full benchmark
#============================================================

import sys
import traceback

def test_pytorch():
    """Test PyTorch implementation"""
    try:
        from ltc_coreml import LTCRNN as PyTorchLTCRNN, RandomWiring as PyTorchRandomWiring, generate_spiral_data
        import torch

        # Simple test
        wiring = PyTorchRandomWiring(2, 2, 8)
        model = PyTorchLTCRNN(wiring, 2, 8, 2)
        test_input = torch.rand(1, 3, 2)
        output = model(test_input)

        print("✅ PyTorch: Working! Output shape:", output.shape)
        return True
    except Exception as e:
        print(f"❌ PyTorch: Failed - {e}")
        traceback.print_exc()
        return False

def test_mlx():
    """Test MLX implementation"""
    try:
        from ltc_mlx import LTCRNN as MLXLTCRNN, RandomWiring as MLXRandomWiring
        import mlx.core as mx

        # Simple test
        wiring = MLXRandomWiring(2, 2, 8)
        model = MLXLTCRNN(wiring, 2, 8, 2)
        test_input = mx.random.normal((1, 3, 2))
        output = model(test_input)

        print("✅ MLX: Working! Output shape:", output.shape)
        return True
    except Exception as e:
        print(f"❌ MLX: Failed - {e}")
        traceback.print_exc()
        return False

def test_coreml():
    """Test CoreML availability"""
    try:
        import coremltools as ct
        import os

        if os.path.exists("ltc_model.mlpackage"):
            model = ct.models.MLModel("ltc_model.mlpackage")
            print("✅ CoreML: Working! Model loaded successfully")
            return True
        else:
            print("⚠️  CoreML: Model not found. Run ltc_coreml.py first")
            return False
    except ImportError:
        print("❌ CoreML: Not installed. Run: pip install coremltools")
        return False
    except Exception as e:
        print(f"❌ CoreML: Failed - {e}")
        return False

def main():
    print("🧪 Quick Framework Test")
    print("=" * 30)

    pytorch_ok = test_pytorch()
    mlx_ok = test_mlx()
    coreml_ok = test_coreml()

    print("\n" + "=" * 30)
    print("📋 Test Summary:")
    print(f"PyTorch: {'✅' if pytorch_ok else '❌'}")
    print(f"MLX:     {'✅' if mlx_ok else '❌'}")
    print(f"CoreML:  {'✅' if coreml_ok else '❌'}")

    if pytorch_ok and mlx_ok:
        print("\n🚀 Ready to run benchmark_comparison.py!")
    else:
        print("\n⚠️  Fix issues above before running benchmark")

if __name__ == "__main__":
    main()