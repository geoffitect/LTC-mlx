#============================================================
# Quick test of fixes - run 100 epochs to verify stability
#============================================================

import time
import numpy as np

def quick_test_mlx():
    """Test MLX with 100 epochs to check for stability"""
    print("🧪 Testing MLX fixes (100 epochs)...")

    try:
        from ltc_mlx import LTCRNN as MLXLTCRNN, RandomWiring as MLXRandomWiring, generate_spiral_data
        import mlx.core as mx
        import mlx.nn as mlx_nn
        import mlx.optimizers as mlx_optim

        # Hyperparameters
        input_dim = 2
        hidden_dim = 8
        output_dim = 2
        seq_len = 3
        batch_size = 8  # Smaller batch for quick test
        learning_rate = 0.001  # Conservative learning rate
        num_epochs = 100

        # Generate small dataset
        data = generate_spiral_data(100, 2)  # Smaller dataset
        all_inputs = data[:-1, :]
        all_targets = data[1:, :]

        # Create simple batches
        trajectory_count = max(1, len(all_inputs) - seq_len)
        train_inputs = [all_inputs[i:i + seq_len] for i in range(0, trajectory_count, 5)]  # Every 5th
        train_targets = [all_targets[i:i + seq_len] for i in range(0, trajectory_count, 5)]

        # Initialize model
        wiring = MLXRandomWiring(input_dim, output_dim, hidden_dim)
        model = MLXLTCRNN(wiring, input_dim, hidden_dim, output_dim)

        # Loss and optimizer
        def mse_loss(predictions, targets):
            return mx.mean((predictions - targets) ** 2)

        def loss_fn(model, x, y_target):
            outputs = model(x)
            loss = mse_loss(outputs, y_target)
            # Add L2 regularization
            # Simplified L2 loss without tree_flatten
            l2_loss = 0.0
            return loss + l2_loss

        optimizer = mlx_optim.Adam(learning_rate=learning_rate)
        loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

        losses = []
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for i in range(0, len(train_inputs), batch_size):
                x_batch = train_inputs[i:i+batch_size]
                y_batch = train_targets[i:i+batch_size]

                if len(x_batch) == 0:
                    continue

                x = mx.stack([mx.array(xi) for xi in x_batch])
                y = mx.stack([mx.array(yi) for yi in y_batch])

                # Forward pass and compute gradients
                loss, grads = loss_and_grad_fn(model, x, y)

                # No gradient clipping for now - just use raw gradients
                clipped_grads = grads

                # Update model parameters
                optimizer.update(model, clipped_grads)
                mx.eval(model.parameters(), optimizer.state)

                epoch_loss += loss.item()

            losses.append(epoch_loss)

            # Check for explosion
            if epoch_loss > 10.0:
                print(f"❌ Loss explosion detected at epoch {epoch+1}: {epoch_loss:.6f}")
                return False

            if (epoch + 1) % 25 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        total_time = time.time() - start_time
        final_loss = losses[-1]

        print(f"✅ MLX test completed successfully!")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Loss stable: {final_loss < 1.0}")

        return True

    except Exception as e:
        print(f"❌ MLX test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_test_coreml():
    """Test CoreML with fixed input names"""
    print("\n🍎 Testing CoreML fixes...")

    try:
        import coremltools as ct
        import os

        if not os.path.exists("ltc_model.mlpackage"):
            print("⚠️  CoreML model not found. Run ltc_coreml.py first")
            return False

        # Load model
        model = ct.models.MLModel("ltc_model.mlpackage")

        # Check input spec
        print("  Model input spec:")
        input_descriptions = model.input_description
        print(f"    Input descriptions type: {type(input_descriptions)}")

        # Try multiple approaches to get input name
        input_name = None

        # Approach 1: Try spec property
        try:
            spec = model.get_spec()
            if hasattr(spec, 'description') and hasattr(spec.description, 'input'):
                input_name = spec.description.input[0].name
                print(f"    Found input name via spec: {input_name}")
        except:
            pass

        # Approach 2: Try predict with common names to find the right one
        if input_name is None:
            test_input = np.random.rand(1, 3, 2).astype(np.float32)
            for candidate_name in ["inputs", "input", "x", "input_1", "0"]:
                try:
                    model.predict({candidate_name: test_input})
                    input_name = candidate_name
                    print(f"    Found working input name: {input_name}")
                    break
                except:
                    continue

        # Fallback
        if input_name is None:
            input_name = "inputs"
            print(f"    Using fallback input name: {input_name}")
        test_input = np.random.rand(1, 3, 2).astype(np.float32)
        input_dict = {input_name: test_input}

        # Test inference
        start_time = time.time()
        for _ in range(100):
            result = model.predict(input_dict)
        total_time = time.time() - start_time

        print(f"✅ CoreML test completed successfully!")
        print(f"  100 inferences in {total_time:.3f}s")
        print(f"  Average: {total_time*10:.2f}ms per inference")

        return True

    except ImportError:
        print("⚠️  CoreML not installed")
        return False
    except Exception as e:
        print(f"❌ CoreML test failed: {e}")
        return False

def main():
    print("🚀 Testing Benchmark Fixes")
    print("=" * 40)

    mlx_ok = quick_test_mlx()
    coreml_ok = quick_test_coreml()

    print("\n" + "=" * 40)
    print("📋 Test Results:")
    print(f"  MLX fixes:    {'✅' if mlx_ok else '❌'}")
    print(f"  CoreML fixes: {'✅' if coreml_ok else '❌'}")

    if mlx_ok:
        print("\n🚀 Ready to run full benchmark with improved MLX!")
    else:
        print("\n⚠️  MLX issues remain - check the fixes")

if __name__ == "__main__":
    main()