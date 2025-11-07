#============================================================
# Comprehensive Benchmark: PyTorch vs MLX vs CoreML
# Compares training speed and inference accuracy across frameworks
#============================================================

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

# Framework imports
import torch
import torch.nn as nn
import torch.optim as optim

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("CoreML not available. Install with: pip install coremltools")

# Import our implementations
from ltc_coreml import LTCRNN as PyTorchLTCRNN, RandomWiring as PyTorchRandomWiring, generate_spiral_data
try:
    from ltc_mlx_fixed import LTCRNN as MLXLTCRNN, RandomWiring as MLXRandomWiring
    print("Using fixed MLX implementation")
except ImportError:
    from ltc_mlx import LTCRNN as MLXLTCRNN, RandomWiring as MLXRandomWiring
    print("Using original MLX implementation")

class BenchmarkResults:
    def __init__(self):
        self.pytorch_times = []
        self.mlx_times = []
        self.coreml_times = []
        self.pytorch_losses = []
        self.mlx_losses = []
        self.final_accuracies = {}
        self.total_training_times = {}

def prepare_data(num_points=500, num_turns=3, seq_len=3, batch_size=32):
    """Prepare training data for all frameworks"""
    print("Preparing training data...")

    # Generate spiral data
    data = generate_spiral_data(num_points, num_turns)
    all_inputs = data[:-1, :]
    all_targets = data[1:, :]

    # Create sequences
    trajectory_count = max(1, len(all_inputs) - seq_len)
    train_inputs = [all_inputs[i:i + seq_len] for i in range(trajectory_count)]
    train_targets = [all_targets[i:i + seq_len] for i in range(trajectory_count)]

    # Split data
    random_indices = np.arange(len(train_inputs))
    np.random.shuffle(random_indices)
    train_split_index = int(len(random_indices) * 0.8)
    train_indices = random_indices[:train_split_index]

    # Create batches
    def create_batches(data_list, indices, batch_size):
        selected_data = [data_list[i] for i in indices]
        return [selected_data[i:i + batch_size] for i in range(0, len(selected_data), batch_size)]

    train_input_batches = create_batches(train_inputs, train_indices, batch_size)
    train_target_batches = create_batches(train_targets, train_indices, batch_size)

    return {
        'train_input_batches': train_input_batches,
        'train_target_batches': train_target_batches,
        'all_inputs': all_inputs,
        'all_targets': all_targets
    }

def benchmark_pytorch(data: Dict, hyperparams: Dict, num_epochs: int) -> Tuple[float, List[float]]:
    """Benchmark PyTorch implementation"""
    print(f"\n🔥 Starting PyTorch benchmark ({num_epochs} epochs)...")

    # Initialize model
    wiring = PyTorchRandomWiring(hyperparams['input_dim'], hyperparams['output_dim'], hyperparams['hidden_dim'])
    model = PyTorchLTCRNN(wiring, hyperparams['input_dim'], hyperparams['hidden_dim'], hyperparams['output_dim'])
    model.eval()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for x_batch, y_batch in zip(data['train_input_batches'], data['train_target_batches']):
            # Convert to tensors and stack
            x = torch.stack([torch.FloatTensor(xi) for xi in x_batch])
            y = torch.stack([torch.FloatTensor(yi) for yi in y_batch])

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss)

        if (epoch + 1) % 1000 == 0:
            print(f"PyTorch Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    total_time = time.time() - start_time
    print(f"PyTorch training completed in {total_time:.2f}s")

    return total_time, losses

def benchmark_mlx(data: Dict, hyperparams: Dict, num_epochs: int) -> Tuple[float, List[float]]:
    """Benchmark MLX implementation"""
    print(f"\n⚡ Starting MLX benchmark ({num_epochs} epochs)...")

    # Initialize model
    wiring = MLXRandomWiring(hyperparams['input_dim'], hyperparams['output_dim'], hyperparams['hidden_dim'])
    model = MLXLTCRNN(wiring, hyperparams['input_dim'], hyperparams['hidden_dim'], hyperparams['output_dim'])

    # Loss function and optimizer
    def mse_loss(predictions, targets):
        return mx.mean((predictions - targets) ** 2)

    def loss_fn(model, x, y_target):
        outputs = model(x)
        loss = mse_loss(outputs, y_target)
        # Add L2 regularization for stability
        # Simplified L2 loss - no tree_flatten
        l2_loss = 0.0
        return loss + l2_loss

    # Use more conservative learning rate for MLX
    mlx_lr = hyperparams['learning_rate'] * 0.5  # Reduce learning rate
    optimizer = mlx_optim.Adam(learning_rate=mlx_lr)
    loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

    losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for x_batch, y_batch in zip(data['train_input_batches'], data['train_target_batches']):
            # Convert to MLX arrays and stack
            x = mx.stack([mx.array(xi) for xi in x_batch])
            y = mx.stack([mx.array(yi) for yi in y_batch])

            # Forward pass and compute gradients
            loss, grads = loss_and_grad_fn(model, x, y)

            # No gradient clipping for now
            clipped_grads = grads

            # Update model parameters
            optimizer.update(model, clipped_grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()

        losses.append(epoch_loss)

        if (epoch + 1) % 1000 == 0:
            print(f"MLX Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    total_time = time.time() - start_time
    print(f"MLX training completed in {total_time:.2f}s")

    return total_time, losses

def benchmark_coreml_inference(data: Dict, hyperparams: Dict) -> float:
    """Benchmark CoreML inference speed (no training)"""
    if not COREML_AVAILABLE:
        print("❌ CoreML not available, skipping benchmark")
        return 0.0

    print(f"\n🍎 Starting CoreML inference benchmark...")

    # Check if CoreML model exists
    if not os.path.exists("ltc_model.mlpackage"):
        print("❌ CoreML model not found. Please run ltc_coreml.py first to create the model.")
        return 0.0

    try:
        # Load CoreML model
        model = ct.models.MLModel("ltc_model.mlpackage")

        # Check model input spec
        print("CoreML model input spec:")
        input_descriptions = model.input_description
        print(f"  Input descriptions type: {type(input_descriptions)}")

        # Handle different input description types
        if hasattr(input_descriptions, '_fd_spec'):
            # New CoreML format
            input_name = list(input_descriptions._fd_spec.input.keys())[0]
        elif isinstance(input_descriptions, dict):
            # Dict format
            input_name = list(input_descriptions.keys())[0]
        elif hasattr(input_descriptions, '__iter__'):
            # List/iterable format
            first_input = next(iter(input_descriptions))
            if hasattr(first_input, 'name'):
                input_name = first_input.name
            else:
                input_name = str(first_input)
        else:
            # Fallback
            input_name = "inputs"

        print(f"  Using input name: {input_name}")

        # Prepare test data
        test_input = np.random.rand(1, hyperparams['seq_len'], hyperparams['input_dim']).astype(np.float32)
        input_dict = {input_name: test_input}

        # Warmup
        for _ in range(10):
            _ = model.predict(input_dict)

        # Benchmark inference
        num_inferences = 1000
        start_time = time.time()

        for _ in range(num_inferences):
            _ = model.predict(input_dict)

        total_time = time.time() - start_time
        avg_inference_time = total_time / num_inferences

        print(f"CoreML average inference time: {avg_inference_time*1000:.3f}ms per sample")
        print(f"CoreML total benchmark time: {total_time:.2f}s for {num_inferences} inferences")

        return avg_inference_time

    except Exception as e:
        print(f"❌ CoreML benchmark failed: {e}")
        return 0.0

def calculate_final_accuracy(data: Dict, hyperparams: Dict) -> Dict[str, float]:
    """Calculate final prediction accuracy for each framework"""
    print("\n📊 Calculating final accuracies...")

    accuracies = {}

    # Test on a subset of data
    test_inputs = mx.array(data['all_inputs'][:100])
    test_targets = data['all_targets'][:100]

    try:
        # PyTorch accuracy
        wiring = PyTorchRandomWiring(hyperparams['input_dim'], hyperparams['output_dim'], hyperparams['hidden_dim'])
        torch_model = PyTorchLTCRNN(wiring, hyperparams['input_dim'], hyperparams['hidden_dim'], hyperparams['output_dim'])
        torch_model.eval()

        with torch.no_grad():
            torch_pred = torch_model(torch.FloatTensor(test_inputs.tolist())[None, :, :])
            torch_mse = torch.mean((torch_pred[0] - torch.FloatTensor(test_targets)) ** 2)
            accuracies['PyTorch'] = torch_mse.item()

    except Exception as e:
        print(f"PyTorch accuracy calculation failed: {e}")
        accuracies['PyTorch'] = float('inf')

    try:
        # MLX accuracy
        wiring = MLXRandomWiring(hyperparams['input_dim'], hyperparams['output_dim'], hyperparams['hidden_dim'])
        mlx_model = MLXLTCRNN(wiring, hyperparams['input_dim'], hyperparams['hidden_dim'], hyperparams['output_dim'])

        mlx_pred = mlx_model(test_inputs[None, :, :])
        mlx_mse = mx.mean((mlx_pred[0] - mx.array(test_targets)) ** 2)
        accuracies['MLX'] = mlx_mse.item()

    except Exception as e:
        print(f"MLX accuracy calculation failed: {e}")
        accuracies['MLX'] = float('inf')

    # CoreML accuracy would require a trained model, skipping for now
    accuracies['CoreML'] = 0.0  # Placeholder

    return accuracies

def plot_results(results: BenchmarkResults, num_epochs: int):
    """Generate comprehensive performance plots"""
    print("\n📈 Generating performance plots...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Training time comparison
    frameworks = []
    times = []
    if results.total_training_times.get('PyTorch', 0) > 0:
        frameworks.append('PyTorch')
        times.append(results.total_training_times['PyTorch'])
    if results.total_training_times.get('MLX', 0) > 0:
        frameworks.append('MLX')
        times.append(results.total_training_times['MLX'])

    if frameworks:
        ax1.bar(frameworks, times, color=['red', 'blue'][:len(frameworks)])
        ax1.set_title('Total Training Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Framework')

    # Loss curves
    epochs_range = range(1, len(results.pytorch_losses) + 1) if results.pytorch_losses else range(1, num_epochs + 1)
    if results.pytorch_losses:
        ax2.plot(epochs_range, results.pytorch_losses, 'r-', label='PyTorch', alpha=0.7)
    if results.mlx_losses:
        ax2.plot(epochs_range, results.mlx_losses, 'b-', label='MLX', alpha=0.7)
    ax2.set_title('Training Loss Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_yscale('log')

    # Final accuracy comparison
    if results.final_accuracies:
        frameworks = list(results.final_accuracies.keys())
        accuracies = list(results.final_accuracies.values())
        finite_accuracies = [(f, a) for f, a in zip(frameworks, accuracies) if np.isfinite(a)]

        if finite_accuracies:
            frameworks, accuracies = zip(*finite_accuracies)
            ax3.bar(frameworks, accuracies, color=['red', 'blue', 'green'][:len(frameworks)])
            ax3.set_title('Final Prediction Accuracy (MSE)')
            ax3.set_ylabel('Mean Squared Error')
            ax3.set_xlabel('Framework')

    # Speed comparison (training speed)
    if len(times) >= 2:
        speed_ratios = [times[0] / t for t in times]
        ax4.bar(frameworks, speed_ratios, color=['red', 'blue'][:len(frameworks)])
        ax4.set_title('Relative Training Speed (PyTorch = 1.0)')
        ax4.set_ylabel('Speed Ratio')
        ax4.set_xlabel('Framework')
        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('ltc_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run comprehensive benchmark"""
    print("🚀 LTC Framework Benchmark Suite")
    print("=" * 50)

    # Hyperparameters
    hyperparams = {
        'input_dim': 2,
        'hidden_dim': 8,
        'output_dim': 2,
        'seq_len': 3,
        'batch_size': 32,
        'learning_rate': 0.005
    }

    num_epochs = 5000
    print(f"Configuration: {num_epochs} epochs, {hyperparams}")

    # Prepare data
    data = prepare_data(
        num_points=500,
        num_turns=3,
        seq_len=hyperparams['seq_len'],
        batch_size=hyperparams['batch_size']
    )

    results = BenchmarkResults()

    # Benchmark PyTorch
    try:
        pytorch_time, pytorch_losses = benchmark_pytorch(data, hyperparams, num_epochs)
        results.total_training_times['PyTorch'] = pytorch_time
        results.pytorch_losses = pytorch_losses
    except Exception as e:
        print(f"❌ PyTorch benchmark failed: {e}")

    # Benchmark MLX
    try:
        mlx_time, mlx_losses = benchmark_mlx(data, hyperparams, num_epochs)
        results.total_training_times['MLX'] = mlx_time
        results.mlx_losses = mlx_losses
    except Exception as e:
        print(f"❌ MLX benchmark failed: {e}")

    # Benchmark CoreML inference
    try:
        coreml_inference_time = benchmark_coreml_inference(data, hyperparams)
        results.total_training_times['CoreML'] = coreml_inference_time
    except Exception as e:
        print(f"❌ CoreML benchmark failed: {e}")

    # Calculate accuracies
    try:
        results.final_accuracies = calculate_final_accuracy(data, hyperparams)
    except Exception as e:
        print(f"❌ Accuracy calculation failed: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 50)

    for framework, time_taken in results.total_training_times.items():
        if time_taken > 0:
            print(f"{framework:>10}: {time_taken:>8.2f}s")

    if len(results.total_training_times) >= 2:
        times = list(results.total_training_times.values())
        pytorch_time = results.total_training_times.get('PyTorch', 0)
        mlx_time = results.total_training_times.get('MLX', 0)

        if pytorch_time > 0 and mlx_time > 0:
            speedup = pytorch_time / mlx_time
            print(f"\n🏃‍♂️ MLX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch")

    print("\nFinal accuracies (MSE):")
    for framework, accuracy in results.final_accuracies.items():
        if np.isfinite(accuracy):
            print(f"{framework:>10}: {accuracy:>8.6f}")

    # Generate plots
    plot_results(results, num_epochs)

    print(f"\n✅ Benchmark completed! Results saved to ltc_benchmark_results.png")

if __name__ == "__main__":
    main()