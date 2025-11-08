#!/usr/bin/env python3
"""
Complete benchmark including CoreML inference performance.
Compares training convergence + inference speed across all three frameworks.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)

print("=" * 80)
print("COMPLETE BENCHMARK: PyTorch vs MLX vs CoreML")
print("=" * 80)

# ============================================================================
# Configuration and Setup
# ============================================================================

# Hyperparameters
input_dim = 2
hidden_dim = 8
output_dim = 2
num_points = 500
num_turns = 3
learning_rate = 0.005
convergence_epochs = 100  # Reasonable convergence target
seq_len = 3
batch_size = 32

print(f"\nConfiguration:")
print(f"  Training epochs: {convergence_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")

# Generate spiral data (same as before)
def generate_spiral_data(num_points, num_turns, noise=4):
    theta = np.linspace(0, num_turns * 2 * np.pi, num_points)
    z = np.linspace(0, 1, num_points)
    r = z
    x = r * np.sin(theta) + noise * np.random.randn(*theta.shape) / num_points
    y = r * np.cos(theta) + noise * np.random.randn(*theta.shape) / num_points
    return np.stack([x, y], axis=1)

# Generate data once for all implementations
np.random.seed(42)
data = generate_spiral_data(num_points, num_turns)
all_inputs = data[:-1, :]
all_targets = data[1:, :]

# Prepare training data
trajectory_count = max(1, len(all_inputs) - seq_len)
train_inputs = [all_inputs[i:i + seq_len] for i in range(trajectory_count)]
train_targets = [all_targets[i:i + seq_len] for i in range(trajectory_count)]

# Shuffle and split
np.random.seed(42)
random_train_indices = np.arange(len(train_inputs))
np.random.shuffle(random_train_indices)
train_split_index = int(len(random_train_indices) * 0.8)
random_train_indices = random_train_indices[:train_split_index]

# Optimized data preparation
selected_inputs = [train_inputs[i] for i in random_train_indices]
selected_targets = [train_targets[i] for i in random_train_indices]
np_train_inputs = np.stack(selected_inputs)
np_train_targets = np.stack(selected_targets)

# Create batch indices
num_samples = len(np_train_inputs)
batch_indices = [(i, min(i + batch_size, num_samples)) for i in range(0, num_samples, batch_size)]

# ============================================================================
# Training Functions
# ============================================================================

def train_pytorch_model():
    """Train PyTorch model for comparison"""
    print("\n🔥 Training PyTorch Model")
    print("-" * 40)

    import torch
    import torch.nn as nn
    from ltc_coreml import RandomWiring, LTCRNN

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize model
    wiring = RandomWiring(input_dim, output_dim, hidden_dim)
    model = LTCRNN(wiring, input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    losses = []

    for epoch in range(convergence_epochs):
        model.train()
        total_loss = 0

        for start_idx, end_idx in batch_indices:
            optimizer.zero_grad()

            batch_inputs = torch.FloatTensor(np_train_inputs[start_idx:end_idx])
            batch_targets = torch.FloatTensor(np_train_targets[start_idx:end_idx])

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        losses.append(total_loss)

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{convergence_epochs}, Loss: {total_loss:.6f}")

    train_time = time.time() - start_time

    # Inference timing
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(torch.FloatTensor(all_inputs).unsqueeze(0))

    # Actual timing
    inference_start = time.time()
    num_inferences = 1000
    with torch.no_grad():
        for _ in range(num_inferences):
            predictions = model(torch.FloatTensor(all_inputs).unsqueeze(0))
    inference_time = (time.time() - inference_start) / num_inferences

    # Final validation
    with torch.no_grad():
        val_pred = model(torch.FloatTensor(all_inputs).unsqueeze(0))
        val_loss = criterion(val_pred, torch.FloatTensor(all_targets).unsqueeze(0))

    print(f"  ✅ Training completed: {train_time:.2f}s")
    print(f"  📊 Final loss: {total_loss:.6f}")
    print(f"  ⚡ Inference time: {inference_time*1000:.3f}ms")

    return {
        'train_time': train_time,
        'final_loss': total_loss,
        'val_loss': val_loss.item(),
        'inference_time': inference_time,
        'losses': losses,
        'predictions': val_pred.squeeze(0).numpy()
    }

def train_mlx_model():
    """Train MLX model for comparison"""
    print("\n⚡ Training MLX Model")
    print("-" * 40)

    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as mlx_optim
    from ltc_mlx import RandomWiring, LTCRNN

    # Set seeds
    mx.random.seed(42)
    np.random.seed(42)

    # Initialize model
    wiring = RandomWiring(input_dim, output_dim, hidden_dim)
    model = LTCRNN(wiring, input_dim, hidden_dim, output_dim)
    optimizer = mlx_optim.Adam(learning_rate=learning_rate)

    def mse_loss(predictions, targets):
        return mx.mean((predictions - targets) ** 2)

    def loss_fn(model, x, y_target):
        outputs = model(x)
        return mse_loss(outputs, y_target)

    loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

    # Convert data to MLX
    mx_train_inputs = mx.array(np_train_inputs)
    mx_train_targets = mx.array(np_train_targets)
    mx_all_inputs = mx.array(all_inputs)
    mx_all_targets = mx.array(all_targets)
    mx.eval(mx_train_inputs, mx_train_targets, mx_all_inputs, mx_all_targets)

    start_time = time.time()
    losses = []

    for epoch in range(convergence_epochs):
        total_loss = 0

        for start_idx, end_idx in batch_indices:
            x_batch = mx_train_inputs[start_idx:end_idx]
            y_batch = mx_train_targets[start_idx:end_idx]

            loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
            optimizer.update(model, grads)

            total_loss += loss.item()

        losses.append(total_loss)

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{convergence_epochs}, Loss: {total_loss:.6f}")

    # Final evaluation
    mx.eval(model.parameters(), optimizer.state)
    train_time = time.time() - start_time

    # Inference timing
    # Warmup
    for _ in range(10):
        _ = model(mx_all_inputs[None, :, :])
        mx.eval(_)

    # Actual timing
    inference_start = time.time()
    num_inferences = 1000
    for _ in range(num_inferences):
        predictions = model(mx_all_inputs[None, :, :])
        mx.eval(predictions)
    inference_time = (time.time() - inference_start) / num_inferences

    # Final validation
    val_pred = model(mx_all_inputs[None, :, :])
    val_loss = mse_loss(val_pred, mx_all_targets[None, :, :])
    mx.eval(val_pred, val_loss)

    print(f"  ✅ Training completed: {train_time:.2f}s")
    print(f"  📊 Final loss: {total_loss:.6f}")
    print(f"  ⚡ Inference time: {inference_time*1000:.3f}ms")

    return {
        'train_time': train_time,
        'final_loss': total_loss,
        'val_loss': val_loss.item(),
        'inference_time': inference_time,
        'losses': losses,
        'predictions': np.array(val_pred[0])
    }

def benchmark_coreml_model():
    """Benchmark CoreML model inference"""
    print("\n🍎 Benchmarking CoreML Model")
    print("-" * 40)

    try:
        import coremltools as ct

        if not os.path.exists("ltc_model.mlpackage"):
            print("  ❌ CoreML model not found. Please run ltc_coreml.py first.")
            return None

        # Load model
        model = ct.models.MLModel("ltc_model.mlpackage")

        # Get correct input name
        try:
            spec = model.get_spec()
            if hasattr(spec, 'description') and hasattr(spec.description, 'input'):
                input_name = spec.description.input[0].name
            else:
                input_name = "inputs"  # fallback
        except:
            input_name = "inputs"  # fallback

        print(f"  Using input name: {input_name}")

        # Prepare test data
        test_input = all_inputs.astype(np.float32)
        input_dict = {input_name: test_input[None, :, :]}  # Add batch dimension

        # Warmup
        for _ in range(10):
            _ = model.predict(input_dict)

        # Inference timing
        inference_start = time.time()
        num_inferences = 1000

        predictions_list = []
        for _ in range(num_inferences):
            result = model.predict(input_dict)
            if _ == 0:  # Store first prediction for analysis
                # Get output (name might vary)
                output_key = list(result.keys())[0]
                predictions = result[output_key]
                if len(predictions.shape) == 3 and predictions.shape[0] == 1:
                    predictions = predictions[0]  # Remove batch dimension
                predictions_list.append(predictions)

        inference_time = (time.time() - inference_start) / num_inferences

        # Calculate loss against true targets
        if predictions_list:
            coreml_predictions = predictions_list[0]
            if len(coreml_predictions) == len(all_targets):
                val_loss = np.mean((coreml_predictions - all_targets) ** 2)
            else:
                val_loss = float('inf')  # Size mismatch
                print(f"  ⚠️  Size mismatch: predictions {coreml_predictions.shape}, targets {all_targets.shape}")
        else:
            val_loss = float('inf')
            coreml_predictions = np.zeros_like(all_targets)

        print(f"  ✅ CoreML model loaded successfully")
        print(f"  📊 Validation loss: {val_loss:.6f}")
        print(f"  ⚡ Inference time: {inference_time*1000:.3f}ms")

        return {
            'inference_time': inference_time,
            'val_loss': val_loss,
            'predictions': coreml_predictions,
            'model_size': "Pre-trained"
        }

    except ImportError:
        print("  ❌ CoreML not available. Install with: pip install coremltools")
        return None
    except Exception as e:
        print(f"  ❌ CoreML benchmark failed: {e}")
        return None

# ============================================================================
# Run Complete Benchmark
# ============================================================================

print("\n" + "=" * 80)
print("Running Complete Framework Benchmark")
print("=" * 80)

# Train and benchmark all models
pytorch_results = train_pytorch_model()
mlx_results = train_mlx_model()
coreml_results = benchmark_coreml_model()

# ============================================================================
# Comprehensive Analysis
# ============================================================================

print("\n" + "=" * 80)
print("🏆 COMPREHENSIVE BENCHMARK RESULTS")
print("=" * 80)

print(f"\n📊 TRAINING PERFORMANCE:")
print(f"{'Framework':<12} | {'Train Time':<12} | {'Final Loss':<12} | {'Val Loss':<12}")
print("-" * 60)
print(f"{'PyTorch':<12} | {pytorch_results['train_time']:>10.2f}s | {pytorch_results['final_loss']:>10.6f} | {pytorch_results['val_loss']:>10.6f}")
print(f"{'MLX':<12} | {mlx_results['train_time']:>10.2f}s | {mlx_results['final_loss']:>10.6f} | {mlx_results['val_loss']:>10.6f}")

print(f"\n⚡ INFERENCE PERFORMANCE:")
print(f"{'Framework':<12} | {'Inference Time':<15} | {'Speedup vs PyTorch':<20}")
print("-" * 50)
print(f"{'PyTorch':<12} | {pytorch_results['inference_time']*1000:>13.3f}ms | {'1.00x (baseline)':<20}")
print(f"{'MLX':<12} | {mlx_results['inference_time']*1000:>13.3f}ms | {pytorch_results['inference_time']/mlx_results['inference_time']:>18.2f}x")

if coreml_results:
    print(f"{'CoreML':<12} | {coreml_results['inference_time']*1000:>13.3f}ms | {pytorch_results['inference_time']/coreml_results['inference_time']:>18.2f}x")

print(f"\n🎯 OVERALL EFFICIENCY:")
training_speedup = pytorch_results['train_time'] / mlx_results['train_time']
inference_speedup = pytorch_results['inference_time'] / mlx_results['inference_time']
print(f"  MLX Training: {training_speedup:.2f}x {'faster' if training_speedup > 1 else 'slower'}")
print(f"  MLX Inference: {inference_speedup:.2f}x {'faster' if inference_speedup > 1 else 'slower'}")

if coreml_results:
    coreml_speedup = pytorch_results['inference_time'] / coreml_results['inference_time']
    print(f"  CoreML Inference: {coreml_speedup:.2f}x {'faster' if coreml_speedup > 1 else 'slower'}")

# ============================================================================
# Visualization
# ============================================================================

print("\nGenerating comprehensive comparison plots...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Training curves
ax1.semilogy(pytorch_results['losses'], 'r-', label='PyTorch', alpha=0.8, linewidth=2)
ax1.semilogy(mlx_results['losses'], 'b-', label='MLX', alpha=0.8, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Convergence Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Inference speed comparison
frameworks = ['PyTorch', 'MLX']
inference_times = [pytorch_results['inference_time']*1000, mlx_results['inference_time']*1000]
colors = ['red', 'blue']

if coreml_results:
    frameworks.append('CoreML')
    inference_times.append(coreml_results['inference_time']*1000)
    colors.append('green')

bars = ax2.bar(frameworks, inference_times, color=colors, alpha=0.7)
ax2.set_ylabel('Inference Time (ms)')
ax2.set_title('Inference Speed Comparison')
ax2.grid(True, alpha=0.3)

# Add values on bars
for bar, time in zip(bars, inference_times):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{time:.2f}ms', ha='center', va='bottom')

# Plot 3: Prediction accuracy comparison
ax3.plot(all_targets[:, 0], all_targets[:, 1], 'g-', linewidth=2, label='True Path', alpha=0.8)
ax3.plot(pytorch_results['predictions'][:, 0], pytorch_results['predictions'][:, 1], 'r--',
         linewidth=2, label=f'PyTorch (Loss: {pytorch_results["val_loss"]:.4f})', alpha=0.8)
ax3.plot(mlx_results['predictions'][:, 0], mlx_results['predictions'][:, 1], 'b:',
         linewidth=2, label=f'MLX (Loss: {mlx_results["val_loss"]:.4f})', alpha=0.8)

if coreml_results and np.isfinite(coreml_results['val_loss']):
    ax3.plot(coreml_results['predictions'][:, 0], coreml_results['predictions'][:, 1], 'm-.',
             linewidth=2, label=f'CoreML (Loss: {coreml_results["val_loss"]:.4f})', alpha=0.8)

ax3.set_title('Prediction Accuracy Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')

# Plot 4: Performance summary radar-like plot
categories = ['Training Speed', 'Inference Speed', 'Accuracy']
pytorch_scores = [1.0, 1.0, 1.0]  # Baseline
mlx_scores = [
    training_speedup,
    inference_speedup,
    pytorch_results['val_loss'] / mlx_results['val_loss']  # Accuracy ratio
]

x_pos = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, pytorch_scores, width, label='PyTorch', color='red', alpha=0.7)
bars2 = ax4.bar(x_pos + width/2, mlx_scores, width, label='MLX', color='blue', alpha=0.7)

ax4.set_ylabel('Relative Performance (Higher = Better)')
ax4.set_title('Overall Performance Comparison')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(categories)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('complete_benchmark_results.png', dpi=150, bbox_inches='tight')
print("Complete benchmark results saved to: complete_benchmark_results.png")
plt.show()

print("\n" + "=" * 80)
print("🚀 Complete Benchmark Finished!")
print("=" * 80)