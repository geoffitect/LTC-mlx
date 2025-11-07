#!/usr/bin/env python3
"""
Final comprehensive benchmark with optimized CoreML model.
Compares PyTorch, MLX, and properly trained CoreML across all metrics.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)

print("=" * 80)
print("🏆 FINAL COMPREHENSIVE BENCHMARK")
print("PyTorch vs MLX vs Optimized CoreML")
print("=" * 80)

# ============================================================================
# Configuration and Data Setup
# ============================================================================

# Hyperparameters
input_dim = 2
hidden_dim = 8
output_dim = 2
num_points = 500
num_turns = 3
learning_rate = 0.005
convergence_epochs = 100  # Fair comparison epochs
seq_len = 3
batch_size = 32

def generate_spiral_data(num_points, num_turns, noise=4):
    theta = np.linspace(0, num_turns * 2 * np.pi, num_points)
    z = np.linspace(0, 1, num_points)
    r = z
    x = r * np.sin(theta) + noise * np.random.randn(*theta.shape) / num_points
    y = r * np.cos(theta) + noise * np.random.randn(*theta.shape) / num_points
    return np.stack([x, y], axis=1)

# Generate data
np.random.seed(42)
data = generate_spiral_data(num_points, num_turns)
all_inputs = data[:-1, :]
all_targets = data[1:, :]

# Prepare training/test data
trajectory_count = max(1, len(all_inputs) - seq_len)
train_inputs = [all_inputs[i:i + seq_len] for i in range(trajectory_count)]
train_targets = [all_targets[i:i + seq_len] for i in range(trajectory_count)]

# Use consistent random split
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

# Test data for inference (properly sized sequences)
test_sequences = []
test_targets_seq = []
for i in range(0, len(all_inputs) - seq_len, 10):  # Every 10th sequence for testing
    test_sequences.append(all_inputs[i:i + seq_len])
    test_targets_seq.append(all_targets[i:i + seq_len])

test_sequences = np.stack(test_sequences)
test_targets_seq = np.stack(test_targets_seq)

# Create batch indices
num_samples = len(np_train_inputs)
batch_indices = [(i, min(i + batch_size, num_samples)) for i in range(0, num_samples, batch_size)]

print(f"📊 Dataset Info:")
print(f"  Training sequences: {len(np_train_inputs)}")
print(f"  Test sequences: {len(test_sequences)}")
print(f"  Sequence length: {seq_len}")
print(f"  Training batches: {len(batch_indices)}")

# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_pytorch():
    """Benchmark PyTorch training and inference"""
    print("\n🔥 PyTorch Benchmark")
    print("-" * 40)

    import torch
    import torch.nn as nn
    from ltc_coreml import RandomWiring, LTCRNN

    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize model
    wiring = RandomWiring(input_dim, output_dim, hidden_dim)
    model = LTCRNN(wiring, input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train_start = time.time()
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

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{convergence_epochs}, Loss: {total_loss:.6f}")

    train_time = time.time() - train_start

    # Inference benchmark
    model.eval()
    test_input_tensor = torch.FloatTensor(test_sequences)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input_tensor)

    # Timing
    inference_start = time.time()
    num_inferences = 1000
    with torch.no_grad():
        for _ in range(num_inferences):
            predictions = model(test_input_tensor)
    inference_time = (time.time() - inference_start) / num_inferences

    # Validation loss
    with torch.no_grad():
        test_targets_tensor = torch.FloatTensor(test_targets_seq)
        val_loss = criterion(predictions, test_targets_tensor)

    print(f"  ✅ Training time: {train_time:.2f}s")
    print(f"  📊 Final training loss: {total_loss:.6f}")
    print(f"  🎯 Validation loss: {val_loss.item():.6f}")
    print(f"  ⚡ Inference time: {inference_time*1000:.3f}ms")

    return {
        'train_time': train_time,
        'final_train_loss': total_loss,
        'val_loss': val_loss.item(),
        'inference_time': inference_time,
        'predictions': predictions.numpy()
    }

def benchmark_mlx():
    """Benchmark MLX training and inference"""
    print("\n⚡ MLX Benchmark")
    print("-" * 40)

    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as mlx_optim
    from ltc_mlx_fixed import RandomWiring, LTCRNN

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
    mx_test_inputs = mx.array(test_sequences)
    mx_test_targets = mx.array(test_targets_seq)
    mx.eval(mx_train_inputs, mx_train_targets, mx_test_inputs, mx_test_targets)

    # Training
    train_start = time.time()
    for epoch in range(convergence_epochs):
        total_loss = 0

        for start_idx, end_idx in batch_indices:
            x_batch = mx_train_inputs[start_idx:end_idx]
            y_batch = mx_train_targets[start_idx:end_idx]

            loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
            optimizer.update(model, grads)

            total_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{convergence_epochs}, Loss: {total_loss:.6f}")

    mx.eval(model.parameters(), optimizer.state)
    train_time = time.time() - train_start

    # Inference benchmark
    # Warmup
    for _ in range(10):
        _ = model(mx_test_inputs)
        mx.eval(_)

    # Timing
    inference_start = time.time()
    num_inferences = 1000
    for _ in range(num_inferences):
        predictions = model(mx_test_inputs)
        mx.eval(predictions)
    inference_time = (time.time() - inference_start) / num_inferences

    # Validation loss
    val_loss = mse_loss(predictions, mx_test_targets)
    mx.eval(val_loss)

    print(f"  ✅ Training time: {train_time:.2f}s")
    print(f"  📊 Final training loss: {total_loss:.6f}")
    print(f"  🎯 Validation loss: {val_loss.item():.6f}")
    print(f"  ⚡ Inference time: {inference_time*1000:.3f}ms")

    return {
        'train_time': train_time,
        'final_train_loss': total_loss,
        'val_loss': val_loss.item(),
        'inference_time': inference_time,
        'predictions': np.array(predictions)
    }

def benchmark_coreml():
    """Benchmark optimized CoreML inference"""
    print("\n🍎 CoreML Benchmark")
    print("-" * 40)

    try:
        import coremltools as ct

        # Check for optimized model first
        model_path = "ltc_model_optimized.mlpackage"
        if not os.path.exists(model_path):
            model_path = "ltc_model.mlpackage"
            if not os.path.exists(model_path):
                print("  ❌ No CoreML model found")
                return None

        print(f"  Using model: {model_path}")

        # Load model
        model = ct.models.MLModel(model_path)

        # Get correct input name
        try:
            spec = model.get_spec()
            if hasattr(spec, 'description') and hasattr(spec.description, 'input'):
                input_name = spec.description.input[0].name
            else:
                input_name = "inputs"
        except:
            input_name = "inputs"

        print(f"  Input name: {input_name}")

        # Prepare test data (correctly sized)
        coreml_inputs = []
        for i, test_seq in enumerate(test_sequences):
            input_dict = {input_name: test_seq.astype(np.float32)[None, :, :]}  # Add batch dim
            coreml_inputs.append(input_dict)

        # Test first sequence
        result = model.predict(coreml_inputs[0])
        output_key = list(result.keys())[0]
        print(f"  Output key: {output_key}")

        # Warmup
        for _ in range(10):
            _ = model.predict(coreml_inputs[0])

        # Inference timing
        inference_start = time.time()
        num_inferences = 1000

        all_predictions = []
        for _ in range(num_inferences):
            # Test on first sequence for timing
            result = model.predict(coreml_inputs[0])
            if len(all_predictions) == 0:  # Store first result
                pred = result[output_key]
                if len(pred.shape) == 3 and pred.shape[0] == 1:
                    pred = pred[0]  # Remove batch dimension
                all_predictions.append(pred)

        inference_time = (time.time() - inference_start) / num_inferences

        # Get predictions for all test sequences
        all_test_predictions = []
        for input_dict in coreml_inputs:
            result = model.predict(input_dict)
            pred = result[output_key]
            if len(pred.shape) == 3 and pred.shape[0] == 1:
                pred = pred[0]
            all_test_predictions.append(pred)

        # Stack predictions
        coreml_predictions = np.stack(all_test_predictions)

        # Calculate validation loss
        val_loss = np.mean((coreml_predictions - test_targets_seq) ** 2)

        print(f"  🎯 Validation loss: {val_loss:.6f}")
        print(f"  ⚡ Inference time: {inference_time*1000:.3f}ms")

        return {
            'inference_time': inference_time,
            'val_loss': val_loss,
            'predictions': coreml_predictions,
            'model_path': model_path
        }

    except ImportError:
        print("  ❌ CoreML not available")
        return None
    except Exception as e:
        print(f"  ❌ CoreML benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Run Complete Benchmark
# ============================================================================

print("\n" + "=" * 80)
print("🚀 RUNNING FINAL BENCHMARK")
print("=" * 80)

pytorch_results = benchmark_pytorch()
mlx_results = benchmark_mlx()
coreml_results = benchmark_coreml()

# ============================================================================
# Final Analysis
# ============================================================================

print("\n" + "=" * 80)
print("🏆 FINAL BENCHMARK RESULTS")
print("=" * 80)

print(f"\n📊 TRAINING PERFORMANCE:")
print(f"{'Framework':<12} | {'Time (s)':<10} | {'Final Loss':<12} | {'Val Loss':<12}")
print("-" * 55)
print(f"{'PyTorch':<12} | {pytorch_results['train_time']:>8.2f} | {pytorch_results['final_train_loss']:>10.6f} | {pytorch_results['val_loss']:>10.6f}")
print(f"{'MLX':<12} | {mlx_results['train_time']:>8.2f} | {mlx_results['final_train_loss']:>10.6f} | {mlx_results['val_loss']:>10.6f}")

print(f"\n⚡ INFERENCE PERFORMANCE:")
print(f"{'Framework':<12} | {'Time (ms)':<12} | {'Speedup':<12} | {'Val Loss':<12}")
print("-" * 60)
print(f"{'PyTorch':<12} | {pytorch_results['inference_time']*1000:>10.3f} | {'1.00x':<12} | {pytorch_results['val_loss']:>10.6f}")
print(f"{'MLX':<12} | {mlx_results['inference_time']*1000:>10.3f} | {pytorch_results['inference_time']/mlx_results['inference_time']:>10.2f}x | {mlx_results['val_loss']:>10.6f}")

if coreml_results:
    print(f"{'CoreML':<12} | {coreml_results['inference_time']*1000:>10.3f} | {pytorch_results['inference_time']/coreml_results['inference_time']:>10.2f}x | {coreml_results['val_loss']:>10.6f}")

print(f"\n🏆 WINNER ANALYSIS:")
training_winner = "MLX" if mlx_results['train_time'] < pytorch_results['train_time'] else "PyTorch"
accuracy_winner = "MLX" if mlx_results['val_loss'] < pytorch_results['val_loss'] else "PyTorch"

print(f"  Training Speed: {training_winner}")
print(f"  Training Accuracy: {accuracy_winner}")

if coreml_results:
    inference_times = {
        'PyTorch': pytorch_results['inference_time'],
        'MLX': mlx_results['inference_time'],
        'CoreML': coreml_results['inference_time']
    }
    inference_winner = min(inference_times.keys(), key=lambda k: inference_times[k])
    print(f"  Inference Speed: {inference_winner}")

# ============================================================================
# Visualization
# ============================================================================

print(f"\nGenerating final comparison visualization...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Training speed comparison
frameworks = ['PyTorch', 'MLX']
train_times = [pytorch_results['train_time'], mlx_results['train_time']]
colors = ['red', 'blue']

bars = ax1.bar(frameworks, train_times, color=colors, alpha=0.7)
ax1.set_ylabel('Training Time (s)')
ax1.set_title('Training Speed Comparison')
ax1.grid(True, alpha=0.3)

for bar, time in zip(bars, train_times):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{time:.1f}s', ha='center', va='bottom')

# Plot 2: Inference speed comparison
frameworks_inf = ['PyTorch', 'MLX']
inference_times_ms = [pytorch_results['inference_time']*1000, mlx_results['inference_time']*1000]
colors_inf = ['red', 'blue']

if coreml_results:
    frameworks_inf.append('CoreML')
    inference_times_ms.append(coreml_results['inference_time']*1000)
    colors_inf.append('green')

bars = ax2.bar(frameworks_inf, inference_times_ms, color=colors_inf, alpha=0.7)
ax2.set_ylabel('Inference Time (ms)')
ax2.set_title('Inference Speed Comparison')
ax2.grid(True, alpha=0.3)

for bar, time in zip(bars, inference_times_ms):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{time:.2f}ms', ha='center', va='bottom')

# Plot 3: Accuracy comparison
frameworks_acc = ['PyTorch', 'MLX']
val_losses = [pytorch_results['val_loss'], mlx_results['val_loss']]
colors_acc = ['red', 'blue']

if coreml_results and np.isfinite(coreml_results['val_loss']):
    frameworks_acc.append('CoreML')
    val_losses.append(coreml_results['val_loss'])
    colors_acc.append('green')

bars = ax3.bar(frameworks_acc, val_losses, color=colors_acc, alpha=0.7)
ax3.set_ylabel('Validation Loss (Lower = Better)')
ax3.set_title('Prediction Accuracy Comparison')
ax3.grid(True, alpha=0.3)

for bar, loss in zip(bars, val_losses):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{loss:.4f}', ha='center', va='bottom')

# Plot 4: Sample predictions comparison
sample_idx = 0  # First test sequence
sample_target = test_targets_seq[sample_idx]
sample_pytorch = pytorch_results['predictions'][sample_idx]
sample_mlx = mlx_results['predictions'][sample_idx]

ax4.plot(sample_target[:, 0], sample_target[:, 1], 'g-o', linewidth=2, markersize=6, label='True')
ax4.plot(sample_pytorch[:, 0], sample_pytorch[:, 1], 'r--s', linewidth=2, markersize=4, label='PyTorch')
ax4.plot(sample_mlx[:, 0], sample_mlx[:, 1], 'b:^', linewidth=2, markersize=4, label='MLX')

if coreml_results and len(coreml_results['predictions']) > sample_idx:
    sample_coreml = coreml_results['predictions'][sample_idx]
    ax4.plot(sample_coreml[:, 0], sample_coreml[:, 1], 'm-.d', linewidth=2, markersize=4, label='CoreML')

ax4.set_title('Sample Sequence Prediction')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')

plt.tight_layout()
plt.savefig('final_benchmark_results.png', dpi=150, bbox_inches='tight')
print("Final results saved to: final_benchmark_results.png")
plt.show()

print(f"\n" + "=" * 80)
print("🎉 FINAL BENCHMARK COMPLETE!")
print("=" * 80)