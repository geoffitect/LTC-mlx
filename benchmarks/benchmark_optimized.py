#!/usr/bin/env python3
"""
Optimized benchmark script to compare PyTorch and MLX implementations.
Addresses major performance bottlenecks identified in profiling.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)

print("=" * 80)
print("OPTIMIZED LTC Network Benchmark: PyTorch vs MLX")
print("=" * 80)

# ============================================================================
# Common Setup
# ============================================================================

# Hyperparameters
input_dim = 2
hidden_dim = 8
output_dim = 2
num_points = 500
num_turns = 3
learning_rate = 0.005
num_epochs = 50
seq_len = 3
batch_size = 32

print(f"\nConfiguration:")
print(f"  Input dim: {input_dim}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Output dim: {output_dim}")
print(f"  Num epochs: {num_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Data points: {num_points}")

# Generate spiral data
def generate_spiral_data(num_points, num_turns, noise=4):
    theta = np.linspace(0, num_turns * 2 * np.pi, num_points)
    z = np.linspace(0, 1, num_points)
    r = z
    x = r * np.sin(theta) + noise * np.random.randn(*theta.shape) / num_points
    y = r * np.cos(theta) + noise * np.random.randn(*theta.shape) / num_points
    return np.stack([x, y], axis=1)

# Generate data once for both implementations
np.random.seed(42)
data = generate_spiral_data(num_points, num_turns)
all_inputs = data[:-1, :]
all_targets = data[1:, :]

# Prepare input and target sequences
trajectory_count = max(1, len(all_inputs) - seq_len)
train_inputs = [all_inputs[i:i + seq_len] for i in range(trajectory_count)]
train_targets = [all_targets[i:i + seq_len] for i in range(trajectory_count)]

# Shuffle and split the data for training
np.random.seed(42)
random_train_indices = np.arange(len(train_inputs))
np.random.shuffle(random_train_indices)
train_split_index = int(len(random_train_indices) * 0.8)
random_train_indices = random_train_indices[:train_split_index]

# OPTIMIZED: Create numpy arrays first, then batch
print("Creating optimized data batches...")
selected_inputs = [train_inputs[i] for i in random_train_indices]
selected_targets = [train_targets[i] for i in random_train_indices]

# Stack into numpy arrays for efficient batching
np_train_inputs = np.stack(selected_inputs)
np_train_targets = np.stack(selected_targets)

# Create batch indices
num_samples = len(np_train_inputs)
batch_indices = [(i, min(i + batch_size, num_samples)) for i in range(0, num_samples, batch_size)]

print(f"Created {len(batch_indices)} batches of data")

# ============================================================================
# PyTorch Implementation (for comparison)
# ============================================================================

print("\n" + "=" * 80)
print("Training PyTorch Implementation")
print("=" * 80)

import torch
import torch.nn as nn
from ltc_coreml import RandomWiring, LTCRNN

# Set PyTorch seed
torch.manual_seed(42)
np.random.seed(42)

# Initialize model
wiring_torch = RandomWiring(input_dim, output_dim, hidden_dim)
model_torch = LTCRNN(wiring_torch, input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=learning_rate)

# Training with optimized data loading
train_start_torch = time.time()
for epoch in range(num_epochs):
    model_torch.train()
    total_loss = 0

    for start_idx, end_idx in batch_indices:
        optimizer_torch.zero_grad()

        # Convert batch to torch tensors
        batch_inputs = torch.FloatTensor(np_train_inputs[start_idx:end_idx])
        batch_targets = torch.FloatTensor(np_train_targets[start_idx:end_idx])

        outputs = model_torch(batch_inputs)
        loss = criterion(outputs, batch_targets)
        total_loss += loss.item()
        loss.backward()
        optimizer_torch.step()

    if (epoch + 1) % 10 == 0:
        print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

train_end_torch = time.time()
train_time_torch = train_end_torch - train_start_torch

# Inference timing
model_torch.eval()
inference_start_torch = time.time()
with torch.no_grad():
    predictions_torch = model_torch(torch.FloatTensor(all_inputs).unsqueeze(0))
    val_loss_torch = criterion(predictions_torch, torch.FloatTensor(all_targets).unsqueeze(0))
inference_end_torch = time.time()
inference_time_torch = inference_end_torch - inference_start_torch

np_predictions_torch = predictions_torch.squeeze(0).numpy()

print(f"\nPyTorch Results:")
print(f"  Training time: {train_time_torch:.4f} seconds")
print(f"  Inference time: {inference_time_torch:.6f} seconds")
print(f"  Final validation loss: {val_loss_torch.item():.6f}")

# ============================================================================
# Optimized MLX Implementation
# ============================================================================

print("\n" + "=" * 80)
print("Training OPTIMIZED MLX Implementation")
print("=" * 80)

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
from ltc_mlx_fixed import RandomWiring as RandomWiringMLX
from ltc_mlx_fixed import LTCRNN as LTCRNNMLX

# Set MLX seed
mx.random.seed(42)
np.random.seed(42)

# Initialize model
wiring_mlx = RandomWiringMLX(input_dim, output_dim, hidden_dim)
model_mlx = LTCRNNMLX(wiring_mlx, input_dim, hidden_dim, output_dim)

# Initialize optimizer
optimizer_mlx = mlx_optim.Adam(learning_rate=learning_rate)

# Optimized loss function (no extra regularization)
def mse_loss(predictions, targets):
    return mx.mean((predictions - targets) ** 2)

def loss_fn(model, x, y_target):
    outputs = model(x)
    return mse_loss(outputs, y_target)

# Create loss and gradient function
loss_and_grad_fn = mlx_nn.value_and_grad(model_mlx, loss_fn)

# OPTIMIZED: Convert all data to MLX upfront
print("Converting data to MLX format...")
data_convert_start = time.time()
mx_train_inputs = mx.array(np_train_inputs)
mx_train_targets = mx.array(np_train_targets)
mx_all_inputs = mx.array(all_inputs)
mx_all_targets = mx.array(all_targets)
# Force evaluation once
mx.eval(mx_train_inputs, mx_train_targets, mx_all_inputs, mx_all_targets)
data_convert_time = time.time() - data_convert_start
print(f"Data conversion took: {data_convert_time:.4f}s")

# Training with optimized data access
train_start_mlx = time.time()
for epoch in range(num_epochs):
    total_loss = 0

    for start_idx, end_idx in batch_indices:
        # OPTIMIZED: Direct slicing from pre-converted MLX arrays
        x_batch = mx_train_inputs[start_idx:end_idx]
        y_batch = mx_train_targets[start_idx:end_idx]

        # Forward pass and compute gradients
        loss, grads = loss_and_grad_fn(model_mlx, x_batch, y_batch)

        # Update model parameters
        optimizer_mlx.update(model_mlx, grads)

        # OPTIMIZED: Reduce mx.eval calls - only eval loss for accumulation
        total_loss += loss.item()

    # OPTIMIZED: Only eval model parameters at end of epoch
    if (epoch + 1) % 10 == 0:
        mx.eval(model_mlx.parameters(), optimizer_mlx.state)
        print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

# Final evaluation
mx.eval(model_mlx.parameters(), optimizer_mlx.state)
train_end_mlx = time.time()
train_time_mlx = train_end_mlx - train_start_mlx

# Inference timing
inference_start_mlx = time.time()
predictions_mlx = model_mlx(mx_all_inputs[None, :, :])
val_loss_mlx = mse_loss(predictions_mlx, mx_all_targets[None, :, :])
# Single eval at the end
mx.eval(predictions_mlx, val_loss_mlx)
inference_end_mlx = time.time()
inference_time_mlx = inference_end_mlx - inference_start_mlx

np_predictions_mlx = np.array(predictions_mlx[0])

print(f"\nOptimized MLX Results:")
print(f"  Data conversion time: {data_convert_time:.4f} seconds")
print(f"  Training time: {train_time_mlx:.4f} seconds")
print(f"  Inference time: {inference_time_mlx:.6f} seconds")
print(f"  Final validation loss: {val_loss_mlx.item():.6f}")

# ============================================================================
# Comparison
# ============================================================================

print("\n" + "=" * 80)
print("Optimized Comparison Summary")
print("=" * 80)

# Adjust MLX time to include data conversion for fair comparison
total_mlx_time = train_time_mlx + data_convert_time

print(f"\nTraining Time:")
print(f"  PyTorch: {train_time_torch:.4f}s")
print(f"  MLX (training only): {train_time_mlx:.4f}s")
print(f"  MLX (total with conversion): {total_mlx_time:.4f}s")
print(f"  Speedup (training only): {train_time_torch / train_time_mlx:.2f}x {'(MLX faster)' if train_time_mlx < train_time_torch else '(PyTorch faster)'}")
print(f"  Speedup (total): {train_time_torch / total_mlx_time:.2f}x {'(MLX faster)' if total_mlx_time < train_time_torch else '(PyTorch faster)'}")

print(f"\nInference Time:")
print(f"  PyTorch: {inference_time_torch:.6f}s")
print(f"  MLX:     {inference_time_mlx:.6f}s")
print(f"  Speedup: {inference_time_torch / inference_time_mlx:.2f}x {'(MLX faster)' if inference_time_mlx < inference_time_torch else '(PyTorch faster)'}")

print(f"\nFinal Validation Loss:")
print(f"  PyTorch: {val_loss_torch.item():.6f}")
print(f"  MLX:     {val_loss_mlx.item():.6f}")
print(f"  Difference: {abs(val_loss_torch.item() - val_loss_mlx.item()):.6f}")
print(f"  MLX accuracy improvement: {val_loss_torch.item() / val_loss_mlx.item():.2f}x better")

# Prediction accuracy comparison
pred_diff = np.abs(np_predictions_torch - np_predictions_mlx)
print(f"\nPrediction Difference (L1 norm):")
print(f"  Mean:  {np.mean(pred_diff):.6f}")
print(f"  Max:   {np.max(pred_diff):.6f}")
print(f"  Std:   {np.std(pred_diff):.6f}")

# ============================================================================
# Visualization
# ============================================================================

print("\nGenerating optimized comparison plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: PyTorch predictions
axes[0].plot(all_targets[:, 0], all_targets[:, 1], 'g-', linewidth=2, label='True Path', alpha=0.7)
axes[0].plot(np_predictions_torch[:, 0], np_predictions_torch[:, 1], 'r--', linewidth=2, label='PyTorch Prediction')
axes[0].set_title(f'PyTorch (Loss: {val_loss_torch.item():.6f})\nTime: {train_time_torch:.2f}s')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

# Plot 2: MLX predictions
axes[1].plot(all_targets[:, 0], all_targets[:, 1], 'g-', linewidth=2, label='True Path', alpha=0.7)
axes[1].plot(np_predictions_mlx[:, 0], np_predictions_mlx[:, 1], 'b--', linewidth=2, label='MLX Prediction')
axes[1].set_title(f'Optimized MLX (Loss: {val_loss_mlx.item():.6f})\nTime: {train_time_mlx:.2f}s')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

# Plot 3: Overlay comparison
axes[2].plot(all_targets[:, 0], all_targets[:, 1], 'g-', linewidth=2, label='True Path', alpha=0.7)
axes[2].plot(np_predictions_torch[:, 0], np_predictions_torch[:, 1], 'r--', linewidth=2, label='PyTorch', alpha=0.7)
axes[2].plot(np_predictions_mlx[:, 0], np_predictions_mlx[:, 1], 'b:', linewidth=2, label='Optimized MLX', alpha=0.7)
axes[2].set_title('Optimized Comparison Overlay')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')

plt.tight_layout()
plt.savefig('benchmark_optimized_results.png', dpi=150, bbox_inches='tight')
print("Plot saved to: benchmark_optimized_results.png")
plt.show()

print("\n" + "=" * 80)
print("Optimized Benchmark Complete!")
print("=" * 80)