#!/usr/bin/env python3
"""
Accuracy validation script to verify that PyTorch and MLX implementations
produce identical results when initialized with the same weights.
"""

import numpy as np
import torch
import mlx.core as mx

# Import implementations
from ltc import RandomWiring, LTCRNN, LIFNeuronLayer
from ltc_mlx import RandomWiring as RandomWiringMLX, LTCRNN as LTCRNNMLX

print("=" * 80)
print("LTC Network Accuracy Validation: PyTorch vs MLX")
print("=" * 80)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
mx.random.seed(42)

# Configuration
input_dim = 2
hidden_dim = 8
output_dim = 2
batch_size = 4
seq_len = 3

print(f"\nConfiguration:")
print(f"  Input dim: {input_dim}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Output dim: {output_dim}")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")

# ============================================================================
# Initialize Models with Same Random Seed
# ============================================================================

print("\n" + "-" * 80)
print("Initializing models with same random seed...")
print("-" * 80)

# PyTorch model
np.random.seed(42)
wiring_torch = RandomWiring(input_dim, output_dim, hidden_dim)
model_torch = LTCRNN(wiring_torch, input_dim, hidden_dim, output_dim)
model_torch.eval()

# MLX model
np.random.seed(42)
wiring_mlx = RandomWiringMLX(input_dim, output_dim, hidden_dim)
model_mlx = LTCRNNMLX(wiring_mlx, input_dim, hidden_dim, output_dim)

# ============================================================================
# Copy Weights from PyTorch to MLX
# ============================================================================

print("\nCopying weights from PyTorch to MLX...")

# Access the LIFNeuronLayer in both models
torch_neuron = model_torch.cell.neuron
mlx_neuron = model_mlx.cell.neuron

# Copy all parameters
mlx_neuron.gleak = mx.array(torch_neuron.gleak.detach().numpy())
mlx_neuron.vleak = mx.array(torch_neuron.vleak.detach().numpy())
mlx_neuron.cm = mx.array(torch_neuron.cm.detach().numpy())
mlx_neuron.w = mx.array(torch_neuron.w.detach().numpy())
mlx_neuron.sigma = mx.array(torch_neuron.sigma.detach().numpy())
mlx_neuron.mu = mx.array(torch_neuron.mu.detach().numpy())
mlx_neuron.erev = mx.array(torch_neuron.erev.detach().numpy())

mlx_neuron.sensory_w = mx.array(torch_neuron.sensory_w.detach().numpy())
mlx_neuron.sensory_sigma = mx.array(torch_neuron.sensory_sigma.detach().numpy())
mlx_neuron.sensory_mu = mx.array(torch_neuron.sensory_mu.detach().numpy())
mlx_neuron.sensory_erev = mx.array(torch_neuron.sensory_erev.detach().numpy())

print("✓ Weights copied successfully")

# ============================================================================
# Generate Test Data
# ============================================================================

print("\nGenerating test data...")

# Create random input data
np.random.seed(123)
test_input = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)

# Convert to framework-specific tensors
test_input_torch = torch.FloatTensor(test_input)
test_input_mlx = mx.array(test_input)

print(f"Test input shape: {test_input.shape}")
print(f"Test input range: [{test_input.min():.4f}, {test_input.max():.4f}]")

# ============================================================================
# Forward Pass
# ============================================================================

print("\n" + "-" * 80)
print("Running forward pass...")
print("-" * 80)

# PyTorch forward pass
with torch.no_grad():
    output_torch = model_torch(test_input_torch)
output_torch_np = output_torch.numpy()

print(f"\nPyTorch output shape: {output_torch_np.shape}")
print(f"PyTorch output range: [{output_torch_np.min():.6f}, {output_torch_np.max():.6f}]")
print(f"PyTorch output mean: {output_torch_np.mean():.6f}")
print(f"PyTorch output std: {output_torch_np.std():.6f}")

# MLX forward pass
output_mlx = model_mlx(test_input_mlx)
output_mlx_np = np.array(output_mlx)

print(f"\nMLX output shape: {output_mlx_np.shape}")
print(f"MLX output range: [{output_mlx_np.min():.6f}, {output_mlx_np.max():.6f}]")
print(f"MLX output mean: {output_mlx_np.mean():.6f}")
print(f"MLX output std: {output_mlx_np.std():.6f}")

# ============================================================================
# Compare Outputs
# ============================================================================

print("\n" + "=" * 80)
print("Comparison Results")
print("=" * 80)

# Absolute difference
abs_diff = np.abs(output_torch_np - output_mlx_np)

print(f"\nAbsolute Difference Statistics:")
print(f"  Mean:   {abs_diff.mean():.10f}")
print(f"  Median: {np.median(abs_diff):.10f}")
print(f"  Max:    {abs_diff.max():.10f}")
print(f"  Min:    {abs_diff.min():.10f}")
print(f"  Std:    {abs_diff.std():.10f}")

# Relative difference (where outputs are not near zero)
mask = np.abs(output_torch_np) > 1e-6
if mask.any():
    rel_diff = np.abs((output_torch_np[mask] - output_mlx_np[mask]) / output_torch_np[mask])
    print(f"\nRelative Difference Statistics (where |torch_out| > 1e-6):")
    print(f"  Mean:   {rel_diff.mean():.10f}")
    print(f"  Median: {np.median(rel_diff):.10f}")
    print(f"  Max:    {rel_diff.max():.10f}")

# Check if outputs are close
tolerances = [1e-4, 1e-5, 1e-6, 1e-7]
print(f"\nNumerical Closeness Check:")
for tol in tolerances:
    is_close = np.allclose(output_torch_np, output_mlx_np, rtol=tol, atol=tol)
    symbol = "✓" if is_close else "✗"
    print(f"  {symbol} rtol={tol:.0e}, atol={tol:.0e}: {is_close}")

# Element-wise comparison
print(f"\nElement-wise Analysis:")
print(f"  Total elements: {output_torch_np.size}")
for tol in [1e-4, 1e-5, 1e-6]:
    matching = np.sum(abs_diff < tol)
    percentage = 100 * matching / output_torch_np.size
    print(f"  Elements within {tol:.0e}: {matching}/{output_torch_np.size} ({percentage:.2f}%)")

# ============================================================================
# Sample Output Comparison
# ============================================================================

print(f"\n" + "-" * 80)
print("Sample Output Comparison (first 3 timesteps, first batch):")
print("-" * 80)

for t in range(min(3, seq_len)):
    print(f"\nTimestep {t}:")
    print(f"  PyTorch: [{output_torch_np[0, t, 0]:.8f}, {output_torch_np[0, t, 1]:.8f}]")
    print(f"  MLX:     [{output_mlx_np[0, t, 0]:.8f}, {output_mlx_np[0, t, 1]:.8f}]")
    print(f"  Diff:    [{abs_diff[0, t, 0]:.10f}, {abs_diff[0, t, 1]:.10f}]")

# ============================================================================
# Test with Different Input
# ============================================================================

print("\n" + "=" * 80)
print("Testing with Zero Input")
print("=" * 80)

zero_input = np.zeros((1, seq_len, input_dim), dtype=np.float32)
zero_input_torch = torch.FloatTensor(zero_input)
zero_input_mlx = mx.array(zero_input)

with torch.no_grad():
    zero_output_torch = model_torch(zero_input_torch).numpy()

zero_output_mlx = np.array(model_mlx(zero_input_mlx))

zero_diff = np.abs(zero_output_torch - zero_output_mlx)
print(f"Max difference: {zero_diff.max():.10f}")
print(f"Mean difference: {zero_diff.mean():.10f}")

# ============================================================================
# Test with Ones Input
# ============================================================================

print("\n" + "=" * 80)
print("Testing with Ones Input")
print("=" * 80)

ones_input = np.ones((1, seq_len, input_dim), dtype=np.float32)
ones_input_torch = torch.FloatTensor(ones_input)
ones_input_mlx = mx.array(ones_input)

with torch.no_grad():
    ones_output_torch = model_torch(ones_input_torch).numpy()

ones_output_mlx = np.array(model_mlx(ones_input_mlx))

ones_diff = np.abs(ones_output_torch - ones_output_mlx)
print(f"Max difference: {ones_diff.max():.10f}")
print(f"Mean difference: {ones_diff.mean():.10f}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Validation Summary")
print("=" * 80)

max_allowed_diff = 1e-5
is_valid = abs_diff.max() < max_allowed_diff

if is_valid:
    print(f"\n✓ VALIDATION PASSED")
    print(f"  Maximum difference ({abs_diff.max():.10f}) is below threshold ({max_allowed_diff})")
    print(f"  The implementations produce numerically equivalent results!")
else:
    print(f"\n✗ VALIDATION FAILED")
    print(f"  Maximum difference ({abs_diff.max():.10f}) exceeds threshold ({max_allowed_diff})")
    print(f"  The implementations may have differences in computation.")

print("\n" + "=" * 80)
