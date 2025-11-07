#!/usr/bin/env python3
"""
Create an optimized CoreML model by training PyTorch to convergence first.
This ensures we have a properly trained model for fair comparison.
"""

import time
import numpy as np
import torch
import torch.nn as nn

print("=" * 80)
print("Creating Optimized CoreML Model from Trained PyTorch")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

# Hyperparameters
input_dim = 2
hidden_dim = 8
output_dim = 2
num_points = 500
num_turns = 3
learning_rate = 0.005
target_loss = 0.005  # Train to good convergence
max_epochs = 300
seq_len = 3
batch_size = 32

print(f"\nTraining Configuration:")
print(f"  Target loss: {target_loss}")
print(f"  Max epochs: {max_epochs}")
print(f"  Learning rate: {learning_rate}")

# ============================================================================
# Data Preparation
# ============================================================================

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

# Create optimized batches
selected_inputs = [train_inputs[i] for i in random_train_indices]
selected_targets = [train_targets[i] for i in random_train_indices]
np_train_inputs = np.stack(selected_inputs)
np_train_targets = np.stack(selected_targets)

num_samples = len(np_train_inputs)
batch_indices = [(i, min(i + batch_size, num_samples)) for i in range(0, num_samples, batch_size)]

print(f"Dataset prepared: {len(batch_indices)} batches")

# ============================================================================
# Train PyTorch Model to Convergence
# ============================================================================

print(f"\n🔥 Training PyTorch Model to Loss < {target_loss}")
print("-" * 50)

from ltc_coreml import RandomWiring, LTCRNN

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Initialize model
wiring = RandomWiring(input_dim, output_dim, hidden_dim)
model = LTCRNN(wiring, input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
start_time = time.time()
best_loss = float('inf')
converged = False

for epoch in range(max_epochs):
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

    # Check convergence
    if total_loss < target_loss:
        print(f"  ✅ Converged at epoch {epoch+1}! Loss: {total_loss:.6f}")
        converged = True
        break

    if total_loss < best_loss:
        best_loss = total_loss

    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1}/{max_epochs}, Loss: {total_loss:.6f} (Best: {best_loss:.6f})")

training_time = time.time() - start_time

if not converged:
    print(f"  ⚠️  Did not fully converge, but reached {total_loss:.6f} in {max_epochs} epochs")

print(f"\n📊 Training Results:")
print(f"  Final loss: {total_loss:.6f}")
print(f"  Training time: {training_time:.2f}s")

# ============================================================================
# Convert to CoreML
# ============================================================================

print(f"\n🍎 Converting to CoreML")
print("-" * 30)

# Set model to evaluation mode
model.eval()

try:
    import coremltools as ct

    # Create example input for tracing
    example_input = torch.rand(1, seq_len, input_dim)
    print(f"Example input shape: {example_input.shape}")

    # Trace the model
    print("Tracing PyTorch model...")
    traced_model = torch.jit.trace(model, example_input)

    # Convert to Core ML
    print("Converting to CoreML...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)],
        source="pytorch",
        convert_to="mlprogram"
    )

    # Save the model
    output_path = "ltc_model_optimized.mlpackage"
    coreml_model.save(output_path)
    print(f"✅ CoreML model saved to: {output_path}")

    # Test the converted model
    print("\n🧪 Testing CoreML Model")
    print("-" * 25)

    # Test inference
    test_input = all_inputs.astype(np.float32)
    coreml_input = {"inputs": test_input[None, :, :]}  # Add batch dimension

    # Get predictions
    coreml_predictions = coreml_model.predict(coreml_input)
    output_key = list(coreml_predictions.keys())[0]
    predictions = coreml_predictions[output_key]

    if len(predictions.shape) == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]  # Remove batch dimension

    # Compare with PyTorch
    model.eval()
    with torch.no_grad():
        pytorch_predictions = model(torch.FloatTensor(test_input).unsqueeze(0))
        pytorch_predictions = pytorch_predictions.squeeze(0).numpy()

    # Calculate errors
    coreml_loss = np.mean((predictions - all_targets) ** 2)
    pytorch_loss = np.mean((pytorch_predictions - all_targets) ** 2)
    conversion_error = np.mean((predictions - pytorch_predictions) ** 2)

    print(f"📊 Model Comparison:")
    print(f"  PyTorch validation loss:  {pytorch_loss:.6f}")
    print(f"  CoreML validation loss:   {coreml_loss:.6f}")
    print(f"  Conversion error:         {conversion_error:.6f}")

    if conversion_error < 1e-4:
        print("  ✅ Conversion successful - models match!")
    elif conversion_error < 1e-2:
        print("  ⚠️  Small conversion differences detected")
    else:
        print("  ❌ Large conversion differences - check conversion")

    # Timing test
    print(f"\n⚡ CoreML Inference Speed Test")
    print("-" * 35)

    # Warmup
    for _ in range(10):
        _ = coreml_model.predict(coreml_input)

    # Timing
    num_tests = 1000
    start_time = time.time()
    for _ in range(num_tests):
        _ = coreml_model.predict(coreml_input)
    avg_time = (time.time() - start_time) / num_tests

    print(f"  Average inference time: {avg_time*1000:.3f}ms")
    print(f"  Throughput: {1/avg_time:.0f} inferences/second")

except ImportError:
    print("❌ CoreML Tools not installed. Install with: pip install coremltools")
except Exception as e:
    print(f"❌ CoreML conversion failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print("✅ Optimized CoreML Model Creation Complete!")
print("=" * 80)