#!/usr/bin/env python3
"""
Convergence-based benchmark: Compare time to reach target loss thresholds.
This gives a fair comparison of training efficiency rather than fixed epochs.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)

print("=" * 80)
print("CONVERGENCE-BASED LTC Benchmark: Time to Target Loss")
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
max_epochs = 500  # Maximum epochs to prevent infinite loops
seq_len = 3
batch_size = 32

# Loss thresholds to test
loss_thresholds = [0.01, 0.005, 0.001, 0.0005, 0.0001]

print(f"\nConfiguration:")
print(f"  Loss thresholds: {loss_thresholds}")
print(f"  Max epochs: {max_epochs}")
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

# Generate data once for both implementations
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
# Convergence-based Training Functions
# ============================================================================

def train_pytorch_to_convergence(target_loss):
    """Train PyTorch model until target loss is reached"""
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
    epoch = 0
    losses = []

    while epoch < max_epochs:
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

        epoch += 1
        losses.append(total_loss)

        # Check convergence
        if total_loss <= target_loss:
            break

        if epoch % 50 == 0:
            print(f"    PyTorch Epoch {epoch}, Loss: {total_loss:.6f}, Target: {target_loss:.6f}")

    train_time = time.time() - start_time

    # Final validation
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.FloatTensor(all_inputs).unsqueeze(0))
        val_loss = criterion(val_pred, torch.FloatTensor(all_targets).unsqueeze(0))

    converged = total_loss <= target_loss

    return {
        'converged': converged,
        'epochs': epoch,
        'train_time': train_time,
        'final_train_loss': total_loss,
        'final_val_loss': val_loss.item(),
        'losses': losses,
        'predictions': val_pred.squeeze(0).numpy()
    }

def train_mlx_to_convergence(target_loss):
    """Train MLX model until target loss is reached"""
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as mlx_optim
    from ltc_mlx_fixed import RandomWiring, LTCRNN

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
    epoch = 0
    losses = []

    while epoch < max_epochs:
        total_loss = 0

        for start_idx, end_idx in batch_indices:
            x_batch = mx_train_inputs[start_idx:end_idx]
            y_batch = mx_train_targets[start_idx:end_idx]

            loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
            optimizer.update(model, grads)

            total_loss += loss.item()

        epoch += 1
        losses.append(total_loss)

        # Check convergence
        if total_loss <= target_loss:
            break

        if epoch % 50 == 0:
            print(f"    MLX Epoch {epoch}, Loss: {total_loss:.6f}, Target: {target_loss:.6f}")

    # Final evaluation
    mx.eval(model.parameters(), optimizer.state)
    train_time = time.time() - start_time

    # Final validation
    val_pred = model(mx_all_inputs[None, :, :])
    val_loss = mse_loss(val_pred, mx_all_targets[None, :, :])
    mx.eval(val_pred, val_loss)

    converged = total_loss <= target_loss

    return {
        'converged': converged,
        'epochs': epoch,
        'train_time': train_time,
        'final_train_loss': total_loss,
        'final_val_loss': val_loss.item(),
        'losses': losses,
        'predictions': np.array(val_pred[0])
    }

# ============================================================================
# Run Convergence Benchmark
# ============================================================================

print("\n" + "=" * 80)
print("Running Convergence Benchmark")
print("=" * 80)

results = {
    'pytorch': {},
    'mlx': {}
}

for threshold in loss_thresholds:
    print(f"\n🎯 Target Loss: {threshold:.6f}")
    print("-" * 50)

    # PyTorch
    print("  Training PyTorch...")
    pytorch_result = train_pytorch_to_convergence(threshold)
    results['pytorch'][threshold] = pytorch_result

    if pytorch_result['converged']:
        print(f"    ✅ Converged in {pytorch_result['epochs']} epochs ({pytorch_result['train_time']:.2f}s)")
    else:
        print(f"    ❌ Did not converge (max epochs reached: {pytorch_result['epochs']})")

    # MLX
    print("  Training MLX...")
    mlx_result = train_mlx_to_convergence(threshold)
    results['mlx'][threshold] = mlx_result

    if mlx_result['converged']:
        print(f"    ✅ Converged in {mlx_result['epochs']} epochs ({mlx_result['train_time']:.2f}s)")
    else:
        print(f"    ❌ Did not converge (max epochs reached: {mlx_result['epochs']})")

    # Comparison for this threshold
    if pytorch_result['converged'] and mlx_result['converged']:
        speedup = pytorch_result['train_time'] / mlx_result['train_time']
        epoch_ratio = pytorch_result['epochs'] / mlx_result['epochs']
        print(f"    📊 MLX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} to converge")
        print(f"    📊 MLX needs {epoch_ratio:.2f}x {'fewer' if epoch_ratio > 1 else 'more'} epochs")

# ============================================================================
# Analysis and Visualization
# ============================================================================

print("\n" + "=" * 80)
print("Convergence Analysis Summary")
print("=" * 80)

# Create summary table
print(f"\n{'Loss Target':<12} | {'PyTorch':<20} | {'MLX':<20} | {'Speedup':<10}")
print("-" * 70)

convergence_data = {
    'thresholds': [],
    'pytorch_times': [],
    'mlx_times': [],
    'pytorch_epochs': [],
    'mlx_epochs': [],
    'speedups': []
}

for threshold in loss_thresholds:
    pt_result = results['pytorch'][threshold]
    mlx_result = results['mlx'][threshold]

    if pt_result['converged'] and mlx_result['converged']:
        speedup = pt_result['train_time'] / mlx_result['train_time']

        convergence_data['thresholds'].append(threshold)
        convergence_data['pytorch_times'].append(pt_result['train_time'])
        convergence_data['mlx_times'].append(mlx_result['train_time'])
        convergence_data['pytorch_epochs'].append(pt_result['epochs'])
        convergence_data['mlx_epochs'].append(mlx_result['epochs'])
        convergence_data['speedups'].append(speedup)

        print(f"{threshold:<12.6f} | {pt_result['train_time']:>8.2f}s ({pt_result['epochs']:>3d} ep) | {mlx_result['train_time']:>8.2f}s ({mlx_result['epochs']:>3d} ep) | {speedup:>8.2f}x")
    else:
        pt_status = "✅" if pt_result['converged'] else "❌"
        mlx_status = "✅" if mlx_result['converged'] else "❌"
        print(f"{threshold:<12.6f} | {pt_status:<20} | {mlx_status:<20} | {'N/A':<10}")

# Overall efficiency analysis
if convergence_data['speedups']:
    avg_speedup = np.mean(convergence_data['speedups'])
    print(f"\n🚀 Average MLX speedup to convergence: {avg_speedup:.2f}x")

    if avg_speedup > 1:
        print("🎉 MLX is faster overall when considering convergence efficiency!")
    else:
        print("⚡ PyTorch is still faster even when considering convergence.")

# ============================================================================
# Visualization
# ============================================================================

if convergence_data['thresholds']:
    print("\nGenerating convergence analysis plots...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Time to convergence
    ax1.loglog(convergence_data['thresholds'], convergence_data['pytorch_times'], 'r-o', label='PyTorch', markersize=8)
    ax1.loglog(convergence_data['thresholds'], convergence_data['mlx_times'], 'b-s', label='MLX', markersize=8)
    ax1.set_xlabel('Target Loss')
    ax1.set_ylabel('Time to Convergence (s)')
    ax1.set_title('Time to Reach Target Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Plot 2: Epochs to convergence
    ax2.loglog(convergence_data['thresholds'], convergence_data['pytorch_epochs'], 'r-o', label='PyTorch', markersize=8)
    ax2.loglog(convergence_data['thresholds'], convergence_data['mlx_epochs'], 'b-s', label='MLX', markersize=8)
    ax2.set_xlabel('Target Loss')
    ax2.set_ylabel('Epochs to Convergence')
    ax2.set_title('Epochs to Reach Target Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Plot 3: Speedup vs target loss
    ax3.semilogx(convergence_data['thresholds'], convergence_data['speedups'], 'g-o', linewidth=2, markersize=8)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Target Loss')
    ax3.set_ylabel('MLX Speedup (>1 = MLX faster)')
    ax3.set_title('MLX Convergence Speedup')
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # Plot 4: Training curves comparison for best case
    best_threshold_idx = np.argmax(convergence_data['speedups'])
    best_threshold = convergence_data['thresholds'][best_threshold_idx]

    pytorch_losses = results['pytorch'][best_threshold]['losses']
    mlx_losses = results['mlx'][best_threshold]['losses']

    ax4.semilogy(range(len(pytorch_losses)), pytorch_losses, 'r-', label=f'PyTorch', alpha=0.7)
    ax4.semilogy(range(len(mlx_losses)), mlx_losses, 'b-', label=f'MLX', alpha=0.7)
    ax4.axhline(y=best_threshold, color='green', linestyle='--', alpha=0.7, label=f'Target: {best_threshold:.6f}')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training Loss')
    ax4.set_title(f'Training Curves (Target: {best_threshold:.6f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    print("Convergence analysis saved to: convergence_analysis.png")
    plt.show()

print("\n" + "=" * 80)
print("Convergence Benchmark Complete!")
print("=" * 80)