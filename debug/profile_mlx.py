#============================================================
# MLX Performance Profiler - Find bottlenecks
#============================================================

import time
import numpy as np
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim

def profile_operation(name, func, *args, **kwargs):
    """Profile a single operation"""
    start = time.time()
    result = func(*args, **kwargs)
    if hasattr(result, 'item'):  # Force evaluation for MLX arrays
        try:
            mx.eval(result)
        except:
            pass
    end = time.time()
    print(f"  {name}: {(end-start)*1000:.2f}ms")
    return result

def profile_mlx_bottlenecks():
    """Profile MLX operations to find bottlenecks"""
    print("🔍 MLX Performance Profiling")
    print("=" * 50)

    # Setup
    batch_size = 32
    seq_len = 3
    input_dim = 2
    hidden_dim = 8

    # Test data
    np_data = [np.random.randn(seq_len, input_dim) for _ in range(batch_size)]

    print("\n1. Data Conversion Overhead:")

    # Method 1: Individual conversions (current approach)
    start = time.time()
    mx_arrays = [mx.array(xi) for xi in np_data]
    mx_stacked = mx.stack(mx_arrays)
    mx.eval(mx_stacked)
    method1_time = time.time() - start
    print(f"  Individual + stack: {method1_time*1000:.2f}ms")

    # Method 2: Numpy stack first, then convert
    start = time.time()
    np_stacked = np.stack(np_data)
    mx_direct = mx.array(np_stacked)
    mx.eval(mx_direct)
    method2_time = time.time() - start
    print(f"  Numpy stack + convert: {method2_time*1000:.2f}ms")

    print(f"  Speedup: {method1_time/method2_time:.2f}x")

    print("\n2. Model Operations:")

    from ltc_mlx_fixed import RandomWiring, LTCRNN

    wiring = RandomWiring(input_dim, hidden_dim, hidden_dim)
    model = LTCRNN(wiring, input_dim, hidden_dim, hidden_dim)

    # Test input
    test_input = mx.random.normal((batch_size, seq_len, input_dim))
    test_target = mx.random.normal((batch_size, seq_len, hidden_dim))

    # Forward pass profiling
    print("\n  Forward pass breakdown:")
    start = time.time()

    def timed_forward():
        outputs = model(test_input)
        mx.eval(outputs)
        return outputs

    outputs = profile_operation("Forward pass", timed_forward)

    # Loss computation
    def mse_loss(pred, target):
        return mx.mean((pred - target) ** 2)

    loss = profile_operation("Loss computation", mse_loss, outputs, test_target)

    # Gradient computation
    def loss_fn(model, x, y):
        pred = model(x)
        return mse_loss(pred, y)

    loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

    def compute_grads():
        loss, grads = loss_and_grad_fn(model, test_input, test_target)
        mx.eval(loss)
        mx.eval(grads)
        return loss, grads

    loss, grads = profile_operation("Gradient computation", compute_grads)

    # Optimizer update
    optimizer = mlx_optim.Adam(learning_rate=0.001)

    def optimizer_update():
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    profile_operation("Optimizer update", optimizer_update)

    print("\n3. mx.eval() Impact:")

    # Without mx.eval
    start = time.time()
    for _ in range(10):
        outputs = model(test_input)
    no_eval_time = time.time() - start

    # With mx.eval
    start = time.time()
    for _ in range(10):
        outputs = model(test_input)
        mx.eval(outputs)
    with_eval_time = time.time() - start

    print(f"  Without eval (10 calls): {no_eval_time*1000:.2f}ms")
    print(f"  With eval (10 calls): {with_eval_time*1000:.2f}ms")
    print(f"  Eval overhead: {((with_eval_time - no_eval_time)/no_eval_time)*100:.1f}%")

    print("\n4. Memory Allocation Patterns:")

    # Reusing vs recreating tensors
    start = time.time()
    for _ in range(100):
        temp = mx.random.normal((batch_size, input_dim))
        mx.eval(temp)
    recreate_time = time.time() - start

    # Pre-allocate
    temp_tensor = mx.zeros((batch_size, input_dim))
    start = time.time()
    for _ in range(100):
        temp_tensor = mx.random.normal((batch_size, input_dim))
        mx.eval(temp_tensor)
    reuse_time = time.time() - start

    print(f"  Recreating tensors (100x): {recreate_time*1000:.2f}ms")
    print(f"  Reusing tensors (100x): {reuse_time*1000:.2f}ms")

def create_optimized_data_loader():
    """Create an optimized data loading approach"""
    print("\n🚀 Optimized Data Loading")
    print("=" * 50)

    # Generate test data
    num_points = 500
    seq_len = 3
    batch_size = 32

    from ltc_mlx_fixed import generate_spiral_data

    data = generate_spiral_data(num_points, 3)
    all_inputs = data[:-1, :]
    all_targets = data[1:, :]

    # Current approach (from benchmark)
    trajectory_count = max(1, len(all_inputs) - seq_len)
    train_inputs = [all_inputs[i:i + seq_len] for i in range(trajectory_count)]
    train_targets = [all_targets[i:i + seq_len] for i in range(trajectory_count)]

    def create_batches_old(data_list, batch_size):
        return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

    def create_batches_optimized(data_list, batch_size):
        """Optimized batch creation - convert to numpy first"""
        # Stack into numpy arrays first
        np_data = np.stack(data_list)

        # Create batches as views
        batches = []
        for i in range(0, len(data_list), batch_size):
            batch_data = np_data[i:i + batch_size]
            batches.append(batch_data)
        return batches

    # Time old approach
    start = time.time()
    old_batches = create_batches_old(train_inputs, batch_size)
    # Convert first batch to MLX (simulation)
    x_batch = old_batches[0]
    mx_old = mx.stack([mx.array(xi) for xi in x_batch])
    mx.eval(mx_old)
    old_time = time.time() - start

    # Time new approach
    start = time.time()
    new_batches = create_batches_optimized(train_inputs, batch_size)
    # Convert first batch to MLX
    mx_new = mx.array(new_batches[0])
    mx.eval(mx_new)
    new_time = time.time() - start

    print(f"Old approach: {old_time*1000:.2f}ms")
    print(f"New approach: {new_time*1000:.2f}ms")
    print(f"Speedup: {old_time/new_time:.2f}x")

if __name__ == "__main__":
    profile_mlx_bottlenecks()
    create_optimized_data_loader()