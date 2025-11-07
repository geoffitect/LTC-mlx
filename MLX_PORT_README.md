# LTC Network: PyTorch to MLX Port

This directory contains a complete port of the Liquid Time-Constant (LTC) neural network from PyTorch to Apple's MLX framework.

## Files Created

### Core Implementation
- **`ltc.py`** - Original PyTorch implementation
- **`ltc_mlx.py`** - Complete MLX port with identical architecture

### Testing & Benchmarking
- **`validate_accuracy.py`** - Validates numerical equivalence between implementations
- **`benchmark.py`** - Comprehensive speed and accuracy comparison

## Key Changes in MLX Port

### API Differences
| PyTorch | MLX |
|---------|-----|
| `nn.Module` | `nn.Module` (similar) |
| `nn.Parameter()` | Direct array assignment |
| `torch.Tensor` | `mx.array` |
| `torch.sigmoid()` | `mx.sigmoid()` |
| `nn.Softplus()` | Custom `softplus()` function |
| `.forward()` | `.__call__()` |
| `.backward()` | `value_and_grad()` |
| `torch.optim.Adam` | `mlx.optimizers.Adam` |

### Architecture Components

All components have been ported:
1. **RandomWiring** - Defines sparse connection architecture
2. **LIFNeuronLayer** - Leaky Integrate-and-Fire neurons with ODE solver
3. **LTCCell** - Single cell wrapper
4. **LTCRNN** - Full recurrent network

### Notable Implementation Details

- **Softplus activation**: Implemented as `mx.logaddexp(x, 0.0)` in MLX
- **Parameter initialization**: MLX uses `mx.random.uniform()` instead of `torch.rand()`
- **Dimension operations**: `torch.unsqueeze()` → `mx.expand_dims()`
- **Training loop**: Uses `value_and_grad()` for automatic differentiation
- **Evaluation**: Uses `mx.eval()` to materialize lazy computations

## System Requirements

### Hardware
**MLX requires Apple Silicon (M1/M2/M3/M4)** - It will NOT work on:
- Intel Macs
- Linux systems
- Windows systems

### Software
```bash
pip install mlx mlx-nn torch numpy matplotlib
```

## Usage

### Run Validation (checks numerical equivalence)
```bash
python3 validate_accuracy.py
```

This script:
- Initializes both models with identical weights
- Runs forward pass on test data
- Compares outputs element-wise
- Reports numerical differences

Expected output: Differences should be < 1e-5 (floating point precision)

### Run Benchmark (speed comparison)
```bash
python3 benchmark.py
```

This script:
- Trains both models for 50 epochs
- Measures training time
- Measures inference time
- Compares final accuracy
- Generates comparison plots

Expected results: MLX should be significantly faster on Apple Silicon, especially for larger models.

### Train Individual Models

PyTorch version:
```bash
python3 ltc.py
```

MLX version:
```bash
python3 ltc_mlx.py
```

Both scripts train on spiral trajectory data and visualize results every 100 epochs.

## Performance Expectations

On Apple Silicon, you should expect:
- **Training**: MLX typically 2-5x faster than PyTorch
- **Inference**: MLX typically 3-10x faster than PyTorch
- **Memory**: MLX uses unified memory more efficiently
- **Accuracy**: Virtually identical (differences < 1e-5)

Actual speedups depend on:
- Model size (larger = better MLX advantage)
- Batch size
- Sequence length
- Specific Apple Silicon chip (M1 vs M2 vs M3/M4)

## Model Architecture

The LTC network uses:
- **Input dim**: 2 (x, y coordinates)
- **Hidden dim**: 8 neurons
- **Output dim**: 2 (predicted next position)
- **ODE unfolds**: 6-12 iterations per timestep
- **Activation**: Sigmoid with learnable μ and σ

### Learnable Parameters
Per neuron:
- `gleak`: Leak conductance
- `vleak`: Leak voltage
- `cm`: Membrane capacitance
- `w`: Connection weights
- `sigma`: Sigmoid steepness
- `mu`: Sigmoid center
- `erev`: Reversal potential

Plus sensory (input) versions of w, sigma, mu, and erev.

## Known Limitations

1. **Platform**: MLX only on Apple Silicon
2. **Maturity**: MLX is newer than PyTorch, some ops may differ slightly
3. **Ecosystem**: PyTorch has more third-party tools/libraries
4. **Plotting**: matplotlib.show() may require GUI backend on macOS

## Testing Notes

To verify the port is correct:

1. **Numerical Validation** (`validate_accuracy.py`)
   - Copy weights from PyTorch to MLX
   - Compare outputs on same input
   - Check differences are < 1e-5

2. **Training Convergence** (`benchmark.py`)
   - Both should reach similar loss values
   - Loss curves should follow similar trajectories
   - Final predictions should be visually similar

3. **Edge Cases** (in validation script)
   - Zero input
   - Ones input
   - Random input

## Next Steps

### Optimization Opportunities
1. **Batch size tuning**: MLX may benefit from different batch sizes
2. **Mixed precision**: MLX supports float16 for faster training
3. **JIT compilation**: MLX compiles graphs automatically
4. **Vectorization**: Some loops could be vectorized further

### Extensions
1. **Larger models**: Test with more neurons (32, 64, 128)
2. **Longer sequences**: Test with longer spiral trajectories
3. **Different tasks**: Try classification, time series forecasting
4. **Hardware comparison**: Test on M1 vs M2 vs M3/M4

## References

- **Original Paper**: [Liquid Time-Constant Networks](https://arxiv.org/abs/2006.04439)
- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **PyTorch Source**: git:/KPEKEP/LTCtutorial

## License

MIT License (matches original repository)
