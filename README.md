# Liquid Time-Constant LNN Implementation from scratch

This repository contains an implementation of Liquid Time-Constant Neural Networks (LTC-LNNs) using PyTorch from scratch.
The components have been transferred from a Jupyter notebook to a regular .py file for ease of refactoring with Code CLI.

The implementation is based on the concept of Liquid Time-Constant Networks, which are a type of continuous-time recurrent neural network.

## Getting Started

### Prerequisites

To run this project, you will need to have the following libraries installed:

- PyTorch
- NumPy
- Matplotlib

You can install these libraries using `pip`:

```bash
pip install torch numpy matplotlib
```

## Files
- **ltc.py**: complete script with example training on spiral trajectories

## Acknowledgements

This project was inspired and built upon the following resources:

- [Neural Circuit Policies](https://github.com/mlech26l/ncps/)
- [Liquid Time-Constant Networks](https://github.com/raminmh/liquid_time_constant_networks)

## References

For further details on Liquid Time-Constant Networks and related models, please refer to the following articles:

- [Liquid Time-Constant Networks on Arxiv](https://arxiv.org/abs/2006.04439)
- [Closed-form Continuous-Time Network models](https://arxiv.org/abs/2106.13898)
- [LTC-SE](https://arxiv.org/abs/2304.08691)
- [Liquid-S4](https://arxiv.org/abs/2209.12951)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.