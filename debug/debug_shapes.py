#============================================================
# Debug shapes in MLX implementation
#============================================================

import mlx.core as mx
import numpy as np

def debug_shapes():
    from ltc_mlx_fixed import RandomWiring, LIFNeuronLayer

    # Create test configuration
    input_dim = 2
    hidden_dim = 8
    batch_size = 4

    # Create wiring and layer
    wiring = RandomWiring(input_dim, hidden_dim, hidden_dim)
    layer = LIFNeuronLayer(wiring)

    print("=== WIRING SHAPES ===")
    print(f"adjacency_matrix: {wiring.adjacency_matrix.shape}")
    print(f"sensory_adjacency_matrix: {wiring.sensory_adjacency_matrix.shape}")

    print("\n=== LAYER PARAMETER SHAPES ===")
    print(f"gleak: {layer.gleak.shape}")
    print(f"vleak: {layer.vleak.shape}")
    print(f"cm: {layer.cm.shape}")
    print(f"w: {layer.w.shape}")
    print(f"sigma: {layer.sigma.shape}")
    print(f"mu: {layer.mu.shape}")
    print(f"erev: {layer.erev.shape}")
    print(f"sensory_w: {layer.sensory_w.shape}")
    print(f"sensory_sigma: {layer.sensory_sigma.shape}")
    print(f"sensory_mu: {layer.sensory_mu.shape}")
    print(f"sensory_erev: {layer.sensory_erev.shape}")
    print(f"sparsity_mask: {layer.sparsity_mask.shape}")
    print(f"sensory_sparsity_mask: {layer.sensory_sparsity_mask.shape}")

    print("\n=== FORWARD PASS SHAPES ===")
    inputs = mx.random.normal((batch_size, input_dim))
    state = mx.zeros((batch_size, hidden_dim))

    print(f"inputs: {inputs.shape}")
    print(f"state: {state.shape}")

    # Step through the ode_solver to see where it breaks
    print("\n=== ODE SOLVER STEP BY STEP ===")
    v_pre = state
    print(f"v_pre: {v_pre.shape}")

    # Pre-compute the effects of the sensory neurons
    print(f"\nSensory computation:")
    print(f"inputs: {inputs.shape}")
    print(f"sensory_mu: {layer.sensory_mu.shape}")
    print(f"sensory_sigma: {layer.sensory_sigma.shape}")

    # Check sigmoid function
    v_expanded = mx.expand_dims(inputs, -1)
    print(f"v_expanded: {v_expanded.shape}")
    activation = layer.sensory_sigma * (v_expanded - layer.sensory_mu)
    print(f"activation: {activation.shape}")
    sigmoid_result = mx.sigmoid(activation)
    print(f"sigmoid_result: {sigmoid_result.shape}")

    # Continue with sensory activation
    sensory_w_softplus = mx.logaddexp(layer.sensory_w, 0.0)
    print(f"sensory_w_softplus: {sensory_w_softplus.shape}")

    sensory_activation = sensory_w_softplus * sigmoid_result
    print(f"sensory_activation: {sensory_activation.shape}")

    sensory_activation = sensory_activation * layer.sensory_sparsity_mask
    print(f"sensory_activation (after mask): {sensory_activation.shape}")

    sensory_reversal_activation = sensory_activation * layer.sensory_erev
    print(f"sensory_reversal_activation: {sensory_reversal_activation.shape}")

    # Calculate the numerator and denominator for sensory inputs
    w_numerator_sensory = mx.sum(sensory_reversal_activation, axis=0)
    w_denominator_sensory = mx.sum(sensory_activation, axis=0)
    print(f"w_numerator_sensory: {w_numerator_sensory.shape}")
    print(f"w_denominator_sensory: {w_denominator_sensory.shape}")

    print(f"\nNeuron connection computation:")
    # Initialize weights for neuron connections
    w_param = mx.logaddexp(layer.w, 0.0)
    print(f"w_param: {w_param.shape}")

    # Activation based on previous state
    v_expanded = mx.expand_dims(v_pre, -1)
    print(f"v_expanded for connections: {v_expanded.shape}")
    print(f"layer.mu: {layer.mu.shape}")
    print(f"layer.sigma: {layer.sigma.shape}")

    activation = layer.sigma * (v_expanded - layer.mu)
    print(f"connection activation: {activation.shape}")
    sigmoid_result = mx.sigmoid(activation)
    print(f"connection sigmoid_result: {sigmoid_result.shape}")

    w_activation = w_param * sigmoid_result
    print(f"w_activation: {w_activation.shape}")

    w_activation = w_activation * layer.sparsity_mask
    print(f"w_activation (after mask): {w_activation.shape}")

    reversal_activation = w_activation * layer.erev
    print(f"reversal_activation: {reversal_activation.shape}")

    # This is where the error occurs
    print(f"\nProblem area:")
    print(f"reversal_activation shape: {reversal_activation.shape}")
    print(f"Trying to sum over axis=1...")
    try:
        w_numerator = mx.sum(reversal_activation, axis=1)
        print(f"w_numerator: {w_numerator.shape}")
        print(f"w_numerator_sensory: {w_numerator_sensory.shape}")
        print("Broadcasting check:")
        result = w_numerator + w_numerator_sensory
        print(f"Addition result: {result.shape}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    debug_shapes()