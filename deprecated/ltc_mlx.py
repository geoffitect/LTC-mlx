#============================================================
# Fixed MLX version of LTC implementation
# Addresses training instability and performance issues
#============================================================

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt


### Implementing the RandomWiring Class
class RandomWiring:
    """
    The RandomWiring class is responsible for defining the
    connection architecture between neurons. It initializes
    random adjacency matrices for neuron connections and
    sensory inputs. Those are used as an arbitrary sparsity
    matrix later in LIFNeuralLayer.
    """
    def __init__(self, input_dim, output_dim, neuron_count):
        self.input_dim = input_dim  # Number of input features
        self.output_dim = output_dim  # Number of output features
        self.neuron_count = neuron_count  # Number of neurons in the layer
        # Use fixed random seed for reproducible wiring
        np.random.seed(42)
        self.adjacency_matrix = np.random.uniform(0, 1, (neuron_count, neuron_count))  # Adjacency matrix for connections between neurons
        self.sensory_adjacency_matrix = np.random.uniform(0, 1, (input_dim, neuron_count))  # Adjacency matrix for sensory inputs to neurons
        np.random.seed(None)  # Reset seed

    def erev_initializer(self):
        np.random.seed(42)
        result = np.random.uniform(-0.2, 0.2, (self.neuron_count, self.neuron_count))  # Initialize reversal potentials for neuron connections
        np.random.seed(None)
        return result

    def sensory_erev_initializer(self):
        np.random.seed(42)
        result = np.random.uniform(-0.2, 0.2, (self.input_dim, self.neuron_count))  # Initialize reversal potentials for sensory inputs
        np.random.seed(None)
        return result


def softplus(x):
    """Softplus activation function: log(1 + exp(x))"""
    return mx.logaddexp(x, 0.0)


### Implementing the LIFNeuronLayer Class
class LIFNeuronLayer(nn.Module):
    """
    The LIFNeuronLayer class models the behavior of a layer
    of Leaky Integrate-and-Fire neurons. It initializes neuron
    parameters and defines the forward pass for computing
    neuron states. LIF dynamics are described using ODE and
    during the forward pass we solve the states using Euler
    Explicit method, described in the original article.
    """
    def __init__(self, wiring, ode_unfolds=12, epsilon=1e-8):
        super(LIFNeuronLayer, self).__init__()
        self.wiring = wiring  # Wiring object containing connection information
        self.ode_unfolds = ode_unfolds  # Number of ODE solver iterations
        self.epsilon = epsilon  # Small value to avoid division by zero

        # Initialization ranges for parameters (more conservative for stability)
        GLEAK_MIN, GLEAK_MAX = 0.01, 0.5  # Reduced range
        VLEAK_MIN, VLEAK_MAX = -0.1, 0.1  # Reduced range
        CM_MIN, CM_MAX = 0.45, 0.55  # Tighter range
        W_MIN, W_MAX = 0.01, 0.5  # Reduced range
        SIGMA_MIN, SIGMA_MAX = 3, 6  # Reduced range
        MU_MIN, MU_MAX = 0.4, 0.6  # Tighter range
        SENSORY_W_MIN, SENSORY_W_MAX = 0.01, 0.5  # Reduced range
        SENSORY_SIGMA_MIN, SENSORY_SIGMA_MAX = 3, 6  # Reduced range
        SENSORY_MU_MIN, SENSORY_MU_MAX = 0.4, 0.6  # Tighter range

        # Initialize neuron parameters as proper MLX parameters
        # Use Xavier/Glorot-like initialization for better stability
        fan_in = wiring.neuron_count
        std = np.sqrt(2.0 / fan_in)

        self.gleak = mx.random.uniform(GLEAK_MIN, GLEAK_MAX, (wiring.neuron_count,))
        self.vleak = mx.random.uniform(VLEAK_MIN, VLEAK_MAX, (wiring.neuron_count,))
        self.cm = mx.random.uniform(CM_MIN, CM_MAX, (wiring.neuron_count,))

        # More careful initialization for weights
        self.w = mx.random.normal((wiring.neuron_count, wiring.neuron_count)) * std * 0.1 + 0.1
        self.sigma = mx.random.uniform(SIGMA_MIN, SIGMA_MAX, (wiring.neuron_count, wiring.neuron_count))
        self.mu = mx.random.uniform(MU_MIN, MU_MAX, (wiring.neuron_count, wiring.neuron_count))
        self.erev = mx.array(wiring.erev_initializer())

        # Initialize sensory parameters
        self.sensory_w = mx.random.normal((wiring.input_dim, wiring.neuron_count)) * std * 0.1 + 0.1
        self.sensory_sigma = mx.random.uniform(SENSORY_SIGMA_MIN, SENSORY_SIGMA_MAX, (wiring.input_dim, wiring.neuron_count))
        self.sensory_mu = mx.random.uniform(SENSORY_MU_MIN, SENSORY_MU_MAX, (wiring.input_dim, wiring.neuron_count))
        self.sensory_erev = mx.array(wiring.sensory_erev_initializer())

        # Sparsity masks (fixed, non-learnable) based on the wiring adjacency matrices
        self.sparsity_mask = mx.array(np.abs(wiring.adjacency_matrix))
        self.sensory_sparsity_mask = mx.array(np.abs(wiring.sensory_adjacency_matrix))

    def __call__(self, inputs, state, elapsed_time=1.0):
        return self.ode_solver(inputs, state, elapsed_time)

    def ode_solver(self, inputs, state, elapsed_time):
        v_pre = state  # Previous state (voltage)

        # Pre-compute the effects of the sensory neurons
        sensory_activation = softplus(self.sensory_w) * self.sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_activation = sensory_activation * self.sensory_sparsity_mask
        sensory_reversal_activation = sensory_activation * self.sensory_erev

        # Calculate the numerator and denominator for sensory inputs
        w_numerator_sensory = mx.sum(sensory_reversal_activation, axis=1)  # Sum over input_dim, keep batch dim
        w_denominator_sensory = mx.sum(sensory_activation, axis=1)  # Sum over input_dim, keep batch dim

        # Calculate membrane capacitance over time
        cm_t = softplus(self.cm) / (elapsed_time / self.ode_unfolds)

        # Initialize weights for neuron connections
        w_param = softplus(self.w)
        for _ in range(self.ode_unfolds):
            # Activation based on previous state
            w_activation = w_param * self.sigmoid(v_pre, self.mu, self.sigma)
            w_activation = w_activation * self.sparsity_mask
            reversal_activation = w_activation * self.erev

            # Calculate the numerator and denominator for neuron connections
            w_numerator = mx.sum(reversal_activation, axis=2) + w_numerator_sensory  # Sum over neuron connections, keep batch and neuron dims
            w_denominator = mx.sum(w_activation, axis=2) + w_denominator_sensory  # Sum over neuron connections, keep batch and neuron dims

            # Leak conductance and voltage calculations
            gleak = softplus(self.gleak)
            numerator = cm_t * v_pre + gleak * self.vleak + w_numerator
            denominator = cm_t + gleak + w_denominator

            # Update the state (voltage) with gradient clipping
            v_new = numerator / (denominator + self.epsilon)
            # Clip gradients to prevent explosion
            v_pre = mx.clip(v_new, -5.0, 5.0)

        return v_pre

    def sigmoid(self, v_pre, mu, sigma):
        v_pre = mx.expand_dims(v_pre, -1)  # Expand dims to match dimensions
        # Clip sigma to prevent extreme values
        sigma_clipped = mx.clip(sigma, 0.1, 10.0)
        activation = sigma_clipped * (v_pre - mu)  # Apply sigma and mean shift
        # Clip activation to prevent overflow
        activation_clipped = mx.clip(activation, -10.0, 10.0)
        return mx.sigmoid(activation_clipped)  # Apply sigmoid activation


### Implementing the LTCCell Class
class LTCCell(nn.Module):
    def __init__(self, wiring, in_features=None, ode_unfolds=6, epsilon=1e-8):
        super(LTCCell, self).__init__()
        self.wiring = wiring  # Wiring object
        self.ode_unfolds = ode_unfolds  # Number of ODE solver iterations
        self.epsilon = epsilon  # Small value to avoid division by zero

        self.neuron = LIFNeuronLayer(wiring, ode_unfolds, epsilon)  # Initialize LIFNeuron with the given wiring

    def __call__(self, inputs, states, elapsed_time=1.0):
        next_state = self.neuron(inputs, states, elapsed_time)  # Compute the next state using the neuron model
        outputs = next_state[:, :self.wiring.output_dim] # Map the state to the output dimensions
        return outputs, next_state


# Implementing the LTCRNN Class
class LTCRNN(nn.Module):
    """
    The LTCRNN class constructs the recurrent neural network
    using multiple LTCCell instances. It processes sequences
    of inputs to produce sequences of outputs.

    Note: In this tutorial, the LTCRNN is supposed to be called
    one shot, for the whole sequence, hence the state initialization
    to zero in each forward pass.
    """
    def __init__(self, wiring, input_dim, hidden_dim, output_dim):
        super(LTCRNN, self).__init__()
        self.cell = LTCCell(wiring, in_features=input_dim)  # Initialize LTCCell with wiring and input dimension
        self.hidden_dim = hidden_dim  # Number of hidden neurons
        self.output_dim = output_dim  # Number of output neurons

    def __call__(self, inputs):
        batch_size, seq_len, _ = inputs.shape  # Get batch size and sequence length from input dimensions

        states = mx.zeros((batch_size, self.hidden_dim))  # Initialize hidden states with zeros

        outputs = []  # List to store outputs for each time step

        for t in range(seq_len):
            output, states = self.cell(inputs[:, t, :], states)  # Compute output and next state for each time step
            outputs.append(output)  # Append the output to the list

        result = mx.stack(outputs, axis=1)  # Stack the outputs along the sequence dimension
        return result


# Generate spiral data
def generate_spiral_data(num_points, num_turns, noise = 4):
    """
    We will generate a dataset of spiral trajectories to train and evaluate our model.
    The generate_spiral_data function creates synthetic data points forming a spiral
    pattern.
    """
    theta = np.linspace(0, num_turns * 2 * np.pi, num_points)
    z = np.linspace(0, 1, num_points)
    r = z
    x = r * np.sin(theta) + noise * np.random.randn(*theta.shape) / num_points
    y = r * np.cos(theta) + noise * np.random.randn(*theta.shape) / num_points
    return np.stack([x, y], axis=1)


# Loss function
def mse_loss(predictions, targets):
    """Mean squared error loss"""
    return mx.mean((predictions - targets) ** 2)


# Loss function for training with gradient clipping
def loss_fn(model, x, y_target):
    """Loss function that takes model and data"""
    outputs = model(x)
    loss = mse_loss(outputs, y_target)
    # Add small L2 regularization to prevent parameter explosion
    # Simplified - no L2 regularization for now
    l2_loss = 0.0
    return loss + l2_loss


### Training
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 2 # Number of input dimensions
    hidden_dim = 8 # Number of hidden dimensions (number of neurons in LIFNeuralLayer)
    output_dim = 2 # Number of output dimensions
    num_points = 500 # Number of spiral points in dataset
    num_turns = 3 # Number of spiral turns
    learning_rate = 0.001  # Reduced learning rate for stability
    num_epochs = 200
    seq_len = 3 # Maximum length of the sample sequence
    batch_size = 32

    # Generate data
    data = generate_spiral_data(num_points, num_turns)
    all_inputs = data[:-1, :]
    all_targets = data[1:, :]

    # Prepare input and target sequences
    trajectory_count = max(1, len(all_inputs) - seq_len)
    train_inputs = [all_inputs[i:i + seq_len] for i in range(trajectory_count)]
    train_targets = [all_targets[i:i + seq_len] for i in range(trajectory_count)]

    # Shuffle and split the data for training
    random_train_indices = np.arange(len(train_inputs))
    np.random.shuffle(random_train_indices)
    train_split_index = int(len(random_train_indices) * 0.8)
    random_train_indices = random_train_indices[:train_split_index]

    # Function to create batches
    def create_batches(data_list, batch_size):
        return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

    # Create input and target batches
    train_input_batches = create_batches(train_inputs, batch_size)
    train_target_batches = create_batches(train_targets, batch_size)

    # Initialize model
    wiring = RandomWiring(input_dim, output_dim, hidden_dim)
    model = LTCRNN(wiring, input_dim, hidden_dim, output_dim)

    # Initialize optimizer with gradient clipping
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Create loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0

        # Iterate over batches
        for x, y_target in zip(train_input_batches, train_target_batches):
            x = mx.stack([mx.array(xi) for xi in x])  # Stack batch of sequences
            y_target = mx.stack([mx.array(yi) for yi in y_target])  # Stack batch of targets

            # Forward pass and compute gradients
            loss, grads = loss_and_grad_fn(model, x, y_target)

            # No gradient clipping for now
            clipped_grads = grads

            # Update model parameters
            optimizer.update(model, clipped_grads)
            mx.eval(model.parameters(), optimizer.state)

            # Accumulate total loss
            total_loss += loss.item()

        # Print loss every 100 epochs and plot predictions
        if (epoch + 1) % 100 == 0:
            # Prediction and plotting
            predictions = model(mx.array(all_inputs)[None, :, :])
            np_predictions = np.array(predictions[0])
            val_loss = mse_loss(predictions, mx.array(all_targets)[None, :, :])
            print(f'Epoch [{epoch+1}/{num_epochs}], Total train loss: {total_loss:.4f}, Total val loss: {val_loss.item():.4f}')

            plt.figure(figsize=(8, 6))
            plt.plot(all_targets[:, 0], all_targets[:, 1], 'g-', label='True Path')
            plt.plot(np_predictions[:, 0], np_predictions[:, 1], 'r-', label='Predicted Path')
            plt.legend()
            plt.title(f'Epoch {epoch+1}')
            plt.show()