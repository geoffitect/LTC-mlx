#============================================================
# Python version of Jupyter release: git:/KPEKEP/LTCtutorial
#============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


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
        self.adjacency_matrix = np.random.uniform(0, 1, (neuron_count, neuron_count))  # Adjacency matrix for connections between neurons
        self.sensory_adjacency_matrix = np.random.uniform(0, 1, (input_dim, neuron_count))  # Adjacency matrix for sensory inputs to neurons

    def erev_initializer(self):
        return np.random.uniform(-0.2, 0.2, (self.neuron_count, self.neuron_count))  # Initialize reversal potentials for neuron connections

    def sensory_erev_initializer(self):
        return np.random.uniform(-0.2, 0.2, (self.input_dim, self.neuron_count))  # Initialize reversal potentials for sensory inputs
    
### Implementing the LIFNeuronLayer Class
class LIFNeuronLayer(nn.Module):
    """
    The LIFNeuronLayer class models the behavior of a layer 
    of Leaky Integrate-and-Fire neurons. It initializes neuron 
    parameters and defines the forward pass for computing 
    neuron states. LIF dynamics are described using ODE and 
    during the forward pass we solve the states using Euler 
    Explicit method, described in the original article. Note
    potential to integrate other advanced samplers here, such
    as DPM(2++) and ancestrals.
    """
    def __init__(self, wiring, ode_unfolds=12, epsilon=1e-8):
        super(LIFNeuronLayer, self).__init__()
        self.wiring = wiring  # Wiring object containing connection information
        self.ode_unfolds = ode_unfolds  # Number of ODE solver iterations
        self.epsilon = epsilon  # Small value to avoid division by zero
        self.softplus = nn.Softplus()  # Softplus activation function

        # Initialization ranges for parameters
        GLEAK_MIN, GLEAK_MAX = 0.001, 1.0
        VLEAK_MIN, VLEAK_MAX = -0.2, 0.2
        CM_MIN, CM_MAX = 0.4, 0.6
        W_MIN, W_MAX = 0.001, 1.0
        SIGMA_MIN, SIGMA_MAX = 3, 8
        MU_MIN, MU_MAX = 0.3, 0.8
        SENSORY_W_MIN, SENSORY_W_MAX = 0.001, 1.0
        SENSORY_SIGMA_MIN, SENSORY_SIGMA_MAX = 3, 8
        SENSORY_MU_MIN, SENSORY_MU_MAX = 0.3, 0.8

        # Initialize neuron parameters with random values within specified ranges
        self.gleak = nn.Parameter(torch.rand(wiring.neuron_count) * (GLEAK_MAX - GLEAK_MIN) + GLEAK_MIN)
        self.vleak = nn.Parameter(torch.rand(wiring.neuron_count) * (VLEAK_MAX - VLEAK_MIN) + VLEAK_MIN)
        self.cm = nn.Parameter(torch.rand(wiring.neuron_count) * (CM_MAX - CM_MIN) + CM_MIN)
        self.w = nn.Parameter(torch.rand(wiring.neuron_count, wiring.neuron_count) * (W_MAX - W_MIN) + W_MIN)
        self.sigma = nn.Parameter(torch.rand(wiring.neuron_count, wiring.neuron_count) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN)
        self.mu = nn.Parameter(torch.rand(wiring.neuron_count, wiring.neuron_count) * (MU_MAX - MU_MIN) + MU_MIN)
        self.erev = nn.Parameter(torch.Tensor(wiring.erev_initializer()))
        
        # Initialize sensory parameters with random values within specified ranges
        self.sensory_w = nn.Parameter(torch.rand(wiring.input_dim, wiring.neuron_count) * (SENSORY_W_MAX - SENSORY_W_MIN) + SENSORY_W_MIN)
        self.sensory_sigma = nn.Parameter(torch.rand(wiring.input_dim, wiring.neuron_count) * (SENSORY_SIGMA_MAX - SENSORY_SIGMA_MIN) + SENSORY_SIGMA_MIN)
        self.sensory_mu = nn.Parameter(torch.rand(wiring.input_dim, wiring.neuron_count) * (SENSORY_MU_MAX - SENSORY_MU_MIN) + SENSORY_MU_MIN)
        self.sensory_erev = nn.Parameter(torch.Tensor(wiring.sensory_erev_initializer()))

        # Sparsity masks (fixed, non-learnable) based on the wiring adjacency matrices
        self.sparsity_mask = torch.Tensor(np.abs(wiring.adjacency_matrix))
        self.sensory_sparsity_mask = torch.Tensor(np.abs(wiring.sensory_adjacency_matrix))

    def forward(self, inputs, state, elapsed_time=1.0):
        return self.ode_solver(inputs, state, elapsed_time)

    def ode_solver(self, inputs, state, elapsed_time):
        v_pre = state  # Previous state (voltage)

        # Pre-compute the effects of the sensory neurons
        sensory_activation = self.softplus(self.sensory_w) * self.sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_activation = sensory_activation * self.sensory_sparsity_mask
        sensory_reversal_activation = sensory_activation * self.sensory_erev

        # Calculate the numerator and denominator for sensory inputs
        w_numerator_sensory = torch.sum(sensory_reversal_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_activation, dim=1)

        # Calculate membrane capacitance over time
        cm_t = self.softplus(self.cm) / (elapsed_time / self.ode_unfolds)

        # Initialize weights for neuron connections
        w_param = self.softplus(self.w)
        for _ in range(self.ode_unfolds):
            # Activation based on previous state
            w_activation = w_param * self.sigmoid(v_pre, self.mu, self.sigma)
            w_activation = w_activation * self.sparsity_mask
            reversal_activation = w_activation * self.erev

            # Calculate the numerator and denominator for neuron connections
            w_numerator = torch.sum(reversal_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            # Leak conductance and voltage calculations
            gleak = self.softplus(self.gleak)
            numerator = cm_t * v_pre + gleak * self.vleak + w_numerator
            denominator = cm_t + gleak + w_denominator

            # Update the state (voltage)
            v_pre = numerator / (denominator + self.epsilon)

        return v_pre

    def sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # Unsqueeze to match dimensions
        activation = sigma * (v_pre - mu)  # Apply sigma and mean shift
        return torch.sigmoid(activation)  # Apply sigmoid activation

### Implementing the LTCCell Class
class LTCCell(nn.Module):
    def __init__(self, wiring, in_features=None, ode_unfolds=6, epsilon=1e-8):
        super(LTCCell, self).__init__()        
        self.wiring = wiring  # Wiring object
        self.ode_unfolds = ode_unfolds  # Number of ODE solver iterations
        self.epsilon = epsilon  # Small value to avoid division by zero

        self.neuron = LIFNeuronLayer(wiring, ode_unfolds, epsilon)  # Initialize LIFNeuron with the given wiring

    def forward(self, inputs, states, elapsed_time=1.0):
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
        
    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()  # Get batch size and sequence length from input dimensions
        
        states = torch.zeros(batch_size, self.hidden_dim)  # Initialize hidden states with zeros
            
        outputs = []  # List to store outputs for each time step

        for t in range(seq_len):
            output, states = self.cell(inputs[:, t, :], states)  # Compute output and next state for each time step
            outputs.append(output)  # Append the output to the list

        result = torch.stack(outputs, dim=1)  # Stack the outputs along the sequence dimension
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


### Training
if __name__ == "__main__":
# Hyperparameters
    input_dim = 2 # Number of input dimensions
    hidden_dim = 8 # Number of hidden dimensions (number of neurons in LIFNeuralLayer)
    output_dim = 2 # Number of output dimensions
    num_points = 500 # Number of spiral points in dataset
    num_turns = 3 # Number of spiral turns
    learning_rate = 0.005
    num_epochs = 200
    seq_len = 3 # Maximum length of the sample sequence
    batch_size = 32

    # Generate data
    data = generate_spiral_data(num_points, num_turns)
    all_inputs = data[:-1, :]
    all_targets = data[1:, :]

    # Prepare input and target sequences
    trajectory_count = max(1, len(all_inputs) - seq_len)
    train_inputs = [torch.FloatTensor(all_inputs[i:i + seq_len]) for i in range(trajectory_count)]
    train_targets = [torch.FloatTensor(all_targets[i:i + seq_len]) for i in range(trajectory_count)]

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

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()    
        total_loss = 0

        # Iterate over batches
        for x, y_target in zip(train_input_batches, train_target_batches):
            optimizer.zero_grad()
            x = torch.stack(x)  # Stack batch of sequences
            y_target = torch.stack(y_target)  # Stack batch of targets
            outputs = model(x)  # Forward pass through the model
            loss = criterion(outputs, y_target)  # Compute loss

            # Accumulate total loss and perform backward pass and optimization step
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Print loss every 100 epochs and plot predictions
        if (epoch + 1) % 100 == 0:
            # Prediction and plotting
            model.eval()
            with torch.no_grad():
                predictions = model(torch.FloatTensor(all_inputs).unsqueeze(0))
                np_predictions = predictions.squeeze(0).numpy()
                val_loss = criterion(predictions, torch.FloatTensor(all_targets).unsqueeze(0))  # Compute loss
            print(f'Epoch [{epoch+1}/{num_epochs}], Total train loss: {total_loss:.4f}, Total val loss: {val_loss:.4f}')

            plt.plot(all_targets[:, 0], all_targets[:, 1], 'g-', label='True Path')
            plt.plot(np_predictions[:, 0], np_predictions[:, 1], 'r-', label='Predicted Path')
            plt.legend()
            plt.show()