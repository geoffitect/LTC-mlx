#============================================================
# Phase 2: 3Di → Backbone LTC Network
# Convert 3Di structural tokens to backbone coordinates
#============================================================

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from tqdm import tqdm
import json
from collections import Counter

# Import Phase 1 components
from enhanced_real_trainer_fixed import REAL_FOLDSEEK_3DI_ALPHABET, REAL_FOLDSEEK_3DI_TO_IDX

print("🦴 PHASE 2: 3Di → Backbone LTC Network")
print("=" * 80)

# Phase 2 Configuration
BACKBONE_CONFIG = {
    'seq_len': 512,
    'struct_vocab_size': 20,  # 3Di tokens
    'coordinate_dim': 9,      # N(3) + CA(3) + C(3) coordinates
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 3,          # Deeper for coordinate prediction
    'dropout_rate': 0.1,
    'learning_rate': 0.0001,  # Lower LR for coordinate regression
}

# 3Di mappings
IDX_TO_3DI = {i: char for char, i in REAL_FOLDSEEK_3DI_TO_IDX.items()}

class LTCBackboneNeuron(nn.Module):
    """LTC neuron specialized for backbone coordinate prediction"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Spline parameters for coordinate dynamics
        self.spline_layers = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )

        # Time-constant learning (critical for protein dynamics)
        self.tau = nn.Linear(input_size + hidden_size, hidden_size)

        # Sensory processing for 3Di structural information
        self.sensory_mu = nn.Linear(input_size, hidden_size)
        self.sensory_sigma = nn.Linear(input_size, hidden_size)

        print(f"🧠 LTCBackboneNeuron: {input_size} → {hidden_size} (coordinate dynamics)")

    def __call__(self, inputs: mx.array, hidden_state: mx.array, dt: float = 0.1):
        """
        Forward pass for backbone coordinate prediction

        Args:
            inputs: 3Di token embeddings [batch, seq, embedding_dim]
            hidden_state: Previous hidden state [batch, seq, hidden_size]
            dt: Time step for differential equation
        """
        batch_size, seq_len, _ = inputs.shape

        # Combine input and hidden state
        combined = mx.concatenate([inputs, hidden_state], axis=-1)

        # Spline-based feature transformation
        spline_features = self.spline_layers(combined)

        # Time constants (learnable dynamics)
        tau_values = mx.sigmoid(self.tau(combined))
        tau_values = tau_values * 0.9 + 0.1  # Keep in [0.1, 1.0] range

        # Sensory processing of 3Di structure
        mu = self.sensory_mu(inputs)
        sigma = mx.sigmoid(self.sensory_sigma(inputs))

        # LTC dynamics: dh/dt = (-h + f(x)) / tau
        # Where f(x) combines spline features and sensory processing
        target_state = spline_features + mu * sigma

        # Euler integration for coordinate evolution
        dh_dt = (-hidden_state + target_state) / (tau_values + 1e-8)
        new_hidden = hidden_state + dt * dh_dt

        return new_hidden

class ThreeDiToBackboneModel(nn.Module):
    """Complete 3Di → Backbone coordinate prediction model"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # 3Di token embedding
        self.struct_embedding = nn.Embedding(
            config['struct_vocab_size'],
            config['embedding_dim']
        )

        # Stack of LTC layers for hierarchical coordinate prediction
        self.ltc_layers = []
        input_dim = config['embedding_dim']

        for i in range(config['num_layers']):
            layer = LTCBackboneNeuron(input_dim, config['hidden_dim'])
            self.ltc_layers.append(layer)
            input_dim = config['hidden_dim']  # Next layer input

        # Coordinate prediction head (N, CA, C atoms)
        self.coord_predictor = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(config['hidden_dim'], config['coordinate_dim'])
        )

        print(f"🦴 ThreeDiToBackboneModel initialized:")
        print(f"  📏 Sequence length: {config['seq_len']}")
        print(f"  🔤 3Di vocab size: {config['struct_vocab_size']}")
        print(f"  📍 Coordinate dimensions: {config['coordinate_dim']} (N+CA+C)")
        print(f"  🧠 Hidden dimensions: {config['hidden_dim']}")
        print(f"  🏗️  LTC layers: {config['num_layers']}")

    def __call__(self, struct_tokens: mx.array):
        """
        Predict backbone coordinates from 3Di tokens

        Args:
            struct_tokens: 3Di token indices [batch, seq_len]

        Returns:
            coordinates: Predicted backbone coords [batch, seq_len, 9]
                        Format: [N_x, N_y, N_z, CA_x, CA_y, CA_z, C_x, C_y, C_z]
        """
        batch_size, seq_len = struct_tokens.shape

        # Embed 3Di structural tokens
        embedded = self.struct_embedding(struct_tokens)  # [batch, seq, embed_dim]

        # Initialize hidden state for LTC dynamics
        hidden = mx.zeros((batch_size, seq_len, self.config['hidden_dim']))

        # Process through LTC layers (hierarchical coordinate learning)
        current_input = embedded

        for i, ltc_layer in enumerate(self.ltc_layers):
            hidden = ltc_layer(current_input, hidden)
            current_input = hidden  # Feed to next layer

            if i == 0:
                print(f"🧠 LTC Layer {i+1}: {current_input.shape} → structural dynamics")

        # Predict backbone coordinates
        coordinates = self.coord_predictor(hidden)  # [batch, seq, 9]

        return coordinates

class BackboneDataset:
    """Dataset for training 3Di → Backbone coordinate prediction"""

    def __init__(self, struct_sequences: List[str], coordinates: List[np.ndarray]):
        """
        Args:
            struct_sequences: List of 3Di structure strings
            coordinates: List of backbone coordinate arrays [seq_len, 9]
        """
        self.struct_sequences = struct_sequences
        self.coordinates = coordinates

        print(f"🦴 BackboneDataset: {len(struct_sequences)} sequences")

        # Validate data alignment
        valid_pairs = 0
        for seq, coords in zip(struct_sequences, coordinates):
            if len(seq) == coords.shape[0]:
                valid_pairs += 1

        print(f"✅ Valid sequence-coordinate pairs: {valid_pairs}/{len(struct_sequences)}")

    def __len__(self):
        return len(self.struct_sequences)

    def __getitem__(self, idx: int):
        """Get a single training example"""
        struct_seq = self.struct_sequences[idx]
        coords = self.coordinates[idx]

        # Convert 3Di sequence to token indices
        struct_tokens = []
        for char in struct_seq:
            if char in REAL_FOLDSEEK_3DI_TO_IDX:
                struct_tokens.append(REAL_FOLDSEEK_3DI_TO_IDX[char])
            else:
                struct_tokens.append(0)  # Unknown token

        # Pad/truncate to fixed length
        seq_len = BACKBONE_CONFIG['seq_len']
        if len(struct_tokens) < seq_len:
            struct_tokens.extend([0] * (seq_len - len(struct_tokens)))
        else:
            struct_tokens = struct_tokens[:seq_len]

        # Pad/truncate coordinates
        if coords.shape[0] < seq_len:
            padding = np.zeros((seq_len - coords.shape[0], 9))
            coords = np.vstack([coords, padding])
        else:
            coords = coords[:seq_len]

        return mx.array(struct_tokens), mx.array(coords.astype(np.float32))

def create_batch(dataset: BackboneDataset, indices: List[int], batch_size: int = 32):
    """Create a batch of training data"""
    struct_batch = []
    coord_batch = []

    for idx in indices[:batch_size]:
        struct_tokens, coordinates = dataset[idx]
        struct_batch.append(struct_tokens)
        coord_batch.append(coordinates)

    return mx.stack(struct_batch), mx.stack(coord_batch)

def coordinate_loss(predicted: mx.array, target: mx.array, mask: mx.array = None):
    """
    Compute coordinate prediction loss (RMSD-based)

    Args:
        predicted: Predicted coordinates [batch, seq, 9]
        target: Target coordinates [batch, seq, 9]
        mask: Valid position mask [batch, seq] (optional)
    """
    # Compute squared differences
    squared_diff = (predicted - target) ** 2

    # Reshape to compute per-atom RMSD
    # [batch, seq, 9] → [batch, seq, 3, 3] (N, CA, C atoms)
    squared_diff = mx.reshape(squared_diff, (predicted.shape[0], predicted.shape[1], 3, 3))

    # Sum over coordinates (x,y,z) for each atom
    atom_squared_error = mx.sum(squared_diff, axis=-1)  # [batch, seq, 3]

    # Average over atoms (N, CA, C)
    residue_squared_error = mx.mean(atom_squared_error, axis=-1)  # [batch, seq]

    if mask is not None:
        # Mask padding positions
        residue_squared_error = residue_squared_error * mask
        total_error = mx.sum(residue_squared_error)
        valid_positions = mx.sum(mask)
        return total_error / (valid_positions + 1e-8)
    else:
        return mx.mean(residue_squared_error)

def test_backbone_model():
    """Test the Phase 2 backbone model"""
    print("🧪 Testing Phase 2: 3Di → Backbone model...")

    model = ThreeDiToBackboneModel(BACKBONE_CONFIG)

    # Test with dummy data
    batch_size = 4
    seq_len = BACKBONE_CONFIG['seq_len']

    # Random 3Di tokens
    test_tokens = mx.random.randint(0, BACKBONE_CONFIG['struct_vocab_size'],
                                   (batch_size, seq_len))

    # Forward pass
    predicted_coords = model(test_tokens)

    print(f"✅ Model test successful!")
    print(f"  📥 Input shape: {test_tokens.shape} (3Di tokens)")
    print(f"  📤 Output shape: {predicted_coords.shape} (backbone coordinates)")
    print(f"  🎯 Expected: [batch={batch_size}, seq={seq_len}, coords=9]")

    # Test coordinate loss
    dummy_target = mx.random.normal((batch_size, seq_len, 9))
    loss = coordinate_loss(predicted_coords, dummy_target)
    print(f"  📊 Test loss: {float(loss):.4f}")

    return model

if __name__ == "__main__":
    print("🚀 Phase 2: 3Di → Backbone LTC Network")
    print("=" * 60)

    # Test the model
    model = test_backbone_model()

    print("\n🎯 Ready for Phase 2 training!")
    print("  1. Load 3Di sequences and backbone coordinates")
    print("  2. Train coordinate prediction model")
    print("  3. Integrate with Phase 1 for complete pipeline")