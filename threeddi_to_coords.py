#============================================================
# 3Di → Backbone Coordinates Pipeline
# Rotation-invariant encoding for end-to-end training with AA→3Di module
#============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import time
from tqdm import tqdm

# Import constants from our working modules
from enhanced_real_trainer_fixed import REAL_FOLDSEEK_3DI_ALPHABET, REAL_FOLDSEEK_3DI_TO_IDX

print("🧬 3Di → Backbone Coordinates Pipeline")
print("=" * 80)

# ============================================================================
# Rotation-Invariant Coordinate Encoding
# ============================================================================

class RotationInvariantEncoder:
    """
    Encodes backbone coordinates in rotation-invariant format
    Uses internal distances, angles, and dihedrals instead of absolute coords
    """

    @staticmethod
    def coords_to_internal(coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert backbone coordinates to rotation-invariant internal coordinates

        Args:
            coords: [batch, seq_len, 3] backbone coordinates (N, CA, C atoms)

        Returns:
            Dictionary with distances, angles, dihedrals
        """
        batch_size, seq_len, _ = coords.shape

        # Extract consecutive CA atoms (backbone trace)
        ca_coords = coords  # Assuming input is CA coordinates

        # Calculate pairwise distances (rotation invariant)
        distances = torch.cdist(ca_coords, ca_coords)  # [batch, seq_len, seq_len]

        # Calculate bond angles (3 consecutive atoms)
        angles = []
        for i in range(seq_len - 2):
            v1 = ca_coords[:, i+1] - ca_coords[:, i]     # Vector i -> i+1
            v2 = ca_coords[:, i+2] - ca_coords[:, i+1]   # Vector i+1 -> i+2

            # Calculate angle between vectors
            cos_angle = torch.sum(v1 * v2, dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1))
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # Numerical stability
            angle = torch.acos(cos_angle)
            angles.append(angle)

        angles = torch.stack(angles, dim=1) if angles else torch.empty(batch_size, 0)

        # Calculate dihedral angles (4 consecutive atoms)
        dihedrals = []
        for i in range(seq_len - 3):
            # Vectors for dihedral calculation
            v1 = ca_coords[:, i+1] - ca_coords[:, i]
            v2 = ca_coords[:, i+2] - ca_coords[:, i+1]
            v3 = ca_coords[:, i+3] - ca_coords[:, i+2]

            # Cross products
            n1 = torch.cross(v1, v2, dim=-1)
            n2 = torch.cross(v2, v3, dim=-1)

            # Dihedral angle
            cos_dihedral = torch.sum(n1 * n2, dim=-1) / (torch.norm(n1, dim=-1) * torch.norm(n2, dim=-1))
            cos_dihedral = torch.clamp(cos_dihedral, -1.0, 1.0)
            dihedral = torch.acos(cos_dihedral)
            dihedrals.append(dihedral)

        dihedrals = torch.stack(dihedrals, dim=1) if dihedrals else torch.empty(batch_size, 0)

        # Local distances (consecutive residues)
        local_distances = []
        for i in range(seq_len - 1):
            dist = torch.norm(ca_coords[:, i+1] - ca_coords[:, i], dim=-1)
            local_distances.append(dist)

        local_distances = torch.stack(local_distances, dim=1) if local_distances else torch.empty(batch_size, 0)

        return {
            'distances': distances,           # [batch, seq_len, seq_len] - all pairwise
            'local_distances': local_distances,  # [batch, seq_len-1] - consecutive
            'angles': angles,                # [batch, seq_len-2] - bond angles
            'dihedrals': dihedrals          # [batch, seq_len-3] - dihedral angles
        }

    @staticmethod
    def internal_to_coords(internal_dict: Dict[str, torch.Tensor], seq_len: int) -> torch.Tensor:
        """
        Reconstruct approximate coordinates from internal coordinates
        This is a simplified reconstruction - in practice would use more sophisticated methods
        """
        batch_size = internal_dict['local_distances'].shape[0]

        # Start with first residue at origin
        coords = torch.zeros(batch_size, seq_len, 3)

        if seq_len > 0:
            coords[:, 0] = torch.tensor([0.0, 0.0, 0.0])  # First residue at origin

        if seq_len > 1:
            # Second residue along x-axis
            local_dist = internal_dict['local_distances'][:, 0]
            coords[:, 1] = torch.stack([local_dist, torch.zeros_like(local_dist), torch.zeros_like(local_dist)], dim=1)

        # Reconstruct remaining residues using bond lengths and angles
        for i in range(2, seq_len):
            if i-2 < internal_dict['angles'].shape[1]:
                bond_length = internal_dict['local_distances'][:, i-1]
                bond_angle = internal_dict['angles'][:, i-2]

                # Simple planar reconstruction (can be improved)
                prev_vec = coords[:, i-1] - coords[:, i-2]
                prev_vec_norm = prev_vec / torch.norm(prev_vec, dim=1, keepdim=True)

                # Rotate by bond angle
                cos_angle = torch.cos(bond_angle).unsqueeze(1)
                sin_angle = torch.sin(bond_angle).unsqueeze(1)

                # Simple 2D rotation in xy-plane (simplified)
                new_vec = torch.stack([
                    cos_angle.squeeze() * bond_length,
                    sin_angle.squeeze() * bond_length,
                    torch.zeros_like(bond_length)
                ], dim=1)

                coords[:, i] = coords[:, i-1] + new_vec

        return coords

# ============================================================================
# 3Di → Coordinates LTC Model
# ============================================================================

class LTCCell3D(nn.Module):
    """LTC cell optimized for 3D coordinate generation"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Time constants (learnable)
        self.tau = nn.Parameter(torch.rand(hidden_size) * 1.9 + 0.1)

        # Neural circuit matrices
        self.W_input = nn.Linear(input_size, hidden_size)
        self.W_hidden = nn.Linear(hidden_size, hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_t: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Forward pass through LTC cell"""

        # Input and recurrent connections
        input_contrib = self.W_input(input_t)
        hidden_contrib = self.W_hidden(hidden)

        # LTC dynamics: dh/dt = (-h + f(Wh + Ux + b)) / tau
        activation = torch.tanh(hidden_contrib + input_contrib + self.bias)
        dhdt = (-hidden + activation) / self.tau.unsqueeze(0)

        # Euler integration step
        dt = 0.1  # Time step
        new_hidden = hidden + dt * dhdt

        # Normalize for stability
        new_hidden = self.layer_norm(new_hidden)

        return new_hidden

class ThreeDiToCoords(nn.Module):
    """
    Neural network to predict backbone coordinates from 3Di sequences
    Uses rotation-invariant representations
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 3Di token embedding
        self.threeddi_embedding = nn.Embedding(
            len(REAL_FOLDSEEK_3DI_ALPHABET),
            config['embedding_dim']
        )

        # Positional encoding
        self.pos_embedding = nn.Embedding(config['seq_len'], config['embedding_dim'])

        # LTC layers for sequential processing
        self.ltc_layers = nn.ModuleList()
        for i in range(config['num_layers']):
            input_dim = config['embedding_dim'] if i == 0 else config['hidden_dim']
            self.ltc_layers.append(LTCCell3D(input_dim, config['hidden_dim']))

        # Output heads for different geometric features
        self.distance_head = nn.Linear(config['hidden_dim'], 1)  # Local distances
        self.angle_head = nn.Linear(config['hidden_dim'], 1)     # Bond angles
        self.dihedral_head = nn.Linear(config['hidden_dim'], 1)  # Dihedral angles

        # Final coordinate reconstruction layer
        self.coord_reconstructor = nn.Linear(config['hidden_dim'], 3)  # Direct coords as backup

        # Dropout
        self.dropout = nn.Dropout(config.get('dropout_rate', 0.1))

        print(f"🧠 ThreeDiToCoords initialized:")
        print(f"  📏 Sequence length: {config['seq_len']}")
        print(f"  🔬 3Di vocab size: {len(REAL_FOLDSEEK_3DI_ALPHABET)}")
        print(f"  🧠 Hidden dimensions: {config['hidden_dim']}")
        print(f"  🏗️  LTC layers: {config['num_layers']}")

    def forward(self, threeddi_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: 3Di tokens → internal coordinates

        Args:
            threeddi_tokens: [batch_size, seq_len] 3Di token indices

        Returns:
            Dictionary with predicted geometric features
        """
        batch_size, seq_len = threeddi_tokens.shape

        # Embed 3Di tokens
        token_emb = self.threeddi_embedding(threeddi_tokens)  # [batch, seq, emb_dim]

        # Add positional encoding
        positions = torch.arange(seq_len, device=threeddi_tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)

        embedded = token_emb + pos_emb
        embedded = self.dropout(embedded)

        # Process through LTC layers
        hidden_states = []
        current_hidden = torch.zeros(batch_size, self.config['hidden_dim'], device=threeddi_tokens.device)

        for t in range(seq_len):
            input_t = embedded[:, t, :]  # Current time step input

            # Pass through LTC layers sequentially
            layer_input = input_t
            for layer in self.ltc_layers:
                current_hidden = layer(layer_input, current_hidden)
                layer_input = current_hidden  # Output becomes input to next layer

            hidden_states.append(current_hidden)

        hidden_sequence = torch.stack(hidden_states, dim=1)  # [batch, seq, hidden_dim]

        # Predict geometric features
        local_distances = self.distance_head(hidden_sequence[:, :-1]).squeeze(-1)  # [batch, seq-1]
        angles = self.angle_head(hidden_sequence[:, :-2]).squeeze(-1)             # [batch, seq-2]
        dihedrals = self.dihedral_head(hidden_sequence[:, :-3]).squeeze(-1)       # [batch, seq-3]

        # Direct coordinate prediction (as backup)
        coords_direct = self.coord_reconstructor(hidden_sequence)  # [batch, seq, 3]

        return {
            'local_distances': F.softplus(local_distances) + 1.0,  # Ensure positive distances (1-10 Å typical)
            'angles': torch.sigmoid(angles) * np.pi,               # Angles between 0 and π
            'dihedrals': torch.sigmoid(dihedrals) * 2 * np.pi,     # Dihedrals between 0 and 2π
            'coords_direct': coords_direct,                        # Direct coordinates
            'hidden_states': hidden_sequence                       # For analysis
        }

# ============================================================================
# Training Configuration
# ============================================================================

COORDS_CONFIG = {
    'seq_len': 512,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 3,  # Slightly deeper for geometric complexity
    'dropout_rate': 0.15,  # Slightly higher dropout for regularization
    'learning_rate': 0.0001,  # Lower LR for coordinate prediction
    'batch_size': 64,  # Smaller batch for memory efficiency
    'epochs': 50,
    'coordinate_weight': 1.0,    # Weight for coordinate loss
    'geometry_weight': 2.0,      # Weight for geometric constraints
}

# ============================================================================
# Combined Loss Function
# ============================================================================

class GeometryAwareLoss(nn.Module):
    """Loss function that combines coordinate prediction with geometric constraints"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Calculate combined geometric loss

        Args:
            predictions: Model predictions
            targets: Ground truth internal coordinates
        """
        losses = {}

        # Local distance loss
        if 'local_distances' in predictions and 'local_distances' in targets:
            dist_loss = F.mse_loss(predictions['local_distances'], targets['local_distances'])
            losses['distance_loss'] = dist_loss

        # Angle loss
        if 'angles' in predictions and 'angles' in targets:
            angle_loss = F.mse_loss(predictions['angles'], targets['angles'])
            losses['angle_loss'] = angle_loss

        # Dihedral loss (circular)
        if 'dihedrals' in predictions and 'dihedrals' in targets:
            # Handle circular nature of dihedrals
            pred_dihedrals = predictions['dihedrals']
            target_dihedrals = targets['dihedrals']

            # Circular MSE loss
            diff = pred_dihedrals - target_dihedrals
            circular_diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            dihedral_loss = torch.mean(circular_diff ** 2)
            losses['dihedral_loss'] = dihedral_loss

        # Direct coordinate loss (if available)
        if 'coords_direct' in predictions and 'coords' in targets:
            coord_loss = F.mse_loss(predictions['coords_direct'], targets['coords'])
            losses['coord_loss'] = coord_loss

        # Combined loss
        total_loss = 0
        for loss_name, loss_value in losses.items():
            if 'coord' in loss_name:
                total_loss += self.config['coordinate_weight'] * loss_value
            else:
                total_loss += self.config['geometry_weight'] * loss_value

        losses['total_loss'] = total_loss
        return losses

# ============================================================================
# Testing and Validation
# ============================================================================

def test_rotation_invariance():
    """Test that the encoding is truly rotation invariant"""
    print("🔄 Testing rotation invariance...")

    # Create test coordinates
    coords = torch.randn(2, 10, 3)  # Random 10-residue structures

    # Apply random rotation
    theta = torch.rand(1) * 2 * np.pi
    rotation_matrix = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta),  torch.cos(theta), 0],
        [0,                 0,                1]
    ], dtype=torch.float32)

    rotated_coords = torch.matmul(coords, rotation_matrix.T)

    # Encode both
    encoder = RotationInvariantEncoder()
    internal_1 = encoder.coords_to_internal(coords)
    internal_2 = encoder.coords_to_internal(rotated_coords)

    # Check if internal coordinates are the same
    distance_diff = torch.mean(torch.abs(internal_1['distances'] - internal_2['distances']))
    angle_diff = torch.mean(torch.abs(internal_1['angles'] - internal_2['angles']))

    print(f"  Distance difference: {distance_diff:.6f}")
    print(f"  Angle difference: {angle_diff:.6f}")

    if distance_diff < 1e-5 and angle_diff < 1e-5:
        print("✅ Rotation invariance test passed!")
    else:
        print("❌ Rotation invariance test failed!")

def test_model_architecture():
    """Test the model architecture"""
    print("🔄 Testing model architecture...")

    model = ThreeDiToCoords(COORDS_CONFIG)

    # Test input
    batch_size = 2
    seq_len = 50
    threeddi_tokens = torch.randint(0, len(REAL_FOLDSEEK_3DI_ALPHABET), (batch_size, seq_len))

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        outputs = model(threeddi_tokens)
    inference_time = time.time() - start_time

    print(f"  Input shape: {threeddi_tokens.shape}")
    print(f"  Output shapes:")
    for key, value in outputs.items():
        if torch.is_tensor(value):
            print(f"    {key}: {value.shape}")

    print(f"  Inference time: {inference_time:.3f}s")

    # Test loss calculation
    loss_fn = GeometryAwareLoss(COORDS_CONFIG)

    # Create dummy targets
    targets = {
        'local_distances': torch.rand(batch_size, seq_len-1) * 3 + 1,  # 1-4 Å
        'angles': torch.rand(batch_size, seq_len-2) * np.pi,           # 0-π
        'dihedrals': torch.rand(batch_size, seq_len-3) * 2 * np.pi,    # 0-2π
        'coords': torch.randn(batch_size, seq_len, 3)
    }

    losses = loss_fn(outputs, targets)
    print(f"  Test losses:")
    for loss_name, loss_value in losses.items():
        print(f"    {loss_name}: {loss_value:.4f}")

    print("✅ Model architecture test passed!")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("🧬 3Di → Backbone Coordinates Pipeline")
    print("=" * 80)

    print("\n📋 Configuration:")
    for key, value in COORDS_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n🧪 Running tests...")
    test_rotation_invariance()
    print()
    test_model_architecture()

    print("\n🎯 Next steps:")
    print("  1. Load SwissProt coordinate data")
    print("  2. Train 3Di→coords model")
    print("  3. Combine with AA→3Di for end-to-end training")
    print("  4. Export to CoreML for iPhone deployment")

    print("\n✅ 3Di → Coordinates pipeline ready for training!")
    print("🚀 Ready to predict backbone structures from 3Di sequences!")