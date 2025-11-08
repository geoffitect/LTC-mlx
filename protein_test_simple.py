# Quick test of the protein folding framework with small dataset
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from protein_folding_ltc import ProteinFoldingLTC, PROTEIN_CONFIG, FOLDSEEK_3DI_ALPHABET, FOLDSEEK_3DI_TO_IDX

print("🧪 Quick Protein Folding Test")
print("=" * 40)

# Create synthetic data directly
print("Creating synthetic data...")
sequences = []
coords = []

for i in range(5):
    seq_len = 100

    # Create sequence tokens
    tokens = [FOLDSEEK_3DI_TO_IDX[aa] for aa in np.random.choice(list(FOLDSEEK_3DI_ALPHABET), size=seq_len)]

    # Create coordinates
    backbone_coords = []
    pos = np.array([0.0, 0.0, 0.0])
    for j in range(seq_len):
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        pos += direction * 3.8  # CA-CA distance
        backbone_coords.append(pos.copy().tolist())  # Convert to list

    sequences.append(tokens)
    coords.append(backbone_coords)

# Pad to model length
max_len = PROTEIN_CONFIG['seq_len']
for i in range(len(sequences)):
    seq = sequences[i]
    coord = coords[i]

    if len(seq) > max_len:
        seq = seq[:max_len]
        coord = coord[:max_len]
    else:
        padding = max_len - len(seq)
        seq.extend([0] * padding)
        coord.extend([[0.0, 0.0, 0.0]] * padding)

    sequences[i] = seq
    coords[i] = coord

# Convert to MLX arrays
print("Converting to MLX arrays...")
sequences_mx = mx.array(sequences)
coords_mx = mx.array(coords)
print(f"Batch shapes: {sequences_mx.shape}, {coords_mx.shape}")

# Test model
print("Testing model...")
model = ProteinFoldingLTC(PROTEIN_CONFIG)

# Forward pass
predictions = model(sequences_mx)
print(f"Prediction shape: {predictions.shape}")
print("✅ Framework working!")

# Test single prediction
print("\n🧪 Testing single prediction...")
test_sequence = "ABCDEFGHIJKLMNOP"  # Sample 3Di sequence
tokens = [FOLDSEEK_3DI_TO_IDX.get(aa, 0) for aa in test_sequence]
tokens.extend([0] * (max_len - len(tokens)))  # Pad

input_tokens = mx.array(tokens)[None, :]  # Add batch dimension
prediction = model(input_tokens)
result = prediction[0, :len(test_sequence), :]  # Remove padding
print(f"Predicted {len(test_sequence)} residue coordinates")
print(f"Sample coordinates: {result[:3]}")
print("✅ Single prediction working!")