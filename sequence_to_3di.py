#============================================================
# Sequence → 3Di Spline Model
# Amino Acid Sequence to Structural Token Pipeline
#============================================================

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import List, Tuple, Optional

print("🔬 Sequence → 3Di Spline Model")
print("=" * 50)

# ============================================================================
# Constants and Configuration
# ============================================================================

# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# Foldseek 3Di alphabet
FOLDSEEK_3DI_ALPHABET = "ABCDEFGHIJKLMNPQRSTVWXYZ"[:20]
FOLDSEEK_3DI_TO_IDX = {s: i for i, s in enumerate(FOLDSEEK_3DI_ALPHABET)}
IDX_TO_FOLDSEEK_3DI = {i: s for i, s in enumerate(FOLDSEEK_3DI_ALPHABET)}

# Model configuration for sequence → 3Di prediction
SPLINE_CONFIG = {
    'seq_len': 512,          # Maximum sequence length
    'aa_vocab_size': 20,     # 20 amino acids
    'struct_vocab_size': 20, # 20 3Di structural tokens
    'embedding_dim': 128,    # Amino acid embedding dimension
    'hidden_dim': 256,       # Hidden layer dimension
    'num_layers': 2,         # Number of LTC layers
    'dropout_rate': 0.1,
}

print("📋 Spline Model Configuration:")
for key, value in SPLINE_CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# Data Generation for Sequence → 3Di Mapping
# ============================================================================

def generate_sequence_3di_pairs(num_pairs: int = 1000) -> List[Tuple[str, str]]:
    """Generate synthetic sequence → 3Di pairs for training"""

    print(f"🔄 Generating {num_pairs} sequence → 3Di pairs...")

    pairs = []

    for i in range(num_pairs):
        # Generate realistic protein sequence
        length = np.random.randint(50, 200)

        # Use amino acid composition similar to natural proteins
        # AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY" (20 amino acids)
        aa_probs = np.array([
            0.08, 0.04, 0.09, 0.05, 0.06, 0.07, 0.05,  # A, C, D, E, F, G, H
            0.06, 0.04, 0.09, 0.06, 0.02, 0.05, 0.04,  # I, K, L, M, N, P, Q
            0.05, 0.07, 0.06, 0.07, 0.06, 0.01         # R, S, T, V, W, Y (20 total)
        ])
        aa_probs = aa_probs / aa_probs.sum()  # Normalize

        # Generate sequence
        sequence = ''.join(np.random.choice(list(AMINO_ACIDS), size=length, p=aa_probs))

        # Generate corresponding 3Di sequence
        # Use structural propensities based on amino acid properties
        struct_3di = []

        for aa in sequence:
            # Map amino acids to structural preferences
            if aa in 'AVILM':  # Hydrophobic - prefer beta sheets
                struct_probs = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.1] + [0.05] * 14)
            elif aa in 'EDRK':  # Charged - prefer loops
                struct_probs = np.array([0.05] * 6 + [0.2] * 4 + [0.05] * 10)
            elif aa in 'QNST':  # Polar - mixed preferences
                struct_probs = np.array([0.1] * 10 + [0.05] * 10)
            elif aa in 'FHWY':  # Aromatic - special structures
                struct_probs = np.array([0.05] * 5 + [0.25, 0.25] + [0.05] * 8 + [0.2] * 5)
            elif aa == 'P':     # Proline - turn/loop preference
                struct_probs = np.array([0.05] * 8 + [0.4, 0.4] + [0.01] * 10)
            elif aa == 'G':     # Glycine - flexible
                struct_probs = np.ones(20) / 20  # Uniform
            else:              # Default
                struct_probs = np.ones(20) / 20

            struct_probs = struct_probs / struct_probs.sum()
            struct_token = np.random.choice(list(FOLDSEEK_3DI_ALPHABET), p=struct_probs)
            struct_3di.append(struct_token)

        struct_sequence = ''.join(struct_3di)
        pairs.append((sequence, struct_sequence))

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_pairs} pairs...")

    print(f"✅ Generated {len(pairs)} sequence → 3Di pairs")
    return pairs

# ============================================================================
# Neural Architecture Components
# ============================================================================

class SequenceEmbedding(nn.Module):
    """Amino acid sequence embedding"""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Embedding(SPLINE_CONFIG['seq_len'], embedding_dim)

    def __call__(self, x):
        batch_size, seq_len = x.shape

        # Token embeddings
        token_emb = self.embedding(x)

        # Positional encodings
        positions = mx.broadcast_to(mx.arange(seq_len)[None, :], (batch_size, seq_len))
        pos_emb = self.positional_encoding(positions)

        return token_emb + pos_emb

class SequenceLTCLayer(nn.Module):
    """LTC layer for sequence processing"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LTC dynamics parameters
        self.tau = mx.random.uniform(0.1, 2.0, (hidden_dim,))
        self.A = mx.random.normal((hidden_dim, hidden_dim)) * 0.1
        self.b = mx.zeros((hidden_dim,))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def __call__(self, x, state=None):
        batch_size, seq_len, input_dim = x.shape

        if state is None:
            state = mx.zeros((batch_size, self.hidden_dim))

        # Project input
        x_proj = self.input_proj(x)

        outputs = []
        for t in range(seq_len):
            # LTC dynamics
            input_t = x_proj[:, t, :]

            # Continuous dynamics
            tau_broadcast = self.tau[None, :]
            dh_dt = -state / tau_broadcast + mx.tanh(state @ self.A.T + input_t + self.b)

            # Euler step
            state = state + 0.1 * dh_dt

            # Normalize
            state = self.layer_norm(state)

            # Project to output
            output_t = self.output_proj(state)
            outputs.append(output_t)

        return mx.stack(outputs, axis=1), state

class SequenceTo3DiModel(nn.Module):
    """Complete sequence → 3Di prediction model"""

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Sequence embedding
        self.embedding = SequenceEmbedding(
            config['aa_vocab_size'],
            config['embedding_dim']
        )

        # LTC layers
        self.ltc_layers = [
            SequenceLTCLayer(
                config['embedding_dim'] if i == 0 else config['hidden_dim'],
                config['hidden_dim'],
                config['hidden_dim'] if i < config['num_layers'] - 1 else config['struct_vocab_size']
            )
            for i in range(config['num_layers'])
        ]

        # Dropout
        self.dropout = nn.Dropout(config['dropout_rate'])

    def __call__(self, x):
        # Embed amino acid sequence
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # Pass through LTC layers
        hidden = embedded
        state = None

        for layer in self.ltc_layers:
            hidden, state = layer(hidden, state)
            hidden = self.dropout(hidden)

        # Output is 3Di token logits
        return hidden

# ============================================================================
# Dataset and Training
# ============================================================================

class SequenceDataset:
    """Dataset for sequence → 3Di training"""

    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs
        print(f"📋 Dataset initialized with {len(pairs)} sequence pairs")

    def get_batch(self, batch_size: int):
        """Get a batch of training data"""

        batch_indices = np.random.choice(len(self.pairs), min(batch_size, len(self.pairs)),
                                       replace=len(self.pairs) < batch_size)

        batch_sequences = []
        batch_targets = []

        max_len = SPLINE_CONFIG['seq_len']

        for idx in batch_indices:
            sequence, struct_3di = self.pairs[idx]

            # Convert sequence to tokens
            seq_tokens = [AA_TO_IDX.get(aa, 0) for aa in sequence]
            struct_tokens = [FOLDSEEK_3DI_TO_IDX.get(s, 0) for s in struct_3di]

            # Pad or truncate
            if len(seq_tokens) > max_len:
                seq_tokens = seq_tokens[:max_len]
                struct_tokens = struct_tokens[:max_len]
            else:
                padding = max_len - len(seq_tokens)
                seq_tokens.extend([0] * padding)
                struct_tokens.extend([0] * padding)

            batch_sequences.append(seq_tokens)
            batch_targets.append(struct_tokens)

        return mx.array(batch_sequences), mx.array(batch_targets)

def train_sequence_to_3di_model():
    """Train the sequence → 3Di model"""

    print("\n🚀 Training Sequence → 3Di Model")
    print("-" * 50)

    # Generate training data
    pairs = generate_sequence_3di_pairs(1000)
    dataset = SequenceDataset(pairs)

    # Initialize model
    model = SequenceTo3DiModel(SPLINE_CONFIG)

    # Optimizer
    optimizer = optim.Adam(learning_rate=0.001)

    # Loss function (cross-entropy for classification)
    def cross_entropy_loss(logits, targets):
        # logits: (batch, seq, vocab), targets: (batch, seq)
        log_probs = nn.log_softmax(logits, axis=-1)
        return -mx.mean(mx.take_along_axis(log_probs, targets[:, :, None], axis=-1))

    def loss_fn(model, sequences, targets):
        logits = model(sequences)
        return cross_entropy_loss(logits, targets)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop
    num_epochs = 30
    batch_size = 16

    print(f"📋 Training configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {len(dataset.pairs)}")

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Multiple batches per epoch
        for batch_idx in range(10):
            try:
                sequences, targets = dataset.get_batch(batch_size)

                # Forward pass and gradients
                loss, grads = loss_and_grad_fn(model, sequences, targets)

                # Update model
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                epoch_loss += loss.item()
                num_batches += 1

            except Exception as e:
                print(f"  Batch {batch_idx} failed: {e}")
                continue

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print(f"✅ Training completed! Final loss: {losses[-1]:.4f}")
    return model, losses

def predict_3di_sequence(model, amino_sequence: str) -> str:
    """Predict 3Di sequence from amino acid sequence"""

    # Convert to tokens
    tokens = [AA_TO_IDX.get(aa, 0) for aa in amino_sequence]

    # Pad to model length
    max_len = SPLINE_CONFIG['seq_len']
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens.extend([0] * (max_len - len(tokens)))

    # Add batch dimension
    input_tokens = mx.array(tokens)[None, :]

    # Predict (MLX doesn't need no_grad context)
    logits = model(input_tokens)
    predictions = mx.argmax(logits, axis=-1)[0, :len(amino_sequence)]

    # Convert back to 3Di sequence
    predicted_3di = ''.join([IDX_TO_FOLDSEEK_3DI[idx.item()] for idx in predictions])
    return predicted_3di

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n🔬 SEQUENCE → 3Di SPLINE MODEL")
    print("=" * 60)

    print("\n🎯 Model Objective:")
    print("  Amino Acid Sequence → 3Di Structural Tokens")
    print("  Enables end-to-end protein folding prediction")

    try:
        # Train the model
        model, losses = train_sequence_to_3di_model()

        # Test prediction
        print("\n🧪 Testing sequence → 3Di prediction...")
        test_sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWEKLKKLRQQHKLLQPQRSQ"
        predicted_3di = predict_3di_sequence(model, test_sequence)

        print(f"Input sequence:  {test_sequence}")
        print(f"Predicted 3Di:   {predicted_3di}")
        print(f"✅ Successfully predicted 3Di for {len(test_sequence)} residues")

        print("\n🔗 Pipeline Integration:")
        print("  1. Amino Acid Sequence → [THIS MODEL] → 3Di Tokens")
        print("  2. 3Di Tokens → [Folding LTC] → Backbone Coordinates")
        print("  3. Complete: Sequence → Structure Pipeline!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🔬 Sequence → 3Di Spline Model Complete!")
    print("=" * 60)