#============================================================
# Validation Benchmark - AA ↔ 3Di Alignment Quality
# Showcase model predictions vs. ground truth on 20 random sequences
#============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Dict
import json
from tqdm import tqdm

print("🧪 Validation Benchmark - AA → 3Di Alignment Quality")
print("=" * 80)

# ============================================================================
# Constants and Configuration
# ============================================================================

# Real Foldseek 3Di alphabet (corrected version)
REAL_FOLDSEEK_3DI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
REAL_FOLDSEEK_3DI_TO_IDX = {char: i for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}
REAL_IDX_TO_FOLDSEEK_3DI = {i: char for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}

# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# Model configuration (must match training config)
CONFIG = {
    'seq_len': 512,
    'aa_vocab_size': 20,
    'struct_vocab_size': 20,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout_rate': 0.1,
}

# ============================================================================
# Model Architecture (same as training)
# ============================================================================

class AdaptiveWeights(nn.Module):
    """Learnable adaptive class weights"""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Initialize with uniform weights
        self.class_weights = nn.Parameter(torch.ones(num_classes) / num_classes)

    def forward(self):
        # Apply softmax to ensure weights are positive and sum to 1
        return F.softmax(self.class_weights, dim=0)

class LiquidTimeConstantSplineLayer(nn.Module):
    """LTC layer with spline-based dynamics"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Sequence embedding
        self.seq_embedding = nn.Embedding(config['aa_vocab_size'], config['embedding_dim'])
        self.pos_embedding = nn.Embedding(config['seq_len'], config['embedding_dim'])

        # LTC dynamics
        self.input_proj = nn.Linear(config['embedding_dim'], config['hidden_dim'])

        # Learnable time constants and dynamics
        self.tau = nn.Parameter(torch.rand(config['hidden_dim']) * 1.9 + 0.1)  # Uniform between 0.1 and 2.0
        self.A = nn.Parameter(torch.randn(config['hidden_dim'], config['hidden_dim']) * 0.1)
        self.b = nn.Parameter(torch.zeros(config['hidden_dim']))

        # Output projection
        self.output_proj = nn.Linear(config['hidden_dim'], config['struct_vocab_size'])

        # Normalization
        self.layer_norm = nn.LayerNorm(config['hidden_dim'])

        # Dropout
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Embedding
        token_emb = self.seq_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        embedded = token_emb + pos_emb
        embedded = self.dropout(embedded)

        # Project input
        x_proj = self.input_proj(embedded)

        # LTC dynamics
        h = torch.zeros(batch_size, self.config['hidden_dim'], device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            input_t = x_proj[:, t, :]

            # Continuous dynamics
            tau_broadcast = self.tau.unsqueeze(0)
            dh_dt = -h / tau_broadcast + torch.tanh(h @ self.A.T + input_t + self.b)

            # Euler step
            h = h + 0.1 * dh_dt
            h = self.layer_norm(h)

            # Project to output
            output_t = self.output_proj(h)
            outputs.append(output_t)

        return torch.stack(outputs, dim=1)

class ProteinLTCModel(nn.Module):
    """Complete protein LTC model"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Main LTC layer
        self.ltc_layer = LiquidTimeConstantSplineLayer(config)

        # Adaptive weights for class balancing
        self.adaptive_weights = AdaptiveWeights(config['struct_vocab_size'])

    def forward(self, x):
        return self.ltc_layer(x)

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_validation_sequences(num_samples=20) -> List[Tuple[str, str]]:
    """Load random validation sequences from the dataset"""
    print(f"📖 Loading {num_samples} random validation sequences...")

    try:
        # Load AA sequences
        aa_dict = {}
        with open("aa_sequences.fasta", 'r') as f:
            current_header = None
            current_seq = ""
            for line in f:
                if line.startswith('>'):
                    if current_header and current_seq:
                        # Extract ID from ">AF-A0A009IHW8-F1-model_v6 ..."
                        header_id = current_header[1:].split()[0]  # Remove '>' and take first part
                        aa_dict[header_id] = current_seq
                    current_header = line.strip()
                    current_seq = ""
                else:
                    current_seq += line.strip()

            # Add last sequence
            if current_header and current_seq:
                header_id = current_header[1:].split()[0]  # Remove '>' and take first part
                aa_dict[header_id] = current_seq

        # Load 3Di sequences
        struct_dict = {}
        with open("3di_sequences.tsv", 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    struct_dict[parts[0]] = parts[1]

        # Find common sequences
        common_ids = list(set(aa_dict.keys()) & set(struct_dict.keys()))

        # Sample random sequences
        selected_ids = random.sample(common_ids, min(num_samples, len(common_ids)))

        validation_pairs = []
        for seq_id in selected_ids:
            aa_seq = aa_dict[seq_id]
            struct_seq = struct_dict[seq_id]

            # Filter valid sequences (length > 20, < 300, valid characters)
            if (20 <= len(aa_seq) <= 300 and
                len(aa_seq) == len(struct_seq) and
                all(aa in AMINO_ACIDS for aa in aa_seq) and
                all(s in REAL_FOLDSEEK_3DI_ALPHABET for s in struct_seq)):
                validation_pairs.append((aa_seq, struct_seq))

        print(f"✅ Loaded {len(validation_pairs)} valid sequence pairs")
        return validation_pairs

    except FileNotFoundError as e:
        print(f"❌ Error loading sequences: {e}")
        print("📝 Generating synthetic validation data instead...")
        return generate_synthetic_validation_data(num_samples)

def generate_synthetic_validation_data(num_samples=20) -> List[Tuple[str, str]]:
    """Generate synthetic validation data if files not available"""
    pairs = []

    for i in range(num_samples):
        # Generate realistic length
        length = random.randint(50, 200)

        # Generate amino acid sequence
        aa_seq = ''.join(random.choices(AMINO_ACIDS, k=length))

        # Generate corresponding 3Di sequence with some pattern
        struct_seq = []
        for aa in aa_seq:
            # Simple mapping with some randomness
            if aa in 'AVILM':  # Hydrophobic
                struct_choices = ['D', 'E', 'F', 'G']
            elif aa in 'EDRK':  # Charged
                struct_choices = ['P', 'Q', 'R', 'S']
            else:  # Others
                struct_choices = random.choices(REAL_FOLDSEEK_3DI_ALPHABET, k=4)

            struct_seq.append(random.choice(struct_choices))

        pairs.append((aa_seq, ''.join(struct_seq)))

    return pairs

# ============================================================================
# Model Loading and Inference
# ============================================================================

def load_trained_model(checkpoint_path: str) -> ProteinLTCModel:
    """Load trained model from checkpoint"""
    print(f"🔄 Loading model from {checkpoint_path}...")

    try:
        # Initialize model
        model = ProteinLTCModel(CONFIG)

        # Handle different checkpoint formats
        if checkpoint_path.endswith('.safetensors'):
            # Load safetensors format
            try:
                from safetensors import safe_open
                with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                    state_dict = {}
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                model.load_state_dict(state_dict)
                print(f"✅ Safetensors model loaded successfully!")
            except ImportError:
                print("⚠️  safetensors not available, trying torch.load...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint)
                print(f"✅ Model loaded successfully!")
        else:
            # Load standard PyTorch format
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded successfully!")
                if 'epoch' in checkpoint:
                    print(f"📊 Trained for {checkpoint['epoch']} epochs")
                if 'loss' in checkpoint:
                    print(f"📊 Final loss: {checkpoint['loss']:.4f}")
            else:
                print("⚠️  Checkpoint format not recognized, using random weights")

        model.eval()
        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🎲 Using randomly initialized model for demonstration")
        model = ProteinLTCModel(CONFIG)
        model.eval()
        return model

def predict_3di_sequence(model: ProteinLTCModel, aa_sequence: str) -> str:
    """Predict 3Di sequence from amino acid sequence"""

    # Convert to indices
    aa_indices = [AA_TO_IDX.get(aa, 0) for aa in aa_sequence]

    # Pad or truncate to model length
    max_len = CONFIG['seq_len']
    if len(aa_indices) > max_len:
        aa_indices = aa_indices[:max_len]
        original_length = max_len
    else:
        original_length = len(aa_indices)
        aa_indices.extend([0] * (max_len - len(aa_indices)))

    # Create tensor
    input_tensor = torch.tensor(aa_indices, dtype=torch.long).unsqueeze(0)

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=-1)[0, :original_length]

    # Convert back to 3Di sequence
    predicted_3di = ''.join([REAL_IDX_TO_FOLDSEEK_3DI[idx.item()] for idx in predictions])
    return predicted_3di

# ============================================================================
# Validation Analysis
# ============================================================================

def calculate_accuracy_metrics(true_seq: str, pred_seq: str) -> Dict:
    """Calculate various accuracy metrics"""

    if len(true_seq) != len(pred_seq):
        return {
            'exact_match': 0.0,
            'length_match': False,
            'position_accuracy': 0.0,
            'character_distribution': {}
        }

    # Exact matches per position
    exact_matches = sum(1 for t, p in zip(true_seq, pred_seq) if t == p)
    position_accuracy = exact_matches / len(true_seq)

    # Character distribution
    true_dist = {char: true_seq.count(char) / len(true_seq) for char in REAL_FOLDSEEK_3DI_ALPHABET}
    pred_dist = {char: pred_seq.count(char) / len(pred_seq) for char in REAL_FOLDSEEK_3DI_ALPHABET}

    return {
        'exact_match': float(true_seq == pred_seq),
        'length_match': True,
        'position_accuracy': position_accuracy,
        'exact_matches': exact_matches,
        'total_positions': len(true_seq),
        'true_distribution': true_dist,
        'pred_distribution': pred_dist
    }

def run_validation_benchmark(model: ProteinLTCModel, validation_pairs: List[Tuple[str, str]]):
    """Run validation benchmark and display results"""

    print(f"\n🎯 Running validation benchmark on {len(validation_pairs)} sequences...")
    print("=" * 80)

    all_metrics = []
    vocab_usage = {char: 0 for char in REAL_FOLDSEEK_3DI_ALPHABET}

    for i, (aa_seq, true_3di) in enumerate(tqdm(validation_pairs, desc="Validating")):

        # Predict 3Di sequence
        pred_3di = predict_3di_sequence(model, aa_seq)

        # Calculate metrics
        metrics = calculate_accuracy_metrics(true_3di, pred_3di)
        all_metrics.append(metrics)

        # Track vocabulary usage
        for char in pred_3di:
            if char in vocab_usage:
                vocab_usage[char] += 1

        # Display individual results
        print(f"\n📊 Sequence {i+1}/{len(validation_pairs)}:")
        print(f"   Length: {len(aa_seq)} residues")
        print(f"   AA:       {aa_seq[:60]}{'...' if len(aa_seq) > 60 else ''}")
        print(f"   True 3Di: {true_3di[:60]}{'...' if len(true_3di) > 60 else ''}")
        print(f"   Pred 3Di: {pred_3di[:60]}{'...' if len(pred_3di) > 60 else ''}")
        print(f"   ✅ Position Accuracy: {metrics['position_accuracy']:.2%} ({metrics.get('exact_matches', 0)}/{metrics.get('total_positions', 0)})")

        if metrics['position_accuracy'] > 0.5:
            print(f"   🎉 EXCELLENT: >50% position accuracy!")
        elif metrics['position_accuracy'] > 0.3:
            print(f"   ✅ GOOD: >30% position accuracy")
        elif metrics['position_accuracy'] > 0.1:
            print(f"   📈 FAIR: >10% position accuracy")
        else:
            print(f"   📉 POOR: <10% position accuracy")

    # Overall statistics
    print(f"\n📈 OVERALL VALIDATION RESULTS")
    print("=" * 50)

    avg_accuracy = np.mean([m['position_accuracy'] for m in all_metrics])
    total_predictions = sum(vocab_usage.values())
    vocab_coverage = sum(1 for count in vocab_usage.values() if count > 0) / len(vocab_usage)

    print(f"📊 Average Position Accuracy: {avg_accuracy:.2%}")
    print(f"📊 Vocabulary Coverage: {vocab_coverage:.1%} ({sum(1 for count in vocab_usage.values() if count > 0)}/20 characters used)")

    # Vocabulary usage breakdown
    print(f"\n📝 Predicted Character Distribution:")
    for char in REAL_FOLDSEEK_3DI_ALPHABET:
        percentage = (vocab_usage[char] / total_predictions * 100) if total_predictions > 0 else 0
        bar_length = int(percentage / 2)  # Scale for display
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"   {char}: {percentage:5.1f}% |{bar}| ({vocab_usage[char]} uses)")

    # Performance classification
    print(f"\n🎯 MODEL PERFORMANCE ASSESSMENT:")
    if avg_accuracy > 0.7:
        print(f"   🚀 OUTSTANDING: >70% accuracy - Production ready!")
    elif avg_accuracy > 0.5:
        print(f"   🎉 EXCELLENT: >50% accuracy - Very strong performance!")
    elif avg_accuracy > 0.3:
        print(f"   ✅ GOOD: >30% accuracy - Solid performance, continue training")
    elif avg_accuracy > 0.15:
        print(f"   📈 FAIR: >15% accuracy - Learning progress, more training needed")
    elif avg_accuracy > 0.05:
        print(f"   📉 POOR: >5% accuracy - Early training, much more needed")
    else:
        print(f"   ❌ RANDOM: <5% accuracy - Random performance, check model/data")

    return all_metrics, vocab_usage

# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    print("🎯 Validation Benchmark Setup")
    print("-" * 40)

    # Load validation data
    validation_pairs = load_validation_sequences(20)

    if not validation_pairs:
        print("❌ No validation data available!")
        return

    # Try to load the latest model checkpoint
    possible_checkpoints = [
        "ultimate_pytorch_checkpoints/pytorch_checkpoint_epoch_5.pt",
        "/tmp/ultimate_550k_checkpoints/latest_checkpoint.safetensors",
        "/tmp/ultimate_550k_checkpoints/checkpoint_epoch_010.safetensors",
        "/tmp/ultimate_550k_checkpoints/checkpoint_epoch_005.safetensors",
        "/tmp/best_pytorch_ltc_model.pth",
        "/tmp/pytorch_ltc_checkpoint_epoch_5.pth",
        "/tmp/pytorch_ltc_checkpoint_epoch_10.pth",
        "checkpoint.pth",
        "best_model.pth"
    ]

    model = None
    for checkpoint_path in possible_checkpoints:
        try:
            model = load_trained_model(checkpoint_path)
            print(f"✅ Using checkpoint: {checkpoint_path}")
            break
        except:
            continue

    if model is None:
        print("⚠️  No trained checkpoints found, using random model for demo")
        model = ProteinLTCModel(CONFIG)
        model.eval()

    # Run validation benchmark
    metrics, vocab_usage = run_validation_benchmark(model, validation_pairs)

    # Save results
    results = {
        'validation_pairs': len(validation_pairs),
        'average_accuracy': np.mean([m['position_accuracy'] for m in metrics]),
        'vocab_coverage': sum(1 for count in vocab_usage.values() if count > 0) / 20,
        'vocab_usage': vocab_usage,
        'individual_metrics': metrics
    }

    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to validation_results.json")
    print("🎉 Validation benchmark complete!")

if __name__ == "__main__":
    main()