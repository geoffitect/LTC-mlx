#============================================================
# Real Data Training: Fixed Index Matching
# Training on 550k+ PDB structures with position-based matching
#============================================================

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
import time

# Import our models
from sequence_to_3di import SequenceTo3DiModel, SPLINE_CONFIG, AA_TO_IDX, IDX_TO_AA
from sequence_to_3di import FOLDSEEK_3DI_TO_IDX, IDX_TO_FOLDSEEK_3DI, SequenceDataset

print("🧬 Real Data Trainer: 550k+ PDB Structures (Fixed)")
print("=" * 60)

# ============================================================================
# Fixed Real Data Loading with Position-Based Matching
# ============================================================================

class FixedRealProteinDataLoader:
    """Load real amino acid → 3Di pairs using position-based matching"""

    def __init__(self):
        self.aa_sequences_list = []
        self.three_di_sequences_list = []
        self.valid_pairs = []

        print("🔄 Loading real protein data with position-based matching...")

    def load_amino_acid_sequences_ordered(self, fasta_file: str) -> List[str]:
        """Load amino acid sequences in order"""
        sequences = []

        try:
            with open(fasta_file, 'r') as f:
                current_seq = []

                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        # Save previous sequence
                        if current_seq:
                            sequences.append(''.join(current_seq))
                        current_seq = []
                    else:
                        current_seq.append(line)

                # Save last sequence
                if current_seq:
                    sequences.append(''.join(current_seq))

            print(f"✅ Loaded {len(sequences)} amino acid sequences (ordered)")
            return sequences

        except Exception as e:
            print(f"❌ Error loading amino acid sequences: {e}")
            return []

    def load_3di_sequences_ordered(self, tsv_file: str) -> List[str]:
        """Load 3Di sequences in order"""
        sequences = []

        try:
            with open(tsv_file, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        three_di_seq = parts[1]
                        sequences.append(three_di_seq)

                    if i % 50000 == 0:
                        print(f"  Processed {i} 3Di sequences...")

            print(f"✅ Loaded {len(sequences)} 3Di sequences (ordered)")
            return sequences

        except Exception as e:
            print(f"❌ Error loading 3Di sequences: {e}")
            return []

    def create_valid_pairs_by_position(self, max_pairs: int = 20000) -> List[Tuple[str, str]]:
        """Create valid pairs by matching sequences by position"""

        print(f"🔄 Creating valid pairs by position (max: {max_pairs})...")

        pairs = []
        skipped_length = 0
        skipped_chars = 0
        skipped_mismatch = 0

        # Load data
        self.aa_sequences_list = self.load_amino_acid_sequences_ordered("/tmp/aa_sequences.fasta")
        self.three_di_sequences_list = self.load_3di_sequences_ordered("/tmp/3di_sequences.tsv")

        print(f"📊 Data summary:")
        print(f"  Amino acid sequences: {len(self.aa_sequences_list)}")
        print(f"  3Di sequences: {len(self.three_di_sequences_list)}")

        # Match by position
        min_length = min(len(self.aa_sequences_list), len(self.three_di_sequences_list))
        print(f"  Matching {min_length} sequences by position...")

        count = 0
        for i in range(min_length):
            if count >= max_pairs:
                break

            aa_seq = self.aa_sequences_list[i]
            three_di_seq = self.three_di_sequences_list[i]

            # Quality filters
            # 1. Length filter (reasonable protein lengths)
            if not (30 <= len(aa_seq) <= 500):
                skipped_length += 1
                continue

            # 2. Length match
            if len(aa_seq) != len(three_di_seq):
                skipped_mismatch += 1
                continue

            # 3. Valid amino acids only (standard 20 amino acids)
            invalid_aa = any(aa not in AA_TO_IDX for aa in aa_seq)
            if invalid_aa:
                skipped_chars += 1
                continue

            # 4. Valid 3Di tokens only
            valid_3di_chars = set(FOLDSEEK_3DI_TO_IDX.keys())
            invalid_3di = any(char not in valid_3di_chars for char in three_di_seq)
            if invalid_3di:
                skipped_chars += 1
                continue

            # All filters passed
            pairs.append((aa_seq, three_di_seq))
            count += 1

            if count % 1000 == 0:
                print(f"  Created {count} valid pairs...")

        print(f"\n📋 Pair creation summary:")
        print(f"  Valid pairs: {len(pairs)}")
        print(f"  Skipped (length): {skipped_length}")
        print(f"  Skipped (length mismatch): {skipped_mismatch}")
        print(f"  Skipped (invalid chars): {skipped_chars}")

        if pairs:
            print(f"\n🔬 Sample pair:")
            print(f"  AA:  {pairs[0][0][:60]}...")
            print(f"  3Di: {pairs[0][1][:60]}...")

        self.valid_pairs = pairs
        return pairs

# ============================================================================
# Real Data Training Pipeline
# ============================================================================

def train_on_real_data_fixed():
    """Train sequence → 3Di model on real PDB data with fixed matching"""

    print("\n🚀 Training on Real PDB Data (Fixed)")
    print("=" * 50)

    # Load real data
    loader = FixedRealProteinDataLoader()
    real_pairs = loader.create_valid_pairs_by_position(max_pairs=10000)

    if len(real_pairs) < 100:
        print("❌ Insufficient valid pairs for training")
        return None, None, None

    print(f"\n📋 Training configuration:")
    print(f"  Real data pairs: {len(real_pairs)}")
    print(f"  Model: {SPLINE_CONFIG['num_layers']} LTC layers")
    print(f"  Hidden dim: {SPLINE_CONFIG['hidden_dim']}")

    # Create dataset
    dataset = SequenceDataset(real_pairs)

    # Initialize model
    model = SequenceTo3DiModel(SPLINE_CONFIG)

    # Optimizer
    optimizer = optim.Adam(learning_rate=0.0005)  # Lower LR for real data

    # Loss function
    def cross_entropy_loss(logits, targets):
        log_probs = nn.log_softmax(logits, axis=-1)
        return -mx.mean(mx.take_along_axis(log_probs, targets[:, :, None], axis=-1))

    def loss_fn(model, sequences, targets):
        logits = model(sequences)
        return cross_entropy_loss(logits, targets)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop with real data
    num_epochs = 30  # Fewer epochs initially
    batch_size = 16

    print(f"\n🔥 Starting real data training...")
    losses = []
    start_time = time.time()

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
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")

    print(f"\n✅ Real data training completed!")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Total time: {time.time() - start_time:.1f}s")

    return model, losses, real_pairs

def evaluate_real_model_fixed(model, test_pairs: List[Tuple[str, str]]):
    """Evaluate model on real protein data"""

    print(f"\n🧪 Evaluating Model on Real Data")
    print("=" * 40)

    if not test_pairs:
        print("❌ No test pairs available")
        return 0.0

    # Test on sample pairs
    correct_predictions = 0
    total_predictions = 0

    for i, (aa_seq, true_3di) in enumerate(test_pairs[:5]):
        print(f"\n🔬 Test protein {i+1}:")
        print(f"  Length: {len(aa_seq)} residues")

        # Predict 3Di sequence
        tokens = [AA_TO_IDX.get(aa, 0) for aa in aa_seq]
        max_len = SPLINE_CONFIG['seq_len']

        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([0] * (max_len - len(tokens)))

        input_tokens = mx.array(tokens)[None, :]

        # Predict
        logits = model(input_tokens)
        predictions = mx.argmax(logits, axis=-1)[0, :len(aa_seq)]

        # Convert to 3Di sequence
        predicted_3di = ''.join([
            list(FOLDSEEK_3DI_TO_IDX.keys())[idx.item()]
            for idx in predictions
        ])

        print(f"  True 3Di:      {true_3di[:50]}...")
        print(f"  Predicted 3Di: {predicted_3di[:50]}...")

        # Calculate accuracy
        matches = sum(1 for a, b in zip(true_3di, predicted_3di) if a == b)
        accuracy = matches / len(true_3di) if true_3di else 0

        print(f"  Accuracy: {accuracy:.3f} ({matches}/{len(true_3di)})")

        correct_predictions += matches
        total_predictions += len(true_3di)

    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n📊 Overall Accuracy: {overall_accuracy:.3f}")

    return overall_accuracy

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n🧬 REAL DATA TRAINING - MILESTONE 3 (FIXED)")
    print("=" * 60)

    print("\n🎯 Objectives:")
    print("  1. Load 550k+ real PDB sequences and 3Di structures")
    print("  2. Train spline model on REAL protein folding data")
    print("  3. Evaluate performance on actual protein structures")
    print("  4. Compare with synthetic data results")

    try:
        # Train on real data
        model, losses, training_pairs = train_on_real_data_fixed()

        if model is not None:
            # Evaluate model
            accuracy = evaluate_real_model_fixed(model, training_pairs[-10:])

            print(f"\n🚀 Real Data Results:")
            print(f"  Training pairs: {len(training_pairs)}")
            print(f"  Final loss: {losses[-1]:.4f}")
            print(f"  Prediction accuracy: {accuracy:.3f}")

            print(f"\n🎯 Milestone 3 Assessment:")
            if accuracy > 0.4:
                print("  🚀 OUTSTANDING: Model excels on real PDB data!")
                print("  ✅ Next: Scale to full 550k dataset")
                print("  🧬 Ready: End-to-end folding with real structures")
                print("  📱 Deploy: CoreML conversion for edge devices")
            elif accuracy > 0.25:
                print("  ✅ EXCELLENT: Model learns real protein patterns!")
                print("  🚀 Next: Scale to larger dataset (50k+ pairs)")
                print("  🧬 Ready: Full pipeline with real structures")
            elif accuracy > 0.15:
                print("  ✅ GOOD: Model shows learning on real data")
                print("  🔧 Next: Optimize architecture and training")
                print("  📈 Scale: Gradually increase dataset size")
            else:
                print("  🔧 LEARNING: Initial results on real data")
                print("  📊 Next: Analyze data patterns and model")
                print("  🔍 Debug: Compare real vs synthetic performance")

        else:
            print("❌ Training failed - insufficient data")

    except Exception as e:
        print(f"❌ Real data training failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🧬 Real Data Training Complete - Milestone 3!")
    print("=" * 60)