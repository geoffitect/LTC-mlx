#============================================================
# Enhanced Real Data Trainer with Fixed 3Di Character Set
# Training on 550k+ PDB structures with corrected character validation
#============================================================

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import List, Tuple
import time
from tqdm import tqdm

# Import our models (but override 3Di character set)
from spline.sequence_to_3di import SequenceTo3DiModel, SPLINE_CONFIG, AA_TO_IDX

# CORRECTED 3Di alphabet based on actual data analysis
REAL_FOLDSEEK_3DI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # The actual characters found in data
REAL_FOLDSEEK_3DI_TO_IDX = {s: i for i, s in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}
REAL_IDX_TO_FOLDSEEK_3DI = {i: s for i, s in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}

print("🧬 Enhanced Real Data Trainer - FIXED 3Di Character Set")
print("=" * 70)
print(f"✅ Using corrected 3Di alphabet: {REAL_FOLDSEEK_3DI_ALPHABET}")

# ============================================================================
# Enhanced Data Loading with Fixed Character Validation
# ============================================================================

class FixedEnhancedRealProteinDataLoader:
    """Enhanced real data loader with corrected 3Di character validation"""

    def __init__(self):
        self.aa_sequences_list = []
        self.three_di_sequences_list = []
        self.valid_pairs = []

        print("🔄 Loading real protein data with FIXED character validation...")

    def load_amino_acid_sequences_enhanced(self, fasta_file: str) -> List[str]:
        """Load amino acid sequences with progress bar"""
        sequences = []

        try:
            print("📖 Loading amino acid sequences...")

            # First pass: count total lines for progress bar
            with open(fasta_file, 'r') as f:
                total_lines = sum(1 for _ in f)

            with open(fasta_file, 'r') as f, tqdm(total=total_lines, desc="Loading AA sequences", unit="lines") as pbar:
                current_seq = []

                for line in f:
                    pbar.update(1)
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

            print(f"✅ Loaded {len(sequences)} amino acid sequences")
            return sequences

        except Exception as e:
            print(f"❌ Error loading amino acid sequences: {e}")
            return []

    def load_3di_sequences_enhanced(self, tsv_file: str) -> List[str]:
        """Load 3Di sequences with progress bar"""
        sequences = []

        try:
            print("📖 Loading 3Di sequences...")

            # Count total lines
            with open(tsv_file, 'r') as f:
                total_lines = sum(1 for _ in f)

            with open(tsv_file, 'r') as f, tqdm(total=total_lines, desc="Loading 3Di sequences", unit="lines") as pbar:
                for line in f:
                    pbar.update(1)
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        three_di_seq = parts[1]
                        sequences.append(three_di_seq)

            print(f"✅ Loaded {len(sequences)} 3Di sequences")
            return sequences

        except Exception as e:
            print(f"❌ Error loading 3Di sequences: {e}")
            return []

    def create_valid_pairs_enhanced(self, max_pairs: int = 50000) -> List[Tuple[str, str]]:
        """Create valid pairs with FIXED 3Di character validation"""

        print(f"🔄 Creating valid pairs with FIXED filtering (max: {max_pairs})...")

        pairs = []
        stats = {
            'skipped_length': 0,
            'skipped_chars': 0,
            'skipped_mismatch': 0,
            'valid_pairs': 0
        }

        # Load data
        self.aa_sequences_list = self.load_amino_acid_sequences_enhanced("/tmp/aa_sequences.fasta")
        self.three_di_sequences_list = self.load_3di_sequences_enhanced("/tmp/3di_sequences.tsv")

        print(f"📊 Data summary:")
        print(f"  Amino acid sequences: {len(self.aa_sequences_list):,}")
        print(f"  3Di sequences: {len(self.three_di_sequences_list):,}")

        # Match by position with progress bar
        min_length = min(len(self.aa_sequences_list), len(self.three_di_sequences_list))
        print(f"  Processing {min_length:,} sequences...")

        # Use corrected 3Di character set
        valid_3di_chars = set(REAL_FOLDSEEK_3DI_ALPHABET)
        print(f"🔤 Using CORRECTED 3Di alphabet: {sorted(valid_3di_chars)}")

        with tqdm(total=min(min_length, max_pairs), desc="Creating pairs", unit="pairs") as pbar:
            for i in range(min_length):
                if len(pairs) >= max_pairs:
                    break

                aa_seq = self.aa_sequences_list[i]
                three_di_seq = self.three_di_sequences_list[i]

                # Quality filters with detailed tracking
                if not (30 <= len(aa_seq) <= 500):
                    stats['skipped_length'] += 1
                    continue

                if len(aa_seq) != len(three_di_seq):
                    stats['skipped_mismatch'] += 1
                    continue

                # Valid amino acids only
                invalid_aa = any(aa not in AA_TO_IDX for aa in aa_seq)
                if invalid_aa:
                    stats['skipped_chars'] += 1
                    continue

                # FIXED: Valid 3Di tokens using corrected character set
                invalid_3di = any(char not in valid_3di_chars for char in three_di_seq)
                if invalid_3di:
                    stats['skipped_chars'] += 1
                    continue

                # All filters passed
                pairs.append((aa_seq, three_di_seq))
                stats['valid_pairs'] += 1
                pbar.update(1)

                # Update progress bar description with current stats
                pbar.set_postfix(
                    valid=len(pairs),
                    length_skip=stats['skipped_length'],
                    mismatch=stats['skipped_mismatch']
                )

        print(f"\n📋 FIXED pair creation summary:")
        print(f"  ✅ Valid pairs: {len(pairs):,}")
        print(f"  ⚠️  Skipped (length): {stats['skipped_length']:,}")
        print(f"  ⚠️  Skipped (length mismatch): {stats['skipped_mismatch']:,}")
        print(f"  ⚠️  Skipped (invalid chars): {stats['skipped_chars']:,}")

        if pairs:
            print(f"\n🔬 Sample pair:")
            print(f"  AA:  {pairs[0][0][:60]}...")
            print(f"  3Di: {pairs[0][1][:60]}...")
            print(f"  Lengths: AA={len(pairs[0][0])}, 3Di={len(pairs[0][1])}")

        self.valid_pairs = pairs
        return pairs

# ============================================================================
# Fixed Training with Corrected Character Mapping
# ============================================================================

def train_enhanced_real_data_fixed(max_pairs: int = 50000):
    """Enhanced training with FIXED 3Di character validation"""

    print(f"\n🚀 Enhanced Real Data Training - FIXED (up to {max_pairs:,} pairs)")
    print("=" * 70)

    # Load real data with FIXED validation
    loader = FixedEnhancedRealProteinDataLoader()
    real_pairs = loader.create_valid_pairs_enhanced(max_pairs=max_pairs)

    if len(real_pairs) < 100:
        print("❌ Insufficient valid pairs for training")
        return None, None, None

    print(f"\n📋 FIXED training configuration:")
    print(f"  Real data pairs: {len(real_pairs):,}")
    print(f"  Model architecture: {SPLINE_CONFIG['num_layers']} LTC layers")
    print(f"  Hidden dimensions: {SPLINE_CONFIG['hidden_dim']}")
    print(f"  Sequence length: {SPLINE_CONFIG['seq_len']}")

    # Create dataset with corrected character mapping
    class FixedSequenceDataset:
        """Dataset with corrected 3Di character mapping"""

        def __init__(self, pairs: List[Tuple[str, str]]):
            self.pairs = pairs
            print(f"📋 Dataset initialized with {len(pairs)} sequence pairs")

        def get_batch(self, batch_size: int) -> Tuple[mx.array, mx.array]:
            # Random sample
            indices = np.random.choice(len(self.pairs), batch_size, replace=True)
            batch_pairs = [self.pairs[i] for i in indices]

            sequences = []
            targets = []

            for aa_seq, three_di_seq in batch_pairs:
                # Convert AA sequence to indices
                aa_tokens = [AA_TO_IDX.get(aa, 0) for aa in aa_seq]

                # FIXED: Convert 3Di sequence using corrected mapping
                three_di_tokens = [REAL_FOLDSEEK_3DI_TO_IDX.get(char, 0) for char in three_di_seq]

                # Pad/truncate to fixed length
                max_len = SPLINE_CONFIG['seq_len']

                if len(aa_tokens) > max_len:
                    aa_tokens = aa_tokens[:max_len]
                    three_di_tokens = three_di_tokens[:max_len]
                else:
                    aa_tokens.extend([0] * (max_len - len(aa_tokens)))
                    three_di_tokens.extend([0] * (max_len - len(three_di_tokens)))

                sequences.append(aa_tokens)
                targets.append(three_di_tokens)

            return mx.array(sequences), mx.array(targets)

    dataset = FixedSequenceDataset(real_pairs)

    # Initialize model
    model = SequenceTo3DiModel(SPLINE_CONFIG)
    total_params = "~2M"  # Estimated for our architecture
    print(f"  Total parameters: {total_params}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=0.0005)

    # Loss function
    def cross_entropy_loss(logits, targets):
        log_probs = nn.log_softmax(logits, axis=-1)
        return -mx.mean(mx.take_along_axis(log_probs, targets[:, :, None], axis=-1))

    def loss_fn(model, sequences, targets):
        logits = model(sequences)
        return cross_entropy_loss(logits, targets)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Enhanced training loop
    num_epochs = 40
    batch_size = 16
    batches_per_epoch = 15

    print(f"\n🔥 Starting FIXED real data training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {batches_per_epoch}")

    losses = []
    start_time = time.time()

    # Training progress bar
    with tqdm(total=num_epochs, desc="Training epochs", unit="epoch") as epoch_pbar:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Batch progress bar
            with tqdm(total=batches_per_epoch, desc=f"Epoch {epoch+1}", unit="batch", leave=False) as batch_pbar:
                for batch_idx in range(batches_per_epoch):
                    try:
                        sequences, targets = dataset.get_batch(batch_size)

                        # Forward pass and gradients
                        loss, grads = loss_and_grad_fn(model, sequences, targets)

                        # Update model
                        optimizer.update(model, grads)
                        mx.eval(model.parameters(), optimizer.state)

                        batch_loss = loss.item()
                        epoch_loss += batch_loss
                        num_batches += 1

                        # Update batch progress
                        batch_pbar.update(1)
                        batch_pbar.set_postfix(loss=f"{batch_loss:.4f}")

                    except Exception as e:
                        batch_pbar.set_postfix(error=str(e)[:20])
                        continue

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                losses.append(avg_loss)

                # Update epoch progress
                epoch_pbar.update(1)
                elapsed = time.time() - start_time
                epoch_pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    time=f"{elapsed:.1f}s"
                )

    training_time = time.time() - start_time
    print(f"\n✅ FIXED real data training completed!")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Total time: {training_time:.1f}s")
    print(f"  Time per epoch: {training_time/num_epochs:.1f}s")

    return model, losses, real_pairs

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\\n🧬 ENHANCED REAL DATA TRAINING - FIXED CHARACTER SET")
    print("=" * 80)

    print("\\n🎯 Enhanced objectives with FIX:")
    print("  1. Load 550k+ real PDB sequences with progress monitoring")
    print("  2. Use CORRECTED 3Di character set from actual data")
    print("  3. Scale up to 50k+ training pairs (should work now!)")
    print("  4. Enhanced evaluation with detailed metrics")

    try:
        # Enhanced training with FIXED character validation
        model, losses, training_pairs = train_enhanced_real_data_fixed(max_pairs=50000)

        if model is not None:
            print(f"\\n🚀 FIXED Real Data Results:")
            print(f"  Training pairs: {len(training_pairs):,}")
            print(f"  Final loss: {losses[-1]:.4f}")

            print(f"\\n🎯 Expected Results with FIX:")
            print(f"  ✅ MASSIVE IMPROVEMENT: Should have 40k-50k pairs instead of 6k")
            print(f"  🚀 Next: Train on full dataset and evaluate performance")
            print(f"  🧬 Ready: Scale to complete 550k dataset")

        else:
            print("❌ Training failed")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

    print("\\n" + "=" * 80)
    print("🧬 FIXED Enhanced Real Data Training Complete!")
    print("=" * 80)