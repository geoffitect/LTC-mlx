#============================================================
# DIABOLICAL LTC-RNN DEBUG VERSION
# Let's see why the training loop exits immediately
#============================================================

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pickle
import random
from typing import List, Tuple, Dict, Optional
import time
from tqdm import tqdm
import json
from collections import Counter

# Import our models
from sequence_to_3di import SequenceTo3DiModel, SPLINE_CONFIG, AA_TO_IDX, IDX_TO_AA
from enhanced_real_trainer_fixed import REAL_FOLDSEEK_3DI_ALPHABET, REAL_FOLDSEEK_3DI_TO_IDX

print("🔍 DIABOLICAL LTC-RNN DEBUG VERSION")
print("=" * 60)

# Create reverse mappings
IDX_TO_3DI = {i: char for char, i in REAL_FOLDSEEK_3DI_TO_IDX.items()}

class SimpleDataset:
    """Simplified dataset for debugging"""

    def __init__(self, pairs: List[Tuple[str, str]], split: str = "train"):
        self.pairs = pairs
        self.split = split
        print(f"🔍 {split.upper()} dataset: {len(pairs):,} pairs")

    def get_batch(self, batch_size: int) -> Tuple[mx.array, mx.array]:
        """Simple batch generation with debugging"""

        print(f"🔍 Getting batch of size {batch_size} from {len(self.pairs)} pairs...")

        if len(self.pairs) == 0:
            raise ValueError(f"No pairs in {self.split} dataset")

        # Get first few pairs for debugging
        batch_pairs = self.pairs[:min(batch_size, len(self.pairs))]

        sequences = []
        targets = []
        FIXED_LENGTH = SPLINE_CONFIG['seq_len']

        print(f"🔍 Processing {len(batch_pairs)} pairs...")

        for i, (aa_seq, three_di_seq) in enumerate(batch_pairs):
            try:
                print(f"🔍 Pair {i+1}: AA len={len(aa_seq)}, 3Di len={len(three_di_seq)}")

                # Skip empty
                if not aa_seq or not three_di_seq:
                    print(f"⚠️  Skipping empty sequence {i+1}")
                    continue

                # Convert to indices
                aa_tokens = [AA_TO_IDX.get(aa, 0) for aa in aa_seq]
                three_di_tokens = [REAL_FOLDSEEK_3DI_TO_IDX.get(char, 0) for char in three_di_seq]

                print(f"🔍 Converted to tokens: AA={len(aa_tokens)}, 3Di={len(three_di_tokens)}")

                # Skip too short
                if len(aa_tokens) < 10 or len(three_di_tokens) < 10:
                    print(f"⚠️  Skipping short sequence {i+1}")
                    continue

                # Truncate if needed
                if len(aa_tokens) > FIXED_LENGTH:
                    aa_tokens = aa_tokens[:FIXED_LENGTH]
                if len(three_di_tokens) > FIXED_LENGTH:
                    three_di_tokens = three_di_tokens[:FIXED_LENGTH]

                # Pad to fixed length
                aa_tokens_padded = aa_tokens + [0] * (FIXED_LENGTH - len(aa_tokens))
                three_di_tokens_padded = three_di_tokens + [0] * (FIXED_LENGTH - len(three_di_tokens))

                print(f"🔍 Padded lengths: AA={len(aa_tokens_padded)}, 3Di={len(three_di_tokens_padded)}")

                sequences.append(aa_tokens_padded)
                targets.append(three_di_tokens_padded)

            except Exception as e:
                print(f"❌ Error processing pair {i+1}: {e}")
                continue

        print(f"🔍 Final batch: {len(sequences)} valid sequences")

        if len(sequences) == 0:
            raise ValueError("No valid sequences in batch")

        # Create MLX arrays
        try:
            seq_array = mx.array(sequences)
            tgt_array = mx.array(targets)
            print(f"🔍 MLX arrays created: seq={seq_array.shape}, tgt={tgt_array.shape}")
            return seq_array, tgt_array
        except Exception as e:
            print(f"❌ MLX array creation failed: {e}")
            raise

def simple_focal_loss(logits: mx.array, targets: mx.array, gamma: float = 2.0) -> mx.array:
    """Simplified focal loss for debugging"""

    print(f"🔍 Computing focal loss: logits={logits.shape}, targets={targets.shape}")

    # Create mask for non-padding tokens
    mask = (targets != 0).astype(mx.float32)
    print(f"🔍 Mask created: {mask.shape}, non-zero tokens: {mx.sum(mask).item()}")

    # Softmax and log softmax
    probs = nn.softmax(logits, axis=-1)
    log_probs = nn.log_softmax(logits, axis=-1)

    # Get target probabilities
    target_probs = mx.take_along_axis(probs, targets[:, :, None], axis=-1).squeeze(-1)

    # Focal weights
    focal_weights = mx.power(1.0 - target_probs, gamma)

    # Cross entropy
    ce_loss = -mx.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)

    # Apply focal weights and mask
    focal_loss = focal_weights * ce_loss * mask

    # Mean over valid tokens
    total_loss = mx.sum(focal_loss) / mx.sum(mask)

    print(f"🔍 Loss computed: {total_loss.item():.4f}")
    return total_loss

def debug_training_step(model, sequences: mx.array, targets: mx.array) -> float:
    """Debug a single training step"""

    print(f"🔍 DEBUG TRAINING STEP:")
    print(f"  Input shapes: seq={sequences.shape}, tgt={targets.shape}")

    try:
        # Forward pass
        print(f"🔍 Running forward pass...")
        logits = model(sequences)
        print(f"🔍 Forward pass successful: {logits.shape}")

        # Compute loss
        print(f"🔍 Computing loss...")
        loss = simple_focal_loss(logits, targets)
        print(f"🔍 Loss computed: {loss.item():.4f}")

        return loss.item()

    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def debug_train():
    """Debug version of training"""

    print("\n🔍 DEBUG TRAINING START")
    print("=" * 40)

    # Load minimal data for debugging
    print("🔍 Loading data...")
    try:
        with open("/tmp/correctly_aligned_pairs.pkl", 'rb') as f:
            all_pairs = pickle.load(f)
        print(f"🔍 Loaded {len(all_pairs):,} pairs")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Take only first 1000 pairs for debugging
    debug_pairs = all_pairs[:1000]
    print(f"🔍 Using {len(debug_pairs)} pairs for debugging")

    # Simple split
    train_pairs = debug_pairs[:700]
    val_pairs = debug_pairs[700:800]
    test_pairs = debug_pairs[800:]

    print(f"🔍 Splits: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # Create datasets
    train_dataset = SimpleDataset(train_pairs, 'train')
    val_dataset = SimpleDataset(val_pairs, 'val')

    # Test data loading
    print("\n🔍 TESTING DATA LOADING...")
    try:
        test_seq, test_tgt = train_dataset.get_batch(4)
        print(f"✅ Data loading successful")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return

    # Initialize model
    print("\n🔍 INITIALIZING MODEL...")
    try:
        model = SequenceTo3DiModel(SPLINE_CONFIG)
        print(f"✅ Model initialized")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return

    # Test forward pass
    print("\n🔍 TESTING FORWARD PASS...")
    try:
        test_logits = model(test_seq)
        print(f"✅ Forward pass successful: {test_logits.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # Test loss computation
    print("\n🔍 TESTING LOSS COMPUTATION...")
    try:
        test_loss = simple_focal_loss(test_logits, test_tgt)
        print(f"✅ Loss computation successful: {test_loss.item():.4f}")
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return

    # Test training step
    print("\n🔍 TESTING TRAINING STEP...")
    try:
        optimizer = optim.Adam(learning_rate=0.001)

        def loss_fn(model, sequences, targets):
            logits = model(sequences)
            return simple_focal_loss(logits, targets)

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        loss, grads = loss_and_grad_fn(model, test_seq, test_tgt)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        print(f"✅ Training step successful: loss={loss.item():.4f}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Minimal training loop
    print("\n🔍 MINIMAL TRAINING LOOP...")
    num_epochs = 5
    batch_size = 8
    batches_per_epoch = 3

    print(f"🔍 Starting {num_epochs} epochs, {batch_size} batch size, {batches_per_epoch} batches/epoch")

    for epoch in range(num_epochs):
        print(f"\n🔍 EPOCH {epoch + 1}/{num_epochs}")

        epoch_loss = 0.0
        successful_batches = 0

        for batch_idx in range(batches_per_epoch):
            print(f"🔍 Batch {batch_idx + 1}/{batches_per_epoch}")

            try:
                # Get batch
                sequences, targets = train_dataset.get_batch(batch_size)

                # Training step
                loss, grads = loss_and_grad_fn(model, sequences, targets)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                batch_loss = loss.item()
                epoch_loss += batch_loss
                successful_batches += 1

                print(f"✅ Batch {batch_idx + 1} complete: loss={batch_loss:.4f}")

            except Exception as e:
                print(f"❌ Batch {batch_idx + 1} failed: {e}")
                continue

        if successful_batches > 0:
            avg_loss = epoch_loss / successful_batches
            print(f"🔍 Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}, successful_batches={successful_batches}")
        else:
            print(f"❌ Epoch {epoch + 1} failed: no successful batches")

    print(f"\n✅ DEBUG TRAINING COMPLETE!")

if __name__ == "__main__":
    print("\n🔍 DIABOLICAL LTC DEBUG SESSION")
    print("=" * 50)

    try:
        debug_train()
    except Exception as e:
        print(f"💥 DEBUG SESSION FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("🔍 DEBUG SESSION COMPLETE")
    print("=" * 50)