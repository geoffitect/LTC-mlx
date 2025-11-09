#============================================================
# PyTorch Enhanced Real Data Trainer with Fixed 3Di Character Set
# CUDA-compatible version - MLX dependencies removed
# Training on 550k+ PDB structures with corrected character validation
#============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import time
from tqdm import tqdm

# CORRECTED 3Di alphabet based on actual data analysis
REAL_FOLDSEEK_3DI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # The actual characters found in data
REAL_FOLDSEEK_3DI_TO_IDX = {s: i for i, s in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}
REAL_IDX_TO_FOLDSEEK_3DI = {i: s for i, s in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}

# Amino acid mapping (compatible with train_torch.py)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

print("🧬 PyTorch Enhanced Real Data Trainer - FIXED 3Di Character Set")
print("=" * 70)
print(f"✅ Using corrected 3Di alphabet: {REAL_FOLDSEEK_3DI_ALPHABET}")

# ============================================================================
# Enhanced Data Loading with Fixed Character Validation (PyTorch Compatible)
# ============================================================================

class PyTorchEnhancedRealProteinDataLoader:
    """PyTorch-compatible enhanced real data loader with corrected 3Di character validation"""

    def __init__(self):
        self.aa_sequences_list = []
        self.three_di_sequences_list = []
        self.valid_pairs = []

        print("🔄 Loading real protein data with FIXED character validation (PyTorch)...")

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

    def create_valid_pairs_enhanced(self, max_pairs: int = 50000,
                                  aa_file: str = "aa_sequences.fasta",
                                  tsv_file: str = "3di_sequences.tsv") -> List[Tuple[str, str]]:
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
        self.aa_sequences_list = self.load_amino_acid_sequences_enhanced(aa_file)
        self.three_di_sequences_list = self.load_3di_sequences_enhanced(tsv_file)

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
                invalid_aa = any(aa.upper() not in AA_TO_IDX for aa in aa_seq)
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
# PyTorch Dataset Class for Training
# ============================================================================

class PyTorchSequenceDataset(torch.utils.data.Dataset):
    """PyTorch dataset with corrected 3Di character mapping"""

    def __init__(self, pairs: List[Tuple[str, str]], seq_len: int = 512):
        self.pairs = pairs
        self.seq_len = seq_len
        print(f"📋 PyTorch Dataset initialized with {len(pairs)} sequence pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        aa_seq, three_di_seq = self.pairs[idx]

        # Convert AA sequence to indices
        aa_tokens = [AA_TO_IDX.get(aa.upper(), 0) for aa in aa_seq]

        # FIXED: Convert 3Di sequence using corrected mapping
        three_di_tokens = [REAL_FOLDSEEK_3DI_TO_IDX.get(char, 0) for char in three_di_seq]

        # Pad/truncate to fixed length
        if len(aa_tokens) > self.seq_len:
            aa_tokens = aa_tokens[:self.seq_len]
            three_di_tokens = three_di_tokens[:self.seq_len]
        else:
            aa_tokens.extend([0] * (self.seq_len - len(aa_tokens)))
            three_di_tokens.extend([0] * (self.seq_len - len(three_di_tokens)))

        return torch.tensor(aa_tokens, dtype=torch.long), torch.tensor(three_di_tokens, dtype=torch.long)

# ============================================================================
# Integration Function for Main Trainer
# ============================================================================

def get_validated_sequences(max_pairs: int = 550000,
                          aa_file: str = "aa_sequences.fasta",
                          tsv_file: str = "3di_sequences.tsv") -> Tuple[List[str], List[str]]:
    """
    Load and validate sequences for the main optimized trainer
    Returns tuple of (aa_sequences, struct_sequences) ready for training
    """

    print("📊 Loading validated sequences for optimized trainer...")

    loader = PyTorchEnhancedRealProteinDataLoader()
    pairs = loader.create_valid_pairs_enhanced(max_pairs, aa_file, tsv_file)

    if not pairs:
        raise ValueError("No valid pairs found! Check your data files.")

    # Separate into individual lists for compatibility with optimized_cuda_trainer.py
    aa_sequences = [pair[0] for pair in pairs]
    struct_sequences = [pair[1] for pair in pairs]

    print(f"✅ Prepared {len(aa_sequences):,} sequence pairs for training")
    return aa_sequences, struct_sequences

# ============================================================================
# Simple Training Function (for testing purposes)
# ============================================================================

def quick_pytorch_test_training(max_pairs: int = 1000):
    """Quick PyTorch training test to verify everything works"""

    print(f"\n🚀 Quick PyTorch Test Training (up to {max_pairs:,} pairs)")
    print("=" * 70)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    # Load data
    loader = PyTorchEnhancedRealProteinDataLoader()
    pairs = loader.create_valid_pairs_enhanced(max_pairs=max_pairs)

    if len(pairs) < 10:
        print("❌ Insufficient valid pairs for training")
        return None

    # Create dataset
    dataset = PyTorchSequenceDataset(pairs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    # Simple model for testing
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 64)
            self.lstm = nn.LSTM(64, 128, batch_first=True)
            self.output = nn.Linear(128, 20)

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            return self.output(lstm_out)

    model = SimpleTestModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🔧 Test training configuration:")
    print(f"  Pairs: {len(pairs):,}")
    print(f"  Batches: {len(dataloader)}")
    print(f"  Device: {device}")

    # Quick training test (2 epochs)
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        num_batches = 0

        for aa_batch, struct_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            aa_batch = aa_batch.to(device)
            struct_batch = struct_batch.to(device)

            optimizer.zero_grad()
            outputs = model(aa_batch)

            loss = criterion(outputs.view(-1, 20), struct_batch.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1}: loss = {avg_loss:.4f}")

    print("✅ PyTorch test training successful!")
    return model

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n🧬 PYTORCH ENHANCED REAL DATA TRAINER - FIXED CHARACTER SET")
    print("=" * 80)

    print("\n🎯 PyTorch objectives with FIX:")
    print("  1. Load 550k+ real PDB sequences (PyTorch compatible)")
    print("  2. Use CORRECTED 3Di character set from actual data")
    print("  3. Provide clean interface for optimized_cuda_trainer.py")
    print("  4. Validate MLX→PyTorch conversion works correctly")

    try:
        # Test sequence loading
        print("\n🧪 Testing sequence loading...")
        aa_sequences, struct_sequences = get_validated_sequences(max_pairs=1000)

        print(f"\n✅ Sequence loading test:")
        print(f"  AA sequences: {len(aa_sequences):,}")
        print(f"  3Di sequences: {len(struct_sequences):,}")
        print(f"  Sample AA: {aa_sequences[0][:50]}...")
        print(f"  Sample 3Di: {struct_sequences[0][:50]}...")

        # Quick training test
        print("\n🧪 Testing PyTorch training...")
        test_model = quick_pytorch_test_training(max_pairs=500)

        if test_model is not None:
            print(f"\n🎉 PYTORCH TRAINER READY!")
            print(f"  ✅ Data loading: WORKING")
            print(f"  ✅ Character mapping: FIXED")
            print(f"  ✅ PyTorch training: WORKING")
            print(f"  ✅ CUDA compatibility: READY")
            print(f"\n🚀 Ready to integrate with optimized_cuda_trainer.py!")

    except Exception as e:
        print(f"❌ PyTorch trainer test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("🧬 PyTorch Enhanced Real Data Trainer Complete!")
    print("=" * 80)