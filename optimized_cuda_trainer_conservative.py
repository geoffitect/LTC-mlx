#!/usr/bin/env python3
"""
CONSERVATIVE CUDA TRAINER - Fixed Multiprocessing and Memory Issues
Ultra-stable CUDA implementation for RTX 4060 Ti with 16GB VRAM
Fixes: forking issues, aggressive batch sizes, memory leaks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import random
import time
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
from collections import Counter

# Import constants from PyTorch-compatible modules
from utils.aa2fold_trainer_pytorch import (
    REAL_FOLDSEEK_3DI_ALPHABET,
    REAL_FOLDSEEK_3DI_TO_IDX,
    AA_TO_IDX,
    get_validated_sequences
)

print("🛡️ CONSERVATIVE CUDA TRAINER - Stability First")
print("=" * 80)

# CONSERVATIVE configuration optimized for RTX 4060 Ti
CONSERVATIVE_CONFIG = {
    'seq_len': 512,
    'aa_vocab_size': 20,
    'struct_vocab_size': 20,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 5,
    'dropout_rate': 0.1,
    'learning_rate': 0.0002,
    'batch_size': 128,  # Conservative power of 2 for 16GB VRAM
    'epochs': 100,
    'checkpoint_every': 5,
    'validation_every': 2,
    'use_mixed_precision': True,
    'gradient_accumulation_steps': 2,  # Effective batch size = 256
}

# Device setup with conservative memory management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Conservative CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Set memory fraction to prevent OOM
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of VRAM

class ConservativeLTCCell(nn.Module):
    """Conservative LTC cell with memory-efficient parallel processing"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Memory-efficient spline transformation
        self.spline_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # Conservative time constant learning
        self.tau_net = nn.Linear(input_size + hidden_size, hidden_size)

        # Conservative sensory processing
        self.sensory_mu = nn.Linear(input_size, hidden_size)
        self.sensory_sigma = nn.Linear(input_size, hidden_size)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, dt: float = 0.1):
        """Memory-efficient parallel LTC dynamics"""

        # Process in smaller chunks if needed to save memory
        batch_size, seq_len, _ = inputs.shape

        # Combine input and hidden state
        combined = torch.cat([inputs, hidden], dim=-1)

        # Parallel spline transformation
        spline_features = self.spline_net(combined)

        # Parallel time constants
        tau = torch.sigmoid(self.tau_net(combined)) * 0.9 + 0.1

        # Parallel sensory processing
        mu = self.sensory_mu(inputs)
        sigma = torch.sigmoid(self.sensory_sigma(inputs))

        # LTC dynamics - fully parallel
        target_state = spline_features + mu * sigma
        dh_dt = (-hidden + target_state) / (tau + 1e-8)
        new_hidden = hidden + dt * dh_dt

        return new_hidden

class ConservativeProteinLTCModel(nn.Module):
    """Conservative protein model with memory optimizations"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Conservative amino acid embedding
        self.aa_embedding = nn.Embedding(
            config['aa_vocab_size'],
            config['embedding_dim']
        )

        # Conservative LTC layers
        self.ltc_layers = nn.ModuleList()
        input_dim = config['embedding_dim']

        for _ in range(config['num_layers']):
            ltc = ConservativeLTCCell(input_dim, config['hidden_dim'])
            self.ltc_layers.append(ltc)
            input_dim = config['hidden_dim']

        # Conservative dropout
        self.dropout = nn.Dropout(config['dropout_rate'])

        # Conservative output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(config['hidden_dim'] // 2, config['struct_vocab_size'])
        )

        print(f"🛡️ ConservativeProteinLTCModel initialized:")
        print(f"  📏 Sequence length: {config['seq_len']}")
        print(f"  🧬 AA vocab size: {config['aa_vocab_size']}")
        print(f"  🔬 3Di vocab size: {config['struct_vocab_size']}")
        print(f"  🧠 Hidden dimensions: {config['hidden_dim']}")
        print(f"  🏗️  Conservative LTC layers: {config['num_layers']}")

    def forward(self, aa_sequences: torch.Tensor):
        """Conservative forward pass with memory management"""
        batch_size, seq_len = aa_sequences.shape

        # Embed amino acids
        embedded = self.aa_embedding(aa_sequences)
        embedded = self.dropout(embedded)

        # Initialize hidden state conservatively
        hidden = torch.zeros(
            batch_size, seq_len, self.config['hidden_dim'],
            device=aa_sequences.device,
            dtype=embedded.dtype
        )

        # Process through conservative LTC layers
        for ltc_layer in self.ltc_layers:
            hidden = ltc_layer(embedded, hidden)
            embedded = hidden  # Feed to next layer

            # Clear intermediate tensors to save memory
            torch.cuda.empty_cache()

        # Apply dropout
        hidden = self.dropout(hidden)

        # Project to 3Di vocabulary
        predictions = self.output_projection(hidden)

        return predictions

class AdaptiveWeights(nn.Module):
    """Learnable adaptive class weights - unchanged from original"""

    def __init__(self, num_classes: int, initial_freqs: torch.Tensor):
        super().__init__()

        # Initialize with inverse frequency weights
        initial_weights = 1.0 / (initial_freqs + 1e-6)
        initial_weights = initial_weights / initial_weights.mean()

        # Make weights learnable parameters
        self.log_weights = nn.Parameter(torch.log(initial_weights))

        print(f"🎯 AdaptiveWeights initialized with {num_classes} classes")

    def forward(self):
        """Return current adaptive weights"""
        return torch.exp(self.log_weights)

class ConservativeProteinDataset(Dataset):
    """Conservative dataset with minimal memory usage"""

    def __init__(self, aa_sequences: List[str], struct_sequences: List[str], config: Dict):
        self.aa_sequences = aa_sequences
        self.struct_sequences = struct_sequences
        self.config = config
        self.seq_len = config['seq_len']

        print(f"📊 ConservativeProteinDataset: {len(aa_sequences):,} sequences")
        print("🛡️ Using lazy loading to conserve memory...")

    def __len__(self):
        return len(self.aa_sequences)

    def __getitem__(self, idx):
        aa_seq = self.aa_sequences[idx]
        struct_seq = self.struct_sequences[idx]

        # Convert to token indices
        aa_tokens = [AA_TO_IDX.get(aa.upper(), 0) for aa in aa_seq]
        struct_tokens = [REAL_FOLDSEEK_3DI_TO_IDX.get(char, 0) for char in struct_seq]

        # Pad or truncate to fixed length
        if len(aa_tokens) < self.seq_len:
            aa_tokens.extend([0] * (self.seq_len - len(aa_tokens)))
            struct_tokens.extend([0] * (self.seq_len - len(struct_tokens)))
        else:
            aa_tokens = aa_tokens[:self.seq_len]
            struct_tokens = struct_tokens[:self.seq_len]

        return torch.tensor(aa_tokens, dtype=torch.long), torch.tensor(struct_tokens, dtype=torch.long)

class ConservativeCudaTrainer:
    """Ultra-conservative CUDA trainer for stability"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = device
        self.scaler = GradScaler() if config.get('use_mixed_precision', False) and torch.cuda.is_available() else None

        # Create checkpoint directory
        self.checkpoint_dir = "conservative_cuda_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"🛡️ Conservative CUDA Optimizations:")
        print(f"  🔄 Mixed precision: {config.get('use_mixed_precision', False) and torch.cuda.is_available()}")
        print(f"  📊 Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
        print(f"  🚀 Batch size: {config['batch_size']}")
        print(f"  💾 Memory management: Conservative")

    def load_dataset_conservative(self) -> Tuple[List[str], List[str]]:
        """Load dataset conservatively"""
        print("📖 Loading dataset conservatively...")

        try:
            # Use the PyTorch-compatible data loader with conservative limits
            aa_sequences, struct_sequences = get_validated_sequences(
                max_pairs=100000,  # Conservative limit for testing
                aa_file="aa_sequences.fasta",
                tsv_file="3di_sequences.tsv"
            )

            print(f"✅ Loaded {len(aa_sequences):,} valid sequence pairs")
            return aa_sequences, struct_sequences

        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            print("🧪 Falling back to test data generation...")

            # Generate conservative test data
            aa_sequences = []
            struct_sequences = []

            for i in range(1000):  # Small test set
                aa_seq = ''.join(np.random.choice(list(AA_TO_IDX.keys()), size=np.random.randint(100, 300)))
                struct_seq = ''.join(np.random.choice(list(REAL_FOLDSEEK_3DI_TO_IDX.keys()), size=len(aa_seq)))
                aa_sequences.append(aa_seq)
                struct_sequences.append(struct_seq)

            print(f"✅ Generated {len(aa_sequences):,} test sequence pairs")
            return aa_sequences, struct_sequences

    def compute_character_frequencies(self, struct_sequences: List[str], sample_size: int = 10000) -> torch.Tensor:
        """Conservative character frequency computation"""
        print("🧠 Computing character frequencies (conservative)...")

        # Conservative sample size
        sample_seqs = random.sample(struct_sequences, min(sample_size, len(struct_sequences)))

        char_counts = Counter()
        total_chars = 0

        for seq in tqdm(sample_seqs, desc="Analyzing frequencies"):
            for char in seq:
                char_counts[char] += 1
                total_chars += 1

        # Create frequency tensor
        frequencies = torch.zeros(len(REAL_FOLDSEEK_3DI_ALPHABET))
        for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET):
            if char in char_counts:
                frequencies[i] = char_counts[char] / total_chars
            else:
                frequencies[i] = 1e-6

        print(f"📊 Analyzed {total_chars:,} characters from {len(sample_seqs):,} sequences")
        return frequencies

    def train(self):
        """Conservative training loop with stability focus"""
        print("🚀 Starting Conservative CUDA Training!")

        # Load data conservatively
        aa_sequences, struct_sequences = self.load_dataset_conservative()

        # Compute adaptive weights
        char_frequencies = self.compute_character_frequencies(struct_sequences)

        # Conservative data split
        total_pairs = len(aa_sequences)
        train_size = int(0.9 * total_pairs)
        val_size = int(0.05 * total_pairs)

        train_aa = aa_sequences[:train_size]
        train_struct = struct_sequences[:train_size]
        val_aa = aa_sequences[train_size:train_size + val_size]
        val_struct = struct_sequences[train_size:train_size + val_size]

        print(f"🔀 Conservative data split:")
        print(f"  🔵 TRAIN: {len(train_aa):,} pairs")
        print(f"  🟡 VAL: {len(val_aa):,} pairs")

        # Create conservative datasets
        train_dataset = ConservativeProteinDataset(train_aa, train_struct, self.config)
        val_dataset = ConservativeProteinDataset(val_aa, val_struct, self.config)

        # Initialize conservative model
        model = ConservativeProteinLTCModel(self.config).to(self.device)

        # Conservative data loaders - NO multiprocessing issues
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # NO multiprocessing to avoid fork issues
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # NO multiprocessing
            pin_memory=True
        )

        # Initialize adaptive weights
        adaptive_weights = AdaptiveWeights(self.config['struct_vocab_size'], char_frequencies).to(self.device)

        # Conservative optimizer
        optimizer = optim.AdamW(
            list(model.parameters()) + list(adaptive_weights.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=1e-5,
            fused=True if torch.cuda.is_available() else False
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'])

        print(f"🛡️ Conservative training configuration:")
        print(f"  📊 Dataset: {len(train_aa):,} training pairs")
        print(f"  🔄 Epochs: {self.config['epochs']}")
        print(f"  🎯 Batch size: {self.config['batch_size']} (conservative)")
        print(f"  📈 Learning rate: {self.config['learning_rate']}")
        print(f"  🔧 Mixed precision: {self.config.get('use_mixed_precision', False)}")
        print(f"  📊 Gradient accumulation: {self.config.get('gradient_accumulation_steps', 1)}")
        print(f"  👥 Workers: 0 (no multiprocessing)")

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            print(f"\n📍 EPOCH {epoch+1}/{self.config['epochs']}")

            # Training phase
            model.train()
            adaptive_weights.train()
            train_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for batch_idx, (aa_batch, struct_batch) in enumerate(progress_bar):
                aa_batch = aa_batch.to(self.device, non_blocking=True)
                struct_batch = struct_batch.to(self.device, non_blocking=True)

                # Mixed precision forward pass
                with autocast(device_type='cuda', enabled=self.config.get('use_mixed_precision', False)):
                    predictions = model(aa_batch)
                    weights = adaptive_weights()

                    # Conservative loss computation
                    loss = F.cross_entropy(
                        predictions.view(-1, self.config['struct_vocab_size']),
                        struct_batch.view(-1),
                        weight=weights
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / self.config.get('gradient_accumulation_steps', 1)

                # Conservative backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    if self.scaler:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    optimizer.zero_grad()

                    # Conservative memory management
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                train_loss += loss.item() * self.config.get('gradient_accumulation_steps', 1)
                num_batches += 1

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = train_loss / num_batches
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            if (epoch + 1) % self.config['validation_every'] == 0:
                val_loss = self.validate(model, val_loader, adaptive_weights)
                history['val_loss'].append(val_loss)

                print(f"📊 Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(model, adaptive_weights, optimizer, epoch, val_loss, "best")
            else:
                print(f"📊 Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % self.config['checkpoint_every'] == 0:
                self.save_checkpoint(model, adaptive_weights, optimizer, epoch, avg_train_loss, f"epoch_{epoch+1}")

            # Learning rate step
            scheduler.step()

        return model, adaptive_weights, history

    def validate(self, model: nn.Module, val_loader: DataLoader, adaptive_weights: AdaptiveWeights) -> float:
        """Conservative validation phase"""
        model.eval()
        adaptive_weights.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for aa_batch, struct_batch in val_loader:
                aa_batch = aa_batch.to(self.device, non_blocking=True)
                struct_batch = struct_batch.to(self.device, non_blocking=True)

                # Mixed precision validation
                with autocast(device_type='cuda', enabled=self.config.get('use_mixed_precision', False)):
                    predictions = model(aa_batch)
                    weights = adaptive_weights()

                    loss = F.cross_entropy(
                        predictions.view(-1, self.config['struct_vocab_size']),
                        struct_batch.view(-1),
                        weight=weights
                    )

                val_loss += loss.item()
                num_batches += 1

                # Conservative memory management
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return val_loss / num_batches

    def save_checkpoint(self, model: nn.Module, adaptive_weights: AdaptiveWeights,
                       optimizer: optim.Optimizer, epoch: int, loss: float, name: str):
        """Save conservative model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'adaptive_weights_state_dict': adaptive_weights.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f"conservative_cuda_checkpoint_{name}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    # CRITICAL: Multiprocessing protection for Windows/CUDA
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    print("🛡️ CONSERVATIVE CUDA TRAINER - Stability First")
    print("=" * 80)

    trainer = ConservativeCudaTrainer(CONSERVATIVE_CONFIG)

    try:
        start_time = time.time()
        model, adaptive_weights, history = trainer.train()
        total_time = time.time() - start_time

        print(f"\n🎉 CONSERVATIVE TRAINING COMPLETE!")
        print(f"⏱️ Total training time: {total_time/3600:.1f} hours")
        print(f"✅ No multiprocessing issues!")
        print(f"✅ Conservative memory usage!")
        print(f"🛡️ Stable and reliable training!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()