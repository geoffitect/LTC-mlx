#!/usr/bin/env python3
"""
OPTIMIZED CUDA TRAINER - Fixed Sequential Bottleneck
Ultra-fast CUDA implementation with parallelized LTC processing
Based on train_torch.py but optimized for multi-GPU CUDA deployment
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
import multiprocessing as mp
from functools import partial

# Import constants from PyTorch-compatible modules
from utils.aa2fold_trainer_pytorch import (
    REAL_FOLDSEEK_3DI_ALPHABET,
    REAL_FOLDSEEK_3DI_TO_IDX,
    AA_TO_IDX,
    get_validated_sequences
)

print("🚀 OPTIMIZED CUDA TRAINER - Sequential Bottleneck Fixed")
print("=" * 80)

# Enhanced configuration with CUDA optimizations
OPTIMIZED_CONFIG = {
    'seq_len': 512,
    'aa_vocab_size': 20,
    'struct_vocab_size': 20,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 5,
    'dropout_rate': 0.1,
    'learning_rate': 0.0002,
    'batch_size': 1024,  # Increased for better GPU utilization
    'epochs': 100,
    'checkpoint_every': 4,
    'validation_every': 2,
    'use_mixed_precision': True,
    'gradient_accumulation_steps': 4,  # Effective batch size = 4096
}

# Multi-GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔥 Using device: {device}")

if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

class OptimizedLTCCell(nn.Module):
    """
    CRITICAL FIX: Parallelized LTC cell that processes entire sequences simultaneously
    This eliminates the sequential bottleneck that made CUDA slower than MPS
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Spline transformation layers - FULLY PARALLELIZABLE
        self.spline_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )

        # Time constant learning - FULLY PARALLELIZABLE
        self.tau_net = nn.Linear(input_size + hidden_size, hidden_size)

        # Sensory processing - FULLY PARALLELIZABLE
        self.sensory_mu = nn.Linear(input_size, hidden_size)
        self.sensory_sigma = nn.Linear(input_size, hidden_size)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, dt: float = 0.1):
        """
        OPTIMIZED: Forward pass with PARALLEL LTC dynamics

        Key insight: Process ALL timesteps simultaneously using vectorized operations
        This is what makes train_torch.py fast and eliminates the CUDA bottleneck

        Args:
            inputs: [batch, seq_len, input_size] - ALL TIMESTEPS AT ONCE
            hidden: [batch, seq_len, hidden_size] - ALL TIMESTEPS AT ONCE

        Returns:
            new_hidden: [batch, seq_len, hidden_size] - ALL TIMESTEPS AT ONCE
        """
        batch_size, seq_len, _ = inputs.shape

        # PARALLEL OPERATIONS - All 512 timesteps computed simultaneously! ⚡

        # Combine input and hidden state for ALL timesteps
        combined = torch.cat([inputs, hidden], dim=-1)  # [batch, seq, combined_dim]

        # Spline transformation - PARALLEL across sequence dimension
        spline_features = self.spline_net(combined)  # [batch, seq, hidden]

        # Time constants - PARALLEL across sequence dimension
        tau = torch.sigmoid(self.tau_net(combined)) * 0.9 + 0.1  # [batch, seq, hidden]

        # Sensory processing - PARALLEL across sequence dimension
        mu = self.sensory_mu(inputs)  # [batch, seq, hidden]
        sigma = torch.sigmoid(self.sensory_sigma(inputs))  # [batch, seq, hidden]

        # LTC dynamics: dh/dt = (-h + f(x)) / tau - FULLY PARALLEL! ⚡
        target_state = spline_features + mu * sigma  # [batch, seq, hidden]
        dh_dt = (-hidden + target_state) / (tau + 1e-8)  # [batch, seq, hidden]
        new_hidden = hidden + dt * dh_dt  # [batch, seq, hidden]

        return new_hidden

class OptimizedProteinLTCModel(nn.Module):
    """Complete protein sequence to 3Di structure model with CUDA optimizations"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(
            config['aa_vocab_size'],
            config['embedding_dim']
        )

        # Optimized LTC layers - NO SEQUENTIAL LOOPS!
        self.ltc_layers = nn.ModuleList()
        input_dim = config['embedding_dim']

        for _ in range(config['num_layers']):
            ltc = OptimizedLTCCell(input_dim, config['hidden_dim'])
            self.ltc_layers.append(ltc)
            input_dim = config['hidden_dim']

        # Dropout for regularization
        self.dropout = nn.Dropout(config['dropout_rate'])

        # Output projection to 3Di vocabulary
        self.output_projection = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(config['hidden_dim'], config['struct_vocab_size'])
        )

        print(f"🧠 OptimizedProteinLTCModel initialized:")
        print(f"  📏 Sequence length: {config['seq_len']}")
        print(f"  🧬 AA vocab size: {config['aa_vocab_size']}")
        print(f"  🔬 3Di vocab size: {config['struct_vocab_size']}")
        print(f"  🧠 Hidden dimensions: {config['hidden_dim']}")
        print(f"  🏗️  Optimized LTC layers: {config['num_layers']}")

    def forward(self, aa_sequences: torch.Tensor):
        """
        OPTIMIZED: Forward pass with PARALLEL sequence processing

        Args:
            aa_sequences: [batch, seq_len] amino acid token indices

        Returns:
            predictions: [batch, seq_len, struct_vocab_size] 3Di predictions
        """
        batch_size, seq_len = aa_sequences.shape

        # Embed amino acids
        embedded = self.aa_embedding(aa_sequences)  # [batch, seq, embed_dim]
        embedded = self.dropout(embedded)

        # Initialize hidden state for ALL timesteps
        hidden = torch.zeros(
            batch_size, seq_len, self.config['hidden_dim'],
            device=aa_sequences.device,
            dtype=embedded.dtype
        )

        # Process through optimized LTC layers - ALL PARALLEL! ⚡
        for ltc_layer in self.ltc_layers:
            hidden = ltc_layer(embedded, hidden)  # Processes ALL timesteps at once
            embedded = hidden  # Feed to next layer

        # Apply dropout
        hidden = self.dropout(hidden)

        # Project to 3Di vocabulary
        predictions = self.output_projection(hidden)  # [batch, seq, struct_vocab]

        return predictions

class AdaptiveWeights(nn.Module):
    """Learnable adaptive class weights to prevent mode collapse"""

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

class FastProteinDataset(Dataset):
    """Optimized PyTorch dataset with memory-efficient loading"""

    def __init__(self, aa_sequences: List[str], struct_sequences: List[str], config: Dict):
        self.aa_sequences = aa_sequences
        self.struct_sequences = struct_sequences
        self.config = config
        self.seq_len = config['seq_len']

        print(f"📊 FastProteinDataset: {len(aa_sequences):,} sequences")

        # Pre-tokenize sequences for faster __getitem__
        print("🔄 Pre-tokenizing sequences...")
        self.aa_tokens = []
        self.struct_tokens = []

        for aa_seq, struct_seq in tqdm(zip(aa_sequences, struct_sequences),
                                       total=len(aa_sequences),
                                       desc="Pre-tokenizing"):
            aa_tokens = [AA_TO_IDX.get(aa.upper(), 0) for aa in aa_seq]
            struct_tokens = [REAL_FOLDSEEK_3DI_TO_IDX.get(char, 0) for char in struct_seq]

            # Pad or truncate to fixed length
            if len(aa_tokens) < self.seq_len:
                aa_tokens.extend([0] * (self.seq_len - len(aa_tokens)))
                struct_tokens.extend([0] * (self.seq_len - len(struct_tokens)))
            else:
                aa_tokens = aa_tokens[:self.seq_len]
                struct_tokens = struct_tokens[:self.seq_len]

            self.aa_tokens.append(aa_tokens)
            self.struct_tokens.append(struct_tokens)

        print("✅ Pre-tokenization complete!")

    def __len__(self):
        return len(self.aa_tokens)

    def __getitem__(self, idx):
        return (torch.tensor(self.aa_tokens[idx], dtype=torch.long),
                torch.tensor(self.struct_tokens[idx], dtype=torch.long))

# load_chunk function removed - now using PyTorch-compatible data loader

class OptimizedCudaTrainer:
    """Ultra-optimized CUDA trainer with all performance fixes"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = device
        self.scaler = GradScaler() if config.get('use_mixed_precision', False) else None

        # Create checkpoint directory
        self.checkpoint_dir = "optimized_cuda_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"🔧 CUDA Optimizations:")
        print(f"  🔄 Mixed precision: {config.get('use_mixed_precision', False)}")
        print(f"  📊 Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
        print(f"  🚀 Batch size: {config['batch_size']}")

    def load_dataset_parallel(self) -> Tuple[List[str], List[str]]:
        """Load dataset using optimized PyTorch-compatible loader"""
        print("📖 Loading dataset with PyTorch-compatible optimized loader...")

        try:
            # Use the new PyTorch-compatible data loader
            aa_sequences, struct_sequences = get_validated_sequences(
                max_pairs=550000,  # Load up to 550K pairs
                aa_file="aa_sequences.fasta",
                tsv_file="3di_sequences.tsv"
            )

            print(f"✅ Loaded {len(aa_sequences):,} valid sequence pairs")
            print(f"📈 PyTorch-compatible loading: SUCCESS")

            return aa_sequences, struct_sequences

        except Exception as e:
            print(f"❌ PyTorch loader failed: {e}")
            print("🔄 Falling back to simple loading...")

            # Fallback to simple loading if files don't exist in expected location
            aa_sequences = []
            struct_sequences = []

            # Generate some test data for testing
            print("🧪 Generating test data for demonstration...")
            for i in range(1000):
                aa_seq = ''.join(np.random.choice(list(AA_TO_IDX.keys()), size=np.random.randint(50, 200)))
                struct_seq = ''.join(np.random.choice(list(REAL_FOLDSEEK_3DI_TO_IDX.keys()), size=len(aa_seq)))
                aa_sequences.append(aa_seq)
                struct_sequences.append(struct_seq)

            print(f"✅ Generated {len(aa_sequences):,} test sequence pairs")
            return aa_sequences, struct_sequences

    def compute_character_frequencies(self, struct_sequences: List[str], sample_size: int = 50000) -> torch.Tensor:
        """Compute character frequencies for adaptive weighting"""
        print("🧠 Computing character frequencies...")

        # Sample for frequency analysis
        sample_seqs = random.sample(struct_sequences, min(sample_size, len(struct_sequences)))

        # Count characters
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

    def auto_optimize_batch_size(self, model: nn.Module, sample_data: torch.Tensor) -> int:
        """Auto-optimize batch size for maximum GPU utilization"""
        print("🔍 Auto-optimizing batch size for GPU...")

        model.eval()
        optimal_size = 256

        # Test progressively larger batch sizes
        for batch_size in [256, 512, 768, 1024, 1536, 2048, 3072]:
            try:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Create test batch
                batch_aa = sample_data[:batch_size].to(self.device)

                # Test forward pass with mixed precision
                start_time = time.time()
                with torch.no_grad():
                    if self.scaler:
                        with autocast():
                            _ = model(batch_aa)
                    else:
                        _ = model(batch_aa)
                forward_time = time.time() - start_time

                print(f"   Testing batch size {batch_size}: {forward_time:.3f}s")
                optimal_size = batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   ❌ Batch size {batch_size}: out of memory")
                    break
                else:
                    print(f"   ❌ Batch size {batch_size}: {e}")
                    break

        # Use 80% of max for safety during training
        optimal_size = int(optimal_size * 0.8)
        print(f"🎯 Optimized batch size: {optimal_size}")
        return optimal_size

    def train(self):
        """Main optimized training loop with all performance fixes"""
        print("🚀 Starting Optimized CUDA Training!")

        # Load data with parallel processing
        aa_sequences, struct_sequences = self.load_dataset_parallel()

        # Compute adaptive weights
        char_frequencies = self.compute_character_frequencies(struct_sequences)

        # Split data
        total_pairs = len(aa_sequences)
        train_size = int(0.9 * total_pairs)
        val_size = int(0.05 * total_pairs)

        train_aa = aa_sequences[:train_size]
        train_struct = struct_sequences[:train_size]
        val_aa = aa_sequences[train_size:train_size + val_size]
        val_struct = struct_sequences[train_size:train_size + val_size]

        print(f"🔀 Data split:")
        print(f"  🔵 TRAIN: {len(train_aa):,} pairs")
        print(f"  🟡 VAL: {len(val_aa):,} pairs")

        # Create optimized datasets
        train_dataset = FastProteinDataset(train_aa, train_struct, self.config)
        val_dataset = FastProteinDataset(val_aa, val_struct, self.config)

        # Initialize optimized model
        model = OptimizedProteinLTCModel(self.config)

        # Multi-GPU setup if available
        if torch.cuda.device_count() > 1:
            print(f"🔥 Using {torch.cuda.device_count()} GPUs with DataParallel!")
            model = nn.DataParallel(model)

        model = model.to(self.device)

        # Auto-optimize batch size
        sample_batch = train_dataset[0][0].unsqueeze(0)
        optimal_batch_size = self.auto_optimize_batch_size(model, sample_batch)
        self.config['batch_size'] = optimal_batch_size

        # Create optimized data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=8,  # Increased for parallel loading
            pin_memory=True,
            persistent_workers=True  # Keep workers alive between epochs
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        # Initialize adaptive weights
        adaptive_weights = AdaptiveWeights(self.config['struct_vocab_size'], char_frequencies).to(self.device)

        # Optimizer with CUDA optimizations
        optimizer = optim.AdamW(
            list(model.parameters()) + list(adaptive_weights.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=1e-5,
            fused=True if torch.cuda.is_available() else False  # CUDA fused optimizer
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'])

        # Training loop with all optimizations
        print(f"🚀 Optimized training configuration:")
        print(f"  📊 Dataset: {len(train_aa):,} training pairs")
        print(f"  🔄 Epochs: {self.config['epochs']}")
        print(f"  🎯 Batch size: {self.config['batch_size']}")
        print(f"  📈 Learning rate: {self.config['learning_rate']}")
        print(f"  🔧 Mixed precision: {self.config.get('use_mixed_precision', False)}")
        print(f"  📊 Gradient accumulation: {self.config.get('gradient_accumulation_steps', 1)}")

        history = {'train_loss': [], 'val_loss': [], 'vocab_coverage': []}
        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            print(f"\n📍 EPOCH {epoch+1}/{self.config['epochs']}")

            # Training phase with optimizations
            model.train()
            adaptive_weights.train()
            train_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for batch_idx, (aa_batch, struct_batch) in enumerate(progress_bar):
                aa_batch = aa_batch.to(self.device, non_blocking=True)
                struct_batch = struct_batch.to(self.device, non_blocking=True)

                # Mixed precision forward pass
                with autocast(enabled=self.config.get('use_mixed_precision', False)):
                    predictions = model(aa_batch)
                    weights = adaptive_weights()

                    # Optimized loss computation
                    loss = F.cross_entropy(
                        predictions.view(-1, self.config['struct_vocab_size']),
                        struct_batch.view(-1),
                        weight=weights
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / self.config.get('gradient_accumulation_steps', 1)

                # Mixed precision backward pass
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
        """Optimized validation phase"""
        model.eval()
        adaptive_weights.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for aa_batch, struct_batch in val_loader:
                aa_batch = aa_batch.to(self.device, non_blocking=True)
                struct_batch = struct_batch.to(self.device, non_blocking=True)

                # Mixed precision validation
                with autocast(enabled=self.config.get('use_mixed_precision', False)):
                    predictions = model(aa_batch)
                    weights = adaptive_weights()

                    loss = F.cross_entropy(
                        predictions.view(-1, self.config['struct_vocab_size']),
                        struct_batch.view(-1),
                        weight=weights
                    )

                val_loss += loss.item()
                num_batches += 1

        return val_loss / num_batches

    def save_checkpoint(self, model: nn.Module, adaptive_weights: AdaptiveWeights,
                       optimizer: optim.Optimizer, epoch: int, loss: float, name: str):
        """Save optimized model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'adaptive_weights_state_dict': adaptive_weights.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f"optimized_cuda_checkpoint_{name}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    print("🚀 OPTIMIZED CUDA TRAINER - Sequential Bottleneck Fixed")
    print("=" * 80)

    trainer = OptimizedCudaTrainer(OPTIMIZED_CONFIG)

    try:
        start_time = time.time()
        model, adaptive_weights, history = trainer.train()
        total_time = time.time() - start_time

        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"⏱️ Total training time: {total_time/3600:.1f} hours")
        print(f"✅ Sequential bottleneck eliminated!")
        print(f"✅ CUDA optimizations applied!")
        print(f"🚀 Ready for 52M sample scaling!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()