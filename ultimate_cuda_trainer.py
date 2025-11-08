#============================================================
# ULTIMATE CUDA TRAINER - RTX 16GB Ada Optimized
# High-performance CUDA training for AA → 3Di protein folding
# Optimized for RTX Ada architecture with 16GB VRAM
#============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import time
import pickle
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import random

# Import our components
from spline.sequence_to_3di import SPLINE_CONFIG, AA_TO_IDX
from enhanced_real_trainer_fixed import REAL_FOLDSEEK_3DI_ALPHABET, REAL_FOLDSEEK_3DI_TO_IDX

print("🚀 ULTIMATE CUDA TRAINER - RTX 16GB Ada Optimized")
print("="*80)

# Ensure CUDA is available
if not torch.cuda.is_available():
    print("❌ CUDA not available! Please install CUDA PyTorch.")
    exit(1)

print(f"🔥 Using device: {torch.cuda.get_device_name()}")
print(f"💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class LTCCell(nn.Module):
    """CUDA-optimized Liquid Time Constant cell with spline dynamics"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate components
        self.W_ii = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # Forget gate components
        self.W_if = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_f = nn.Parameter(torch.ones(hidden_size))

        # Cell state components
        self.W_ic = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        # Output gate components
        self.W_io = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        # Time constant parameters (learnable)
        self.tau = nn.Parameter(torch.ones(hidden_size))

        # Spline dynamics parameters
        self.spline_knots = 8
        self.spline_weights = nn.Parameter(torch.randn(hidden_size, self.spline_knots))

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with CUDA optimizations"""
        batch_size = x.size(0)

        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden

        # Gate computations (fused for CUDA efficiency)
        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h) + self.b_i)
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h) + self.b_f)
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h) + self.b_o)
        g = torch.tanh(self.W_ic(x) + self.W_hc(h) + self.b_c)

        # Spline dynamics with CUDA-optimized B-spline computation
        # Simplified B-spline basis functions for speed
        t = torch.sigmoid(h.sum(dim=-1, keepdim=True))  # Time parameter [0,1]
        t_knots = torch.linspace(0, 1, self.spline_knots, device=x.device).view(1, -1)
        basis = torch.exp(-(t - t_knots)**2 * 10)  # Gaussian-like basis
        basis = F.softmax(basis, dim=-1)

        # Apply spline dynamics
        spline_modulation = torch.einsum('bk,hk->bh', basis, self.spline_weights)
        spline_factor = torch.sigmoid(spline_modulation)

        # Time constant modulation (LTC dynamics)
        dt = 1.0  # Time step
        alpha = torch.sigmoid(-self.tau)  # Decay rate

        # Update cell state with LTC dynamics and spline modulation
        c_new = f * c + i * g * spline_factor
        c_new = alpha * c_new + (1 - alpha) * c  # LTC time evolution

        # Output
        h_new = o * torch.tanh(c_new)

        return h_new, c_new

class ProteinLTCModel(nn.Module):
    """CUDA-optimized LTC model for AA → 3Di prediction"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = nn.Embedding(
            config['aa_vocab_size'],
            config['embedding_dim'],
            padding_idx=0
        )

        # LTC layers
        self.ltc_layers = nn.ModuleList()
        input_size = config['embedding_dim']

        for i in range(config['num_layers']):
            self.ltc_layers.append(LTCCell(input_size, config['hidden_dim']))
            input_size = config['hidden_dim']

        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config['hidden_dim']) for _ in range(config['num_layers'])
        ])

        # Dropout for regularization
        self.dropout = nn.Dropout(config['dropout_rate'])

        # Output projection
        self.output_projection = nn.Linear(config['hidden_dim'], config['struct_vocab_size'])

        # Initialize weights for CUDA efficiency
        self._init_weights()

    def _init_weights(self):
        """Initialize weights optimally for CUDA"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass optimized for CUDA"""
        batch_size, seq_len = x.shape

        # Embedding
        embedded = self.embedding(x)  # [batch, seq, embed_dim]
        embedded = self.dropout(embedded)

        # Process through LTC layers
        outputs = []
        hidden_states = [None] * len(self.ltc_layers)

        # Unroll sequence for CUDA efficiency
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # [batch, embed_dim]

            for layer_idx, ltc_layer in enumerate(self.ltc_layers):
                x_t, hidden_states[layer_idx] = ltc_layer(x_t, hidden_states[layer_idx])
                x_t = self.layer_norms[layer_idx](x_t)
                if layer_idx < len(self.ltc_layers) - 1:  # No dropout on final layer
                    x_t = self.dropout(x_t)

            outputs.append(x_t)

        # Stack outputs
        output_sequence = torch.stack(outputs, dim=1)  # [batch, seq, hidden]

        # Final projection
        logits = self.output_projection(output_sequence)  # [batch, seq, vocab]

        return logits

class AdaptiveWeights(nn.Module):
    """CUDA-optimized learnable adaptive class weights"""
    def __init__(self, num_classes: int, initial_weights: torch.Tensor):
        super().__init__()
        # Initialize with inverse frequency weights
        self.weights = nn.Parameter(initial_weights.clone())

    def forward(self) -> torch.Tensor:
        # Apply softplus to ensure positive weights
        return F.softplus(self.weights)

class ProteinDataset(Dataset):
    """CUDA-optimized dataset for protein sequences"""
    def __init__(self, aa_sequences: List[str], struct_sequences: List[str], seq_len: int):
        self.aa_sequences = aa_sequences
        self.struct_sequences = struct_sequences
        self.seq_len = seq_len

        print(f"📊 ProteinDataset: {len(aa_sequences):,} sequences")

    def __len__(self):
        return len(self.aa_sequences)

    def __getitem__(self, idx):
        aa_seq = self.aa_sequences[idx]
        struct_seq = self.struct_sequences[idx]

        # Convert to indices
        aa_indices = torch.zeros(self.seq_len, dtype=torch.long)
        struct_indices = torch.zeros(self.seq_len, dtype=torch.long)

        # Fill sequences
        for i, aa in enumerate(aa_seq[:self.seq_len]):
            aa_indices[i] = AA_TO_IDX.get(aa.upper(), 0)

        for i, struct_char in enumerate(struct_seq[:self.seq_len]):
            struct_indices[i] = REAL_FOLDSEEK_3DI_TO_IDX.get(struct_char, 0)

        return aa_indices, struct_indices

class UltimateCudaTrainer:
    """CUDA-optimized trainer for maximum RTX Ada performance"""
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda')

        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        print(f"🔧 CUDA Optimizations enabled")
        print(f"📊 cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"⚡ TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

    def load_data(self) -> Tuple[List[str], List[str]]:
        """Load and validate the complete 550k dataset"""
        print("📖 Loading COMPLETE 550k dataset...")

        # Load amino acid sequences from FASTA
        print("📖 Loading amino acid sequences from FASTA...")
        aa_dict = {}
        current_header = None
        current_seq = ""

        with open("aa_sequences.fasta", 'r') as f:
            for line in tqdm(f, desc="Loading AA sequences"):
                line = line.strip()
                if line.startswith('>'):
                    if current_header and current_seq:
                        header_id = current_header.split()[0]
                        aa_dict[header_id] = current_seq
                    current_header = line[1:]
                    current_seq = ""
                else:
                    current_seq += line

            if current_header and current_seq:
                header_id = current_header.split()[0]
                aa_dict[header_id] = current_seq

        print(f"✅ Loaded {len(aa_dict):,} amino acid sequences")

        # Load 3Di sequences from TSV
        print("📖 Loading 3Di sequences from TSV...")
        struct_dict = {}

        with open("3di_sequences.tsv", 'r') as f:
            for line in tqdm(f, desc="Loading 3Di sequences"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    seq_id = parts[0]
                    struct_seq = parts[1].strip()
                    struct_dict[seq_id] = struct_seq

        print(f"✅ Loaded {len(struct_dict):,} 3Di sequences")

        # Align sequences by header ID
        print("🔍 Aligning sequences by header ID and validating...")
        common_headers = set(aa_dict.keys()) & set(struct_dict.keys())
        print(f"📊 Common headers: {len(common_headers):,}")

        # Validate sequences
        valid_aa_sequences = []
        valid_struct_sequences = []

        for header_id in tqdm(common_headers, desc="Validating sequences"):
            aa_seq = aa_dict[header_id]
            struct_seq = struct_dict[header_id]

            # Validation checks
            if len(aa_seq) == 0 or len(struct_seq) == 0:
                continue

            if len(aa_seq) != len(struct_seq):
                continue

            # Check for valid characters
            if any(char not in REAL_FOLDSEEK_3DI_TO_IDX for char in struct_seq):
                continue

            if any(char.upper() not in AA_TO_IDX for char in aa_seq):
                continue

            valid_aa_sequences.append(aa_seq)
            valid_struct_sequences.append(struct_seq)

        print(f"✅ Loaded {len(valid_aa_sequences):,} valid sequence pairs")
        print(f"📈 Success rate: {len(valid_aa_sequences)/len(common_headers)*100:.1f}%")

        return valid_aa_sequences, valid_struct_sequences

    def compute_adaptive_weights(self, struct_sequences: List[str], sample_size: int = 50000) -> torch.Tensor:
        """Compute adaptive class weights for CUDA training"""
        print("🧠 Computing character frequencies for adaptive weights...")

        # Sample sequences for frequency analysis
        sample_indices = random.sample(range(len(struct_sequences)),
                                     min(sample_size, len(struct_sequences)))

        char_counts = Counter()
        total_chars = 0

        for idx in tqdm(sample_indices, desc="Analyzing character frequencies"):
            struct_seq = struct_sequences[idx]
            for char in struct_seq:
                char_counts[char] += 1
                total_chars += 1

        print(f"📊 Analyzed {total_chars:,} characters from {len(sample_indices):,} sequences")

        # Compute inverse frequency weights
        weights = torch.ones(len(REAL_FOLDSEEK_3DI_ALPHABET))

        print("📈 Character frequency distribution:")
        for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET):
            freq = char_counts.get(char, 1) / total_chars
            weight = 1.0 / freq
            weights[i] = weight
            print(f"   {char}: {freq:.4f} freq → {weight:.2f} weight")

        return weights.cuda()

    def auto_batch_size(self, model: nn.Module, sample_data: torch.Tensor) -> int:
        """Auto-detect optimal batch size for RTX 16GB"""
        print("🔍 Auto-detecting optimal batch size for RTX Ada...")

        model.eval()
        optimal_batch_size = 32  # Conservative start

        # Test batch sizes optimized for RTX Ada
        test_sizes = [64, 128, 256, 512, 768, 1024]  # Larger sizes for 16GB

        for batch_size in test_sizes:
            try:
                # Clear cache
                torch.cuda.empty_cache()

                # Create test batch
                test_batch = sample_data[:batch_size].cuda()

                # Test forward pass with timing
                start_time = time.time()
                with torch.no_grad():
                    _ = model(test_batch)

                elapsed = time.time() - start_time
                memory_used = torch.cuda.max_memory_allocated() / 1024**3

                print(f"   Testing batch size {batch_size}: {elapsed:.3f}s, {memory_used:.1f}GB")

                # Check if we're under memory limit (leave 2GB headroom)
                if memory_used < 14.0:
                    optimal_batch_size = batch_size
                else:
                    break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                else:
                    raise e

        torch.cuda.empty_cache()
        print(f"🎯 Selected optimal batch size: {optimal_batch_size}")
        return optimal_batch_size

    def train(self, epochs: int = 100, lr: float = 0.0002):
        """Train with CUDA optimizations and mixed precision"""
        # Load data
        aa_sequences, struct_sequences = self.load_data()

        # Compute adaptive weights
        adaptive_weights_tensor = self.compute_adaptive_weights(struct_sequences)

        # Split data
        split_idx = int(0.9 * len(aa_sequences))
        train_aa = aa_sequences[:split_idx]
        train_struct = struct_sequences[:split_idx]
        val_aa = aa_sequences[split_idx:]
        val_struct = struct_sequences[split_idx:]

        print("🔀 Data split:")
        print(f"  🔵 TRAIN: {len(train_aa):,} pairs")
        print(f"  🟡 VAL: {len(val_aa):,} pairs")

        # Create datasets
        train_dataset = ProteinDataset(train_aa, train_struct, self.config['seq_len'])
        val_dataset = ProteinDataset(val_aa, val_struct, self.config['seq_len'])

        # Initialize model
        model = ProteinLTCModel(self.config).cuda()
        print(f"🧠 ProteinLTCModel initialized:")
        print(f"  📏 Sequence length: {self.config['seq_len']}")
        print(f"  🧬 AA vocab size: {self.config['aa_vocab_size']}")
        print(f"  🔬 3Di vocab size: {self.config['struct_vocab_size']}")
        print(f"  🧠 Hidden dimensions: {self.config['hidden_dim']}")
        print(f"  🏗️  LTC layers: {self.config['num_layers']}")

        # Auto-detect batch size
        sample_data = torch.zeros(32, self.config['seq_len'], dtype=torch.long)
        batch_size = self.auto_batch_size(model, sample_data)

        # Create data loaders optimized for CUDA
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Optimal for CUDA
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        # Initialize adaptive weights
        adaptive_weights = AdaptiveWeights(
            len(REAL_FOLDSEEK_3DI_ALPHABET),
            adaptive_weights_tensor
        ).cuda()

        print(f"🎯 AdaptiveWeights initialized with {len(REAL_FOLDSEEK_3DI_ALPHABET)} classes")

        # Optimizer with CUDA-optimized settings
        optimizer = optim.AdamW([
            {'params': model.parameters()},
            {'params': adaptive_weights.parameters(), 'lr': lr * 0.1}  # Lower LR for weights
        ], lr=lr, weight_decay=1e-4, fused=True)  # Fused optimizer for CUDA

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=lr*0.01
        )

        # Mixed precision scaler for RTX Ada
        scaler = GradScaler('cuda')

        print("🚀 Training configuration:")
        print(f"  📊 Dataset: {len(train_aa):,} training pairs")
        print(f"  🔄 Epochs: {epochs}")
        print(f"  🎯 Batch size: {batch_size}")
        print(f"  📈 Learning rate: {lr}")
        print(f"  ⚡ Mixed precision: Enabled")
        print(f"  🔧 Fused optimizer: Enabled")

        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\n📍 EPOCH {epoch+1}/{epochs}")

            # Training phase
            model.train()
            adaptive_weights.train()
            train_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, (aa_batch, struct_batch) in enumerate(progress_bar):
                aa_batch = aa_batch.cuda(non_blocking=True)
                struct_batch = struct_batch.cuda(non_blocking=True)

                optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast('cuda'):
                    # Forward pass
                    predictions = model(aa_batch)  # [batch, seq, vocab]

                    # Get adaptive weights
                    weights = adaptive_weights()  # [vocab]

                    # Compute weighted cross-entropy loss manually (CUDA optimized)
                    predictions_flat = predictions.view(-1, self.config['struct_vocab_size'])
                    targets_flat = struct_batch.view(-1)

                    # Compute log softmax
                    log_probs = F.log_softmax(predictions_flat, dim=-1)

                    # Apply adaptive weights to the loss
                    nll_loss = F.nll_loss(log_probs, targets_flat, reduction='none')

                    # Apply class-specific weights
                    class_weights = weights[targets_flat]
                    weighted_loss = nll_loss * class_weights

                    loss = weighted_loss.mean()

                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{train_loss/(batch_idx+1):.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

                # Memory cleanup every 100 batches
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()

            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            model.eval()
            adaptive_weights.eval()
            val_loss = 0.0

            with torch.no_grad():
                for aa_batch, struct_batch in tqdm(val_loader, desc="Validation"):
                    aa_batch = aa_batch.cuda(non_blocking=True)
                    struct_batch = struct_batch.cuda(non_blocking=True)

                    with autocast('cuda'):
                        predictions = model(aa_batch)
                        weights = adaptive_weights()

                        # Same loss computation as training
                        predictions_flat = predictions.view(-1, self.config['struct_vocab_size'])
                        targets_flat = struct_batch.view(-1)

                        log_probs = F.log_softmax(predictions_flat, dim=-1)
                        nll_loss = F.nll_loss(log_probs, targets_flat, reduction='none')
                        class_weights = weights[targets_flat]
                        weighted_loss = nll_loss * class_weights
                        loss = weighted_loss.mean()

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # Update learning rate
            scheduler.step()

            # Record history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            print(f"  📊 Train Loss: {avg_train_loss:.4f}")
            print(f"  📊 Val Loss: {avg_val_loss:.4f}")
            print(f"  📈 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print("  💾 Saving best model...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'adaptive_weights_state_dict': adaptive_weights.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'config': self.config
                }, f"/tmp/best_cuda_ltc_model.pth")

            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                print("  💾 Saving checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'adaptive_weights_state_dict': adaptive_weights.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'config': self.config,
                    'history': history
                }, f"/tmp/cuda_ltc_checkpoint_epoch_{epoch+1}.pth")

        print("\n🎉 CUDA TRAINING COMPLETE!")
        print(f"✅ Best validation loss: {best_val_loss:.4f}")

        return model, adaptive_weights, history

if __name__ == "__main__":
    print("🚀 ULTIMATE CUDA TRAINER")
    print("="*80)

    # CUDA-optimized configuration for RTX 16GB
    cuda_config = SPLINE_CONFIG.copy()
    cuda_config.update({
        'seq_len': 512,
        'aa_vocab_size': len(AA_TO_IDX),
        'struct_vocab_size': len(REAL_FOLDSEEK_3DI_ALPHABET),
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout_rate': 0.1
    })

    try:
        # Initialize trainer
        trainer = UltimateCudaTrainer(cuda_config)

        # Start training
        print("🚀 Starting Ultimate CUDA Training!")
        model, adaptive_weights, history = trainer.train()

        print("\n🎉 CUDA TRAINING SUCCESS!")
        print("="*60)
        print("✅ Model training completed")
        print("✅ Best model saved")
        print("✅ Checkpoints created")
        print("\n🚀 Ready for high-speed protein folding on RTX Ada!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()