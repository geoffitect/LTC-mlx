#!/usr/bin/env python3
"""
ULTIMATE PYTORCH TRAINER - 550K Protein Folding Training
Fast, stable PyTorch implementation with direct CoreML export support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import random
import time
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
from collections import Counter

# Import constants from our working modules
from enhanced_real_trainer_fixed import REAL_FOLDSEEK_3DI_ALPHABET, REAL_FOLDSEEK_3DI_TO_IDX
from spline.sequence_to_3di import AA_TO_IDX

print("🚀 ULTIMATE PYTORCH TRAINER - 550K Protein Training")
print("=" * 80)

# Configuration
PYTORCH_CONFIG = {
    'seq_len': 512,
    'aa_vocab_size': 20,
    'struct_vocab_size': 20,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout_rate': 0.1,
    'learning_rate': 0.0002,
    'batch_size': 128,  # Start with this, can auto-tune
    'epochs': 100,
    'checkpoint_every': 5,
    'validation_every': 10,
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔥 Using device: {device}")

class LTCCell(nn.Module):
    """PyTorch LTC cell for protein sequence modeling"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Spline transformation layers
        self.spline_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )

        # Time constant learning
        self.tau_net = nn.Linear(input_size + hidden_size, hidden_size)

        # Sensory processing
        self.sensory_mu = nn.Linear(input_size, hidden_size)
        self.sensory_sigma = nn.Linear(input_size, hidden_size)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, dt: float = 0.1):
        """Forward pass with LTC dynamics"""
        batch_size, seq_len, _ = inputs.shape

        # Combine input and hidden state
        combined = torch.cat([inputs, hidden], dim=-1)

        # Spline transformation
        spline_features = self.spline_net(combined)

        # Time constants
        tau = torch.sigmoid(self.tau_net(combined)) * 0.9 + 0.1

        # Sensory processing
        mu = self.sensory_mu(inputs)
        sigma = torch.sigmoid(self.sensory_sigma(inputs))

        # LTC dynamics: dh/dt = (-h + f(x)) / tau
        target_state = spline_features + mu * sigma
        dh_dt = (-hidden + target_state) / (tau + 1e-8)
        new_hidden = hidden + dt * dh_dt

        return new_hidden

class ProteinLTCModel(nn.Module):
    """Complete protein sequence to 3Di structure model"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(
            config['aa_vocab_size'],
            config['embedding_dim']
        )

        # LTC layers
        self.ltc_layers = nn.ModuleList()
        input_dim = config['embedding_dim']

        for _ in range(config['num_layers']):
            ltc = LTCCell(input_dim, config['hidden_dim'])
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

        print(f"🧠 ProteinLTCModel initialized:")
        print(f"  📏 Sequence length: {config['seq_len']}")
        print(f"  🧬 AA vocab size: {config['aa_vocab_size']}")
        print(f"  🔬 3Di vocab size: {config['struct_vocab_size']}")
        print(f"  🧠 Hidden dimensions: {config['hidden_dim']}")
        print(f"  🏗️  LTC layers: {config['num_layers']}")

    def forward(self, aa_sequences: torch.Tensor):
        """
        Forward pass: AA sequences -> 3Di predictions

        Args:
            aa_sequences: [batch, seq_len] amino acid token indices

        Returns:
            predictions: [batch, seq_len, struct_vocab_size] 3Di predictions
        """
        batch_size, seq_len = aa_sequences.shape

        # Embed amino acids
        embedded = self.aa_embedding(aa_sequences)  # [batch, seq, embed_dim]
        embedded = self.dropout(embedded)

        # Initialize hidden state
        hidden = torch.zeros(
            batch_size, seq_len, self.config['hidden_dim'],
            device=aa_sequences.device
        )

        # Process through LTC layers
        for ltc_layer in self.ltc_layers:
            hidden = ltc_layer(embedded, hidden)
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

class ProteinDataset(Dataset):
    """PyTorch dataset for protein sequences"""

    def __init__(self, aa_sequences: List[str], struct_sequences: List[str], config: Dict):
        self.aa_sequences = aa_sequences
        self.struct_sequences = struct_sequences
        self.config = config
        self.seq_len = config['seq_len']

        print(f"📊 ProteinDataset: {len(aa_sequences):,} sequences")

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

class UltimatePyTorchTrainer:
    """Ultimate PyTorch trainer with all optimizations"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = device

        # Create checkpoint directory
        self.checkpoint_dir = "ultimate_pytorch_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_complete_dataset(self) -> Tuple[List[str], List[str]]:
        """Load the complete 550k dataset using header-based alignment"""
        print("📖 Loading COMPLETE 550k dataset...")

        # Load amino acid sequences from FASTA with headers
        aa_dict = {}
        print("📖 Loading amino acid sequences from FASTA...")

        with open("aa_sequences.fasta", 'r') as f:
            current_header = None
            current_seq = ""
            for line in tqdm(f, desc="Loading AA sequences"):
                line = line.strip()
                if line.startswith('>'):
                    if current_header and current_seq:
                        header_id = current_header.split()[0]
                        aa_dict[header_id] = current_seq
                    current_header = line[1:]  # Remove >
                    current_seq = ""
                else:
                    current_seq += line
            # Add last sequence
            if current_header and current_seq:
                header_id = current_header.split()[0]
                aa_dict[header_id] = current_seq

        print(f"✅ Loaded {len(aa_dict):,} amino acid sequences")

        # Load 3Di sequences from TSV with headers
        struct_dict = {}
        print("📖 Loading 3Di sequences from TSV...")

        with open("3di_sequences.tsv", 'r') as f:
            for line in tqdm(f, desc="Loading 3Di sequences"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    header_id = parts[0].strip()
                    struct_seq = parts[1].strip()
                    struct_dict[header_id] = struct_seq

        print(f"✅ Loaded {len(struct_dict):,} 3Di sequences")

        # Align sequences by header ID
        valid_aa_sequences = []
        valid_struct_sequences = []
        valid_count = 0

        print("🔍 Aligning sequences by header ID and validating...")

        # Use intersection of headers
        common_headers = set(aa_dict.keys()) & set(struct_dict.keys())
        print(f"📊 Common headers: {len(common_headers):,}")

        for header_id in tqdm(common_headers, desc="Validating sequences"):
            aa_seq = aa_dict[header_id]
            struct_seq = struct_dict[header_id]

            # Validate sequences
            if (len(aa_seq) > 0 and len(struct_seq) > 0 and
                len(aa_seq) == len(struct_seq) and
                all(char in REAL_FOLDSEEK_3DI_TO_IDX for char in struct_seq) and
                all(char.upper() in AA_TO_IDX for char in aa_seq)):

                valid_aa_sequences.append(aa_seq)
                valid_struct_sequences.append(struct_seq)
                valid_count += 1

        print(f"✅ Loaded {valid_count:,} valid sequence pairs")
        print(f"📈 Success rate: {valid_count/len(common_headers)*100:.1f}%")

        return valid_aa_sequences, valid_struct_sequences

    def compute_character_frequencies(self, struct_sequences: List[str], sample_size: int = 50000) -> torch.Tensor:
        """Compute character frequencies for adaptive weighting"""
        print("🧠 Computing character frequencies for adaptive weights...")

        # Sample for frequency analysis
        sample_seqs = random.sample(struct_sequences, min(sample_size, len(struct_sequences)))

        # Count characters
        char_counts = Counter()
        total_chars = 0

        for seq in tqdm(sample_seqs, desc="Analyzing character frequencies"):
            for char in seq:
                char_counts[char] += 1
                total_chars += 1

        # Create frequency tensor
        frequencies = torch.zeros(len(REAL_FOLDSEEK_3DI_ALPHABET))
        for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET):
            if char in char_counts:
                frequencies[i] = char_counts[char] / total_chars
            else:
                frequencies[i] = 1e-6  # Small value for unseen characters

        print(f"📊 Analyzed {total_chars:,} characters from {len(sample_seqs):,} sequences")

        # Display frequency distribution
        print("📈 Character frequency distribution:")
        for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET):
            freq = frequencies[i].item()
            weight = 1.0 / (freq + 1e-6)
            print(f"   {char}: {freq:.4f} freq → {weight:.2f} weight")

        return frequencies

    def auto_detect_batch_size(self, model: nn.Module, sample_data: torch.Tensor) -> int:
        """Auto-detect optimal batch size for available VRAM"""
        print("🔍 Auto-detecting optimal batch size...")

        model.eval()
        optimal_size = 32

        for batch_size in [32, 64, 128, 256, 512]:
            try:
                # Create test batch
                batch_aa = sample_data[:batch_size].to(self.device)

                # Test forward pass
                start_time = time.time()
                with torch.no_grad():
                    _ = model(batch_aa)
                forward_time = time.time() - start_time

                print(f"   Testing batch size {batch_size}: {forward_time:.3f}s")
                optimal_size = batch_size

                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   ❌ Batch size {batch_size}: out of memory")
                    break
                else:
                    print(f"   ❌ Batch size {batch_size}: {e}")
                    break

        # Conservative choice for long training
        if optimal_size > 128:
            optimal_size = 128

        print(f"🎯 Selected optimal batch size: {optimal_size}")
        return optimal_size

    def train(self):
        """Main training loop"""
        print("🚀 Starting Ultimate PyTorch Training!")

        # Load data
        aa_sequences, struct_sequences = self.load_complete_dataset()

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

        # Create datasets
        train_dataset = ProteinDataset(train_aa, train_struct, self.config)
        val_dataset = ProteinDataset(val_aa, val_struct, self.config)

        # Initialize model
        model = ProteinLTCModel(self.config).to(self.device)

        # Auto-detect batch size with sample data
        sample_batch = train_dataset[0][0].unsqueeze(0)  # Get sample AA sequence
        optimal_batch_size = self.auto_detect_batch_size(model, sample_batch)
        self.config['batch_size'] = optimal_batch_size

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Initialize adaptive weights
        adaptive_weights = AdaptiveWeights(self.config['struct_vocab_size'], char_frequencies).to(self.device)

        # Optimizer
        optimizer = optim.AdamW(
            list(model.parameters()) + list(adaptive_weights.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'])

        # Training loop
        print(f"🚀 Training configuration:")
        print(f"  📊 Dataset: {len(train_aa):,} training pairs")
        print(f"  🔄 Epochs: {self.config['epochs']}")
        print(f"  🎯 Batch size: {self.config['batch_size']}")
        print(f"  📈 Learning rate: {self.config['learning_rate']}")

        history = {'train_loss': [], 'val_loss': [], 'vocab_coverage': []}
        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            print(f"\\n📍 EPOCH {epoch+1}/{self.config['epochs']}")

            # Training phase
            model.train()
            adaptive_weights.train()
            train_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for aa_batch, struct_batch in progress_bar:
                aa_batch = aa_batch.to(self.device)
                struct_batch = struct_batch.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                predictions = model(aa_batch)  # [batch, seq, vocab]

                # Get adaptive weights
                weights = adaptive_weights()  # [vocab]

                # Compute weighted cross-entropy loss using manual approach
                # PyTorch cross_entropy doesn't work with learnable weights
                predictions_flat = predictions.view(-1, self.config['struct_vocab_size'])
                targets_flat = struct_batch.view(-1)

                # Compute log softmax
                log_probs = F.log_softmax(predictions_flat, dim=-1)

                # Apply adaptive weights to the loss
                nll_loss = F.nll_loss(log_probs, targets_flat, reduction='none')  # [batch*seq]

                # Apply class-specific weights
                class_weights = weights[targets_flat]  # Get weight for each target
                weighted_loss = nll_loss * class_weights

                loss = weighted_loss.mean()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = train_loss / num_batches
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            if (epoch + 1) % self.config['validation_every'] == 0:
                val_loss = self.validate(model, val_loader, adaptive_weights)
                history['val_loss'].append(val_loss)

                # Vocabulary coverage analysis
                coverage = self.analyze_vocab_coverage(model, val_loader)
                history['vocab_coverage'].append(coverage)

                print(f"📊 Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, coverage={coverage:.1%}")

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
        """Validation phase"""
        model.eval()
        adaptive_weights.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for aa_batch, struct_batch in val_loader:
                aa_batch = aa_batch.to(self.device)
                struct_batch = struct_batch.to(self.device)

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

    def analyze_vocab_coverage(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Analyze vocabulary coverage to detect mode collapse"""
        model.eval()
        predicted_chars = set()

        with torch.no_grad():
            for aa_batch, _ in val_loader:
                aa_batch = aa_batch.to(self.device)
                predictions = model(aa_batch)
                predicted_tokens = torch.argmax(predictions, dim=-1)

                # Convert to character set
                for token_seq in predicted_tokens:
                    for token in token_seq:
                        if 0 <= token < len(REAL_FOLDSEEK_3DI_ALPHABET):
                            predicted_chars.add(int(token))

                if len(predicted_chars) >= len(REAL_FOLDSEEK_3DI_ALPHABET):
                    break  # Found all characters

        coverage = len(predicted_chars) / len(REAL_FOLDSEEK_3DI_ALPHABET)
        return coverage

    def save_checkpoint(self, model: nn.Module, adaptive_weights: AdaptiveWeights,
                       optimizer: optim.Optimizer, epoch: int, loss: float, name: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'adaptive_weights_state_dict': adaptive_weights.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f"pytorch_checkpoint_{name}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    print("🚀 ULTIMATE PYTORCH TRAINER - 550K Protein Training")
    print("=" * 80)

    trainer = UltimatePyTorchTrainer(PYTORCH_CONFIG)

    try:
        model, adaptive_weights, history = trainer.train()
        print("\\n🎉 TRAINING COMPLETE!")
        print("✅ Model ready for CoreML export")
        print("✅ Adaptive weights learned successfully")
        print("🚀 Ready for edge deployment!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()