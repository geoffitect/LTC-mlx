#!/usr/bin/env python3
"""
Simultaneous Training: AA→3Di and 3Di→Coords
End-to-end protein folding pipeline with rotation-invariant coordinates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
from tqdm import tqdm
import time

# Import our existing components
import sys
sys.path.append('.')
from ultimate_pytorch_trainer import ProteinLTCModel, PYTORCH_CONFIG
from enhanced_real_trainer_fixed import REAL_FOLDSEEK_3DI_ALPHABET, REAL_FOLDSEEK_3DI_TO_IDX
from spline.sequence_to_3di import AA_TO_IDX, AMINO_ACIDS
from threeddi_to_coords import ThreeDiToCoords, RotationInvariantEncoder, COORDS_CONFIG

print("🧬 Simultaneous AA→3Di + 3Di→Coords Training Pipeline")
print("=" * 70)

class EndToEndProteinFolder(nn.Module):
    """
    Complete end-to-end protein folding model:
    Amino Acids → 3Di Structural Tokens → Backbone Coordinates
    """

    def __init__(self, aa_to_3di_config: dict, coords_config: dict):
        super().__init__()

        self.aa_to_3di_config = aa_to_3di_config
        self.coords_config = coords_config

        # Module 1: AA → 3Di
        self.aa_to_3di = ProteinLTCModel(aa_to_3di_config)

        # Module 2: 3Di → Coordinates
        self.threeddi_to_coords = ThreeDiToCoords(coords_config)

        # Rotation-invariant encoder for target coordinates
        self.coord_encoder = RotationInvariantEncoder()

        print(f"🧠 End-to-End Model initialized:")
        print(f"  📏 Sequence length: {aa_to_3di_config['seq_len']}")
        print(f"  🔤 AA vocab: {aa_to_3di_config['aa_vocab_size']}")
        print(f"  🏗️  3Di vocab: {aa_to_3di_config['struct_vocab_size']}")
        print(f"  🧠 Hidden dims: {coords_config['hidden_dim']}")

    def forward(self, aa_sequence: torch.Tensor, target_coords: Optional[torch.Tensor] = None):
        """
        Forward pass: AA → 3Di → Coordinates

        Args:
            aa_sequence: [batch_size, seq_len] amino acid indices
            target_coords: [batch_size, seq_len, 3] target backbone coordinates (for training)

        Returns:
            Dictionary with predictions and losses
        """
        batch_size, seq_len = aa_sequence.shape
        device = aa_sequence.device

        # Module 1: Predict 3Di from AA sequence
        aa_to_3di_logits = self.aa_to_3di(aa_sequence)  # [batch_size, seq_len, 3di_vocab_size]
        predicted_3di = torch.argmax(aa_to_3di_logits, dim=-1)  # [batch_size, seq_len]

        # Module 2: Predict coordinates from 3Di tokens
        coord_predictions = self.threeddi_to_coords(predicted_3di)

        results = {
            'aa_to_3di_logits': aa_to_3di_logits,
            'predicted_3di': predicted_3di,
            'coordinate_predictions': coord_predictions
        }

        # Calculate losses if target data is provided
        if target_coords is not None:
            # Encode target coordinates to rotation-invariant features
            target_features = self.coord_encoder.coords_to_internal(target_coords)

            # Calculate coordinate reconstruction loss
            coord_loss = self.threeddi_to_coords.calculate_loss(
                coord_predictions, target_features
            )

            results['coordinate_loss'] = coord_loss

        return results

    def calculate_total_loss(self, predictions: dict, targets: dict, weights: dict = None):
        """Calculate weighted total loss for both modules"""

        if weights is None:
            weights = {'aa_to_3di': 1.0, 'coordinates': 2.0}

        total_loss = 0.0
        loss_components = {}

        # AA → 3Di loss
        if 'true_3di' in targets:
            aa_to_3di_loss = F.cross_entropy(
                predictions['aa_to_3di_logits'].view(-1, predictions['aa_to_3di_logits'].size(-1)),
                targets['true_3di'].view(-1)
            )
            loss_components['aa_to_3di'] = aa_to_3di_loss
            total_loss += weights['aa_to_3di'] * aa_to_3di_loss

        # Coordinate loss
        if 'coordinate_loss' in predictions:
            coord_loss = predictions['coordinate_loss']
            loss_components['coordinates'] = coord_loss
            total_loss += weights['coordinates'] * coord_loss

        loss_components['total'] = total_loss
        return total_loss, loss_components

class SimultaneousDataLoader:
    """
    Data loader for simultaneous training on AA, 3Di, and coordinate data
    """

    def __init__(self, batch_size: int = 32, max_length: int = 512):
        self.batch_size = batch_size
        self.max_length = max_length

        # Load sequence data
        print("📖 Loading sequence data...")
        self.aa_sequences, self.threeddi_sequences = self._load_sequences()

        # Load coordinate data
        print("📖 Loading coordinate data...")
        self.coordinate_data = self._load_coordinates()

        # Create aligned dataset
        print("🔗 Aligning datasets...")
        self.aligned_data = self._align_datasets()

        print(f"✅ Data loader ready: {len(self.aligned_data)} aligned samples")

    def _load_sequences(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load AA and 3Di sequences"""
        aa_dict = {}
        struct_dict = {}

        # Load AA sequences
        try:
            with open("aa_sequences.fasta", 'r') as f:
                current_header = None
                current_seq = ""

                for line in f:
                    if line.startswith('>'):
                        if current_header and current_seq:
                            header_id = current_header.split()[0]
                            if header_id.startswith('AF-'):
                                parts = header_id.split('-')
                                if len(parts) >= 2:
                                    uniprot_id = parts[1]
                                    aa_dict[uniprot_id] = current_seq
                            else:
                                aa_dict[header_id] = current_seq
                        current_header = line.strip()[1:]
                        current_seq = ""
                    else:
                        current_seq += line.strip()
        except FileNotFoundError:
            print("⚠️  aa_sequences.fasta not found")

        # Load 3Di sequences
        try:
            with open("3di_sequences.tsv", 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        seq_id = parts[0].strip()
                        if seq_id.startswith('AF-'):
                            parts_id = seq_id.split('-')
                            if len(parts_id) >= 2:
                                uniprot_id = parts_id[1]
                                struct_dict[uniprot_id] = parts[1].strip()
                        else:
                            struct_dict[seq_id] = parts[1].strip()
        except FileNotFoundError:
            print("⚠️  3di_sequences.tsv not found")

        return aa_dict, struct_dict

    def _load_coordinates(self) -> Dict[str, np.ndarray]:
        """Load backbone coordinates"""
        try:
            with open("backbone_coordinates.pkl", 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("⚠️  backbone_coordinates.pkl not found - coordinate training disabled")
            return {}

    def _align_datasets(self) -> List[Dict]:
        """Align all three datasets by sequence ID"""
        aligned = []

        for seq_id in self.aa_sequences:
            if (seq_id in self.threeddi_sequences and
                (not self.coordinate_data or seq_id in self.coordinate_data)):

                aa_seq = self.aa_sequences[seq_id]
                struct_seq = self.threeddi_sequences[seq_id]
                coords = self.coordinate_data.get(seq_id, None)

                # Validate lengths match
                if len(aa_seq) == len(struct_seq):
                    if coords is None or len(coords) == len(aa_seq):
                        aligned.append({
                            'id': seq_id,
                            'aa_sequence': aa_seq,
                            'threeddi_sequence': struct_seq,
                            'coordinates': coords
                        })

        return aligned

    def __len__(self):
        return len(self.aligned_data) // self.batch_size

    def get_batch(self, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Get a batch of aligned data"""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.aligned_data))
        batch_data = self.aligned_data[start_idx:end_idx]

        # Convert sequences to tensors
        aa_sequences = []
        threeddi_sequences = []
        coordinates = []
        valid_coords = False

        for item in batch_data:
            # Convert AA sequence
            aa_indices = [AA_TO_IDX.get(aa, 0) for aa in item['aa_sequence'][:self.max_length]]
            aa_indices += [0] * (self.max_length - len(aa_indices))  # Pad
            aa_sequences.append(aa_indices)

            # Convert 3Di sequence
            struct_indices = [REAL_FOLDSEEK_3DI_TO_IDX.get(s, 0) for s in item['threeddi_sequence'][:self.max_length]]
            struct_indices += [0] * (self.max_length - len(struct_indices))  # Pad
            threeddi_sequences.append(struct_indices)

            # Coordinates (if available)
            if item['coordinates'] is not None:
                coords = item['coordinates'][:self.max_length]
                # Pad coordinates
                padded_coords = np.zeros((self.max_length, 3))
                padded_coords[:len(coords)] = coords
                coordinates.append(padded_coords)
                valid_coords = True
            else:
                coordinates.append(np.zeros((self.max_length, 3)))

        batch = {
            'aa_sequences': torch.tensor(aa_sequences, dtype=torch.long),
            'true_3di': torch.tensor(threeddi_sequences, dtype=torch.long),
        }

        if valid_coords:
            batch['coordinates'] = torch.tensor(coordinates, dtype=torch.float32)

        return batch

def train_simultaneous_model(model: EndToEndProteinFolder, data_loader: SimultaneousDataLoader,
                           num_epochs: int = 10, learning_rate: float = 0.0001):
    """Train the end-to-end model simultaneously"""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Training on device: {device}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {'aa_to_3di': 0.0, 'coordinates': 0.0, 'total': 0.0}
        num_batches = len(data_loader)

        print(f"\n🏃‍♂️ Epoch {epoch+1}/{num_epochs}")

        pbar = tqdm(range(num_batches), desc=f"Training Epoch {epoch+1}")

        for batch_idx in pbar:
            batch = data_loader.get_batch(batch_idx)

            # Move to device
            for key in batch:
                batch[key] = batch[key].to(device)

            optimizer.zero_grad()

            # Forward pass
            target_coords = batch.get('coordinates', None)
            predictions = model(batch['aa_sequences'], target_coords)

            # Calculate loss
            targets = {'true_3di': batch['true_3di']}
            total_loss, loss_components = model.calculate_total_loss(predictions, targets)

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update metrics
            for key in epoch_losses:
                if key in loss_components:
                    epoch_losses[key] += loss_components[key].item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'AA→3Di': f"{loss_components.get('aa_to_3di', 0):.4f}",
                'Coords': f"{loss_components.get('coordinates', 0):.4f}"
            })

        # Epoch summary
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        print(f"📊 Epoch {epoch+1} Summary:")
        print(f"  📈 Total Loss: {epoch_losses['total']:.4f}")
        print(f"  🔤 AA→3Di Loss: {epoch_losses['aa_to_3di']:.4f}")
        print(f"  📍 Coordinates Loss: {epoch_losses['coordinates']:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"simultaneous_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total']
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

def test_end_to_end_inference(model: EndToEndProteinFolder, test_sequence: str):
    """Test end-to-end inference: AA → 3Di → Coordinates"""

    print(f"\n🧪 Testing end-to-end inference")
    print(f"📝 Test sequence: {test_sequence[:50]}{'...' if len(test_sequence) > 50 else ''}")

    device = next(model.parameters()).device
    model.eval()

    # Convert sequence to tensor
    aa_indices = [AA_TO_IDX.get(aa, 0) for aa in test_sequence[:512]]
    aa_indices += [0] * (512 - len(aa_indices))  # Pad
    aa_tensor = torch.tensor([aa_indices], dtype=torch.long).to(device)

    with torch.no_grad():
        start_time = time.time()
        predictions = model(aa_tensor)
        inference_time = time.time() - start_time

    # Decode predictions
    predicted_3di_indices = predictions['predicted_3di'][0][:len(test_sequence)].cpu().numpy()
    idx_to_3di = {i: char for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}
    predicted_3di = ''.join([idx_to_3di.get(idx, 'X') for idx in predicted_3di_indices])

    print(f"📊 Results:")
    print(f"  🔤 Input AA:   {test_sequence[:50]}{'...' if len(test_sequence) > 50 else ''}")
    print(f"  🏗️  Predicted 3Di: {predicted_3di[:50]}{'...' if len(predicted_3di) > 50 else ''}")
    print(f"  ⚡ Inference time: {inference_time:.3f}s")
    print(f"  📏 Length: {len(test_sequence)} residues")

    # Extract coordinate predictions
    coord_preds = predictions['coordinate_predictions']
    if 'coords_direct' in coord_preds:
        coords = coord_preds['coords_direct'][0][:len(test_sequence)].cpu().numpy()
        print(f"  📍 First CA: ({coords[0][0]:.2f}, {coords[0][1]:.2f}, {coords[0][2]:.2f})")
        print(f"  📍 Last CA:  ({coords[-1][0]:.2f}, {coords[-1][1]:.2f}, {coords[-1][2]:.2f})")

    return predictions

def main():
    """Main training function"""

    print("🚀 Starting simultaneous training pipeline...")

    # Initialize model
    model = EndToEndProteinFolder(PYTORCH_CONFIG, COORDS_CONFIG)

    # Initialize data loader
    data_loader = SimultaneousDataLoader(batch_size=16, max_length=512)

    if len(data_loader.aligned_data) == 0:
        print("❌ No aligned training data available")
        return

    # Train model
    train_simultaneous_model(model, data_loader, num_epochs=20, learning_rate=0.0001)

    # Test inference
    test_sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWEKLKKLRQQHKLLQPQRSQ"
    test_end_to_end_inference(model, test_sequence)

    print("\n🎉 Simultaneous training complete!")
    print("📁 Checkpoint files saved for both modules")
    print("🚀 Ready for end-to-end protein structure prediction!")

if __name__ == "__main__":
    main()