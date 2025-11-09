#!/usr/bin/env python3
"""
Quick Validation Test - Test the trained model on a few samples
"""

import torch
import torch.nn.functional as F
import random
import sys

# Import from the training script
sys.path.append('.')
from ultimate_pytorch_trainer import ProteinLTCModel, PYTORCH_CONFIG
from enhanced_real_trainer_fixed import REAL_FOLDSEEK_3DI_ALPHABET, REAL_FOLDSEEK_3DI_TO_IDX
from spline.sequence_to_3di import AA_TO_IDX, AMINO_ACIDS

print("🧪 Quick Validation Test")
print("=" * 50)

def load_sample_sequences(n=10):
    """Load a few sample sequences"""
    print(f"📖 Loading {n} sample sequences...")

    aa_dict = {}
    struct_dict = {}

    # Load AA sequences
    with open("aa_sequences.fasta", 'r') as f:
        current_header = None
        current_seq = ""
        count = 0

        for line in f:
            if line.startswith('>'):
                if current_header and current_seq and count < n:
                    header_id = current_header[1:].split()[0]
                    aa_dict[header_id] = current_seq
                    count += 1
                current_header = line.strip()
                current_seq = ""
                if count >= n:
                    break
            else:
                current_seq += line.strip()

    # Load 3Di sequences
    with open("3di_sequences.tsv", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0] in aa_dict:
                struct_dict[parts[0]] = parts[1]

    # Create matched pairs
    pairs = []
    for seq_id in aa_dict:
        if seq_id in struct_dict:
            aa_seq = aa_dict[seq_id]
            struct_seq = struct_dict[seq_id]

            # Filter for reasonable length and valid characters (relaxed)
            if (50 <= len(aa_seq) <= 400 and
                len(aa_seq) == len(struct_seq)):
                # Just check if most characters are valid
                valid_aa = sum(1 for aa in aa_seq if aa in AMINO_ACIDS) / len(aa_seq)
                valid_struct = sum(1 for s in struct_seq if s in REAL_FOLDSEEK_3DI_ALPHABET) / len(struct_seq)

                if valid_aa > 0.9 and valid_struct > 0.9:  # 90% valid characters
                    pairs.append((aa_seq, struct_seq))

    print(f"✅ Loaded {len(pairs)} valid sequence pairs")
    return pairs

def predict_sequence(model, aa_sequence):
    """Predict 3Di from AA sequence"""
    model.eval()

    # Convert to indices
    aa_indices = [AA_TO_IDX.get(aa, 0) for aa in aa_sequence]

    # Pad to model length
    max_len = PYTORCH_CONFIG['seq_len']
    original_length = len(aa_indices)

    if len(aa_indices) > max_len:
        aa_indices = aa_indices[:max_len]
        original_length = max_len
    else:
        aa_indices.extend([0] * (max_len - len(aa_indices)))

    # Create tensor
    input_tensor = torch.tensor(aa_indices, dtype=torch.long).unsqueeze(0)

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=-1)[0, :original_length]

    # Convert back to 3Di
    idx_to_3di = {i: char for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}
    predicted_3di = ''.join([idx_to_3di[idx.item()] for idx in predictions])

    return predicted_3di

def main():
    print("🔄 Loading trained model...")

    try:
        # Load the model
        model = ProteinLTCModel(PYTORCH_CONFIG)
        checkpoint = torch.load("ultimate_pytorch_checkpoints/pytorch_checkpoint_epoch_5.pt", map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded! Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

        # Load sample sequences
        pairs = load_sample_sequences(10)

        if not pairs:
            print("❌ No valid sequences found")
            return

        print(f"\n🎯 Testing on {len(pairs)} sequences:")
        print("=" * 80)

        total_accuracy = 0
        vocab_usage = {char: 0 for char in REAL_FOLDSEEK_3DI_ALPHABET}

        for i, (aa_seq, true_3di) in enumerate(pairs):
            # Predict
            pred_3di = predict_sequence(model, aa_seq)

            # Calculate accuracy
            if len(pred_3di) == len(true_3di):
                matches = sum(1 for a, b in zip(true_3di, pred_3di) if a == b)
                accuracy = matches / len(true_3di)
                total_accuracy += accuracy
            else:
                accuracy = 0

            # Count vocabulary usage
            for char in pred_3di:
                if char in vocab_usage:
                    vocab_usage[char] += 1

            # Display results
            print(f"\n📊 Sequence {i+1}:")
            print(f"   Length: {len(aa_seq)} residues")
            print(f"   AA:       {aa_seq[:60]}{'...' if len(aa_seq) > 60 else ''}")
            print(f"   True 3Di: {true_3di[:60]}{'...' if len(true_3di) > 60 else ''}")
            print(f"   Pred 3Di: {pred_3di[:60]}{'...' if len(pred_3di) > 60 else ''}")
            print(f"   🎯 Accuracy: {accuracy:.1%} ({int(accuracy * len(true_3di))}/{len(true_3di)} matches)")

            if accuracy > 0.5:
                print(f"   🎉 EXCELLENT!")
            elif accuracy > 0.3:
                print(f"   ✅ GOOD!")
            elif accuracy > 0.15:
                print(f"   📈 FAIR")
            else:
                print(f"   📉 POOR")

        # Overall stats
        avg_accuracy = total_accuracy / len(pairs)
        total_chars = sum(vocab_usage.values())
        vocab_coverage = sum(1 for count in vocab_usage.values() if count > 0) / 20

        print(f"\n📈 OVERALL RESULTS:")
        print(f"   🎯 Average Accuracy: {avg_accuracy:.1%}")
        print(f"   📚 Vocabulary Coverage: {vocab_coverage:.1%} ({sum(1 for count in vocab_usage.values() if count > 0)}/20 chars)")

        print(f"\n📝 Character Usage Distribution:")
        for char in REAL_FOLDSEEK_3DI_ALPHABET:
            percentage = (vocab_usage[char] / total_chars * 100) if total_chars > 0 else 0
            print(f"   {char}: {percentage:5.1f}% ({vocab_usage[char]} uses)")

        # Performance assessment
        if avg_accuracy > 0.7:
            print(f"\n🚀 OUTSTANDING: Model is production ready!")
        elif avg_accuracy > 0.5:
            print(f"\n🎉 EXCELLENT: Very strong performance!")
        elif avg_accuracy > 0.3:
            print(f"\n✅ GOOD: Solid performance!")
        elif avg_accuracy > 0.15:
            print(f"\n📈 FAIR: Learning well, continue training")
        else:
            print(f"\n📉 POOR: Needs more training")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()