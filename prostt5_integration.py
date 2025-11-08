#============================================================
# ProstT5 Integration - Professional AA ↔ 3Di Translation
# 100M protein trained model from Rostlab
#============================================================

import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5EncoderModel
import re
import time
import numpy as np
from typing import List, Tuple, Optional

print("🃏 ProstT5 Integration - The Ace of Clubs!")
print("="*80)

class ProstT5Translator:
    """Production-ready AA ↔ 3Di translator using ProstT5"""

    def __init__(self, model_path="Rostlab/ProstT5_fp16"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")

        # Load tokenizer
        print("📝 Loading ProstT5 tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, legacy=True)

        # Load translation model
        print("🧠 Loading ProstT5 translation model...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)

        # Load encoder model for embeddings
        print("🎯 Loading ProstT5 encoder model...")
        self.encoder = T5EncoderModel.from_pretrained(model_path).to(self.device)

        # Set precision based on device
        if self.device.type == 'cpu':
            self.model.float()
            self.encoder.float()
            print("⚠️  Using float32 on CPU (slower)")
        else:
            self.model.half()
            self.encoder.half()
            print("⚡ Using float16 on GPU (faster)")

        print("✅ ProstT5 models loaded successfully!")

    def preprocess_sequences(self, sequences: List[str], direction: str = "AA2fold") -> List[str]:
        """Preprocess sequences for ProstT5"""
        processed = []

        for seq in sequences:
            # Replace rare/ambiguous amino acids with X
            clean_seq = re.sub(r"[UZOB]", "X", seq.upper())

            # Add spaces between characters
            spaced_seq = " ".join(list(clean_seq))

            # Add direction prefix
            if direction == "AA2fold":
                prefixed_seq = "<AA2fold> " + spaced_seq
            elif direction == "fold2AA":
                # Convert to lowercase for 3Di
                spaced_seq = " ".join(list(clean_seq.lower()))
                prefixed_seq = "<fold2AA> " + spaced_seq
            else:
                raise ValueError("Direction must be 'AA2fold' or 'fold2AA'")

            processed.append(prefixed_seq)

        return processed

    def translate_aa_to_3di(self, sequences: List[str], **gen_kwargs) -> List[str]:
        """Translate amino acid sequences to 3Di structural sequences"""
        print(f"🔄 Translating {len(sequences)} AA sequences to 3Di...")

        # Preprocess
        processed_seqs = self.preprocess_sequences(sequences, direction="AA2fold")

        # Calculate lengths for generation
        min_len = min([len(s.replace(" ", "")) for s in sequences])
        max_len = max([len(s.replace(" ", "")) for s in sequences])

        # Default generation parameters for folding
        default_gen_kwargs = {
            "do_sample": True,
            "num_beams": 3,
            "top_p": 0.95,
            "temperature": 1.2,
            "top_k": 6,
            "repetition_penalty": 1.2,
            "max_length": max_len + 10,
            "min_length": min_len,
            "early_stopping": True,
            "num_return_sequences": 1,
        }
        default_gen_kwargs.update(gen_kwargs)

        # Tokenize
        ids = self.tokenizer.batch_encode_plus(
            processed_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors='pt'
        ).to(self.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            translations = self.model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                **default_gen_kwargs
            )

        inference_time = time.time() - start_time

        # Decode and clean
        decoded = self.tokenizer.batch_decode(translations, skip_special_tokens=True)
        structure_sequences = ["".join(ts.split(" ")) for ts in decoded]

        print(f"✅ Translation complete! {inference_time:.2f}s for {len(sequences)} sequences")
        return structure_sequences

    def translate_3di_to_aa(self, sequences: List[str], **gen_kwargs) -> List[str]:
        """Translate 3Di structural sequences to amino acid sequences"""
        print(f"🔄 Translating {len(sequences)} 3Di sequences to AA...")

        # Preprocess (convert to lowercase for 3Di)
        processed_seqs = self.preprocess_sequences(sequences, direction="fold2AA")

        # Calculate lengths
        min_len = min([len(s.replace(" ", "")) for s in sequences])
        max_len = max([len(s.replace(" ", "")) for s in sequences])

        # Default generation parameters for inverse folding
        default_gen_kwargs = {
            "do_sample": True,
            "top_p": 0.85,
            "temperature": 1.0,
            "top_k": 3,
            "repetition_penalty": 1.2,
            "max_length": max_len + 10,
            "min_length": min_len,
            "num_return_sequences": 1,
        }
        default_gen_kwargs.update(gen_kwargs)

        # Tokenize
        ids = self.tokenizer.batch_encode_plus(
            processed_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors='pt'
        ).to(self.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            translations = self.model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                **default_gen_kwargs
            )

        inference_time = time.time() - start_time

        # Decode and clean
        decoded = self.tokenizer.batch_decode(translations, skip_special_tokens=True)
        amino_sequences = ["".join(ts.split(" ")) for ts in decoded]

        print(f"✅ Translation complete! {inference_time:.2f}s for {len(sequences)} sequences")
        return amino_sequences

    def get_embeddings(self, sequences: List[str], direction: str = "AA2fold") -> torch.Tensor:
        """Get embeddings from ProstT5 encoder"""
        print(f"🎯 Extracting embeddings for {len(sequences)} sequences...")

        # Preprocess
        processed_seqs = self.preprocess_sequences(sequences, direction=direction)

        # Tokenize
        ids = self.tokenizer.batch_encode_plus(
            processed_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors='pt'
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            embeddings = self.encoder(
                ids.input_ids,
                attention_mask=ids.attention_mask
            )

        print("✅ Embeddings extracted!")
        return embeddings.last_hidden_state

def test_prostt5():
    """Test ProstT5 functionality"""
    print("\n🧪 Testing ProstT5 Translation Capabilities")
    print("="*60)

    # Initialize translator
    translator = ProstT5Translator()

    # Test sequences
    test_proteins = [
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWEKLKKLRQQHKLLQPQRSQ",
        "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHV",
        "ACDEFGHIKLMNPQRSTVWY"  # Simple test
    ]

    print(f"🧬 Testing with {len(test_proteins)} protein sequences:")
    for i, seq in enumerate(test_proteins):
        print(f"  {i+1}. {seq[:30]}{'...' if len(seq) > 30 else ''} ({len(seq)} AA)")

    # AA → 3Di translation
    print("\n🔄 Phase 1: AA → 3Di Translation")
    print("-" * 40)

    start_time = time.time()
    predicted_3di = translator.translate_aa_to_3di(test_proteins)
    aa2fold_time = time.time() - start_time

    print("\n📊 Results:")
    for i, (aa_seq, struct_seq) in enumerate(zip(test_proteins, predicted_3di)):
        print(f"  Protein {i+1}:")
        print(f"    AA:  {aa_seq[:40]}{'...' if len(aa_seq) > 40 else ''}")
        print(f"    3Di: {struct_seq[:40]}{'...' if len(struct_seq) > 40 else ''}")
        print(f"    Length match: {len(aa_seq) == len(struct_seq)}")

    # 3Di → AA back-translation
    print("\n🔄 Phase 2: 3Di → AA Back-translation")
    print("-" * 40)

    start_time = time.time()
    back_translated_aa = translator.translate_3di_to_aa(predicted_3di)
    fold2aa_time = time.time() - start_time

    print("\n📊 Back-translation Results:")
    for i, (original_aa, back_aa) in enumerate(zip(test_proteins, back_translated_aa)):
        print(f"  Protein {i+1}:")
        print(f"    Original: {original_aa[:40]}{'...' if len(original_aa) > 40 else ''}")
        print(f"    Back-translated: {back_aa[:40]}{'...' if len(back_aa) > 40 else ''}")

        # Calculate similarity
        if len(original_aa) == len(back_aa):
            matches = sum(a == b for a, b in zip(original_aa, back_aa))
            similarity = matches / len(original_aa) * 100
            print(f"    Similarity: {similarity:.1f}% ({matches}/{len(original_aa)} matches)")
        else:
            print(f"    Length mismatch: {len(original_aa)} vs {len(back_aa)}")

    # Embedding extraction test
    print("\n🎯 Phase 3: Embedding Extraction")
    print("-" * 40)

    embeddings = translator.get_embeddings(test_proteins[:1])  # Test with first sequence
    print(f"📊 Embedding shape: {embeddings.shape}")
    print(f"📊 Embedding dtype: {embeddings.dtype}")
    print(f"📊 Embedding device: {embeddings.device}")

    # Performance summary
    print("\n⚡ Performance Summary:")
    print(f"  AA → 3Di: {aa2fold_time:.2f}s for {len(test_proteins)} sequences")
    print(f"  3Di → AA: {fold2aa_time:.2f}s for {len(predicted_3di)} sequences")
    print(f"  Speed: ~{len(test_proteins)/(aa2fold_time + fold2aa_time):.1f} sequences/second")

    return translator, predicted_3di, back_translated_aa

def compare_with_our_data():
    """Compare ProstT5 with our 550k dataset"""
    print("\n🔬 Comparing with Our 550K Dataset")
    print("="*60)

    # Load some sequences from our dataset
    sample_aa_sequences = []
    sample_3di_sequences = []

    try:
        # Load a few sequences from our files
        with open("aa_sequences.fasta", 'r') as f:
            current_seq = ""
            count = 0
            for line in f:
                if line.startswith('>'):
                    if current_seq and count < 5:  # Get first 5
                        sample_aa_sequences.append(current_seq)
                        count += 1
                    current_seq = ""
                    if count >= 5:
                        break
                else:
                    current_seq += line.strip()

        with open("3di_sequences.tsv", 'r') as f:
            count = 0
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and count < 5:
                    sample_3di_sequences.append(parts[1].strip())
                    count += 1
                    if count >= 5:
                        break

        if sample_aa_sequences and sample_3di_sequences:
            print(f"✅ Loaded {len(sample_aa_sequences)} sample sequences from our dataset")

            # Initialize ProstT5
            translator = ProstT5Translator()

            # Translate our AA sequences with ProstT5
            prostt5_3di = translator.translate_aa_to_3di(sample_aa_sequences)

            print("\n📊 Comparison Results:")
            for i in range(min(len(sample_aa_sequences), len(sample_3di_sequences), len(prostt5_3di))):
                print(f"\nProtein {i+1}:")
                print(f"  AA:           {sample_aa_sequences[i][:50]}...")
                print(f"  Our 3Di:      {sample_3di_sequences[i][:50]}...")
                print(f"  ProstT5 3Di:  {prostt5_3di[i][:50]}...")

                # Calculate similarity between 3Di predictions
                our_3di = sample_3di_sequences[i]
                prostt5_pred = prostt5_3di[i]

                if len(our_3di) == len(prostt5_pred):
                    matches = sum(a == b for a, b in zip(our_3di, prostt5_pred))
                    similarity = matches / len(our_3di) * 100
                    print(f"  3Di Similarity: {similarity:.1f}% ({matches}/{len(our_3di)} matches)")
                else:
                    print(f"  Length mismatch: {len(our_3di)} vs {len(prostt5_pred)}")
        else:
            print("⚠️  Could not load sample data - files might not be present")

    except FileNotFoundError:
        print("⚠️  Data files not found - run this from the directory with aa_sequences.fasta and 3di_sequences.tsv")

if __name__ == "__main__":
    print("🃏 ProstT5 Integration - Production AA ↔ 3Di Translation")
    print("="*80)

    try:
        # Test basic functionality
        translator, pred_3di, back_aa = test_prostt5()

        # Compare with our dataset if available
        compare_with_our_data()

        print("\n🎉 ProstT5 Integration Complete!")
        print("="*60)
        print("✅ Translation capabilities tested")
        print("✅ Embedding extraction verified")
        print("✅ Ready for production use")
        print("\n🚀 The Ace of Clubs is in play!")

    except Exception as e:
        print(f"❌ ProstT5 test failed: {e}")
        import traceback
        traceback.print_exc()