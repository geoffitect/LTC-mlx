#============================================================
# Correct Sequence Alignment
# Fix the indexing between AA and 3Di sequences using TSV index
#============================================================

from typing import List, Tuple, Dict


print("🔧 Correct Sequence Alignment - Index-Based Matching")
print("=" * 60)

class CorrectSequenceAligner:
    """Properly align AA and 3Di sequences using TSV index"""

    def __init__(self):
        self.aa_sequences_dict = {}
        self.three_di_sequences_dict = {}
        self.aligned_pairs = []

    def load_aa_sequences_indexed(self, fasta_file: str) -> Dict[int, str]:
        """Load AA sequences and create an index-based dictionary"""

        print("📖 Loading AA sequences with proper indexing...")
        sequences_dict = {}

        with open(fasta_file, 'r') as f:
            sequence_index = 0
            current_seq = []

            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_seq:
                        sequences_dict[sequence_index] = ''.join(current_seq)
                        sequence_index += 1
                    current_seq = []
                else:
                    current_seq.append(line)

            # Save last sequence
            if current_seq:
                sequences_dict[sequence_index] = ''.join(current_seq)

        print(f"✅ Loaded {len(sequences_dict)} indexed AA sequences")
        return sequences_dict

    def load_3di_sequences_indexed(self, tsv_file: str) -> Dict[int, str]:
        """Load 3Di sequences from TSV with explicit indexing"""

        print("📖 Loading 3Di sequences from indexed TSV...")
        sequences_dict = {}

        with open(tsv_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    index = int(parts[0])  # Use explicit index from TSV
                    three_di_seq = parts[1]
                    sequences_dict[index] = three_di_seq

        print(f"✅ Loaded {len(sequences_dict)} indexed 3Di sequences")
        return sequences_dict

    def align_sequences_by_index(self, max_pairs: int = 100000) -> List[Tuple[str, str]]:
        """Properly align sequences using correct indexing"""

        print(f"🔗 Aligning sequences by correct index (max: {max_pairs:,})...")

        # Load sequences with proper indexing
        self.aa_sequences_dict = self.load_aa_sequences_indexed("/tmp/aa_sequences.fasta")
        self.three_di_sequences_dict = self.load_3di_sequences_indexed("/tmp/3di_sequences.tsv")

        print(f"📊 Index alignment summary:")
        print(f"  AA sequences: {len(self.aa_sequences_dict):,}")
        print(f"  3Di sequences: {len(self.three_di_sequences_dict):,}")

        # Find common indices
        aa_indices = set(self.aa_sequences_dict.keys())
        three_di_indices = set(self.three_di_sequences_dict.keys())
        common_indices = aa_indices & three_di_indices

        print(f"  Common indices: {len(common_indices):,}")

        if len(common_indices) == 0:
            print("❌ No common indices found!")
            return []

        # Sample some indices to check alignment
        sample_indices = sorted(list(common_indices))[:5]
        print(f"\n🔍 Sample alignment check:")

        for idx in sample_indices:
            aa_seq = self.aa_sequences_dict[idx]
            three_di_seq = self.three_di_sequences_dict[idx]

            print(f"  Index {idx}:")
            print(f"    AA  ({len(aa_seq):3d}): {aa_seq[:50]}...")
            print(f"    3Di ({len(three_di_seq):3d}): {three_di_seq[:50]}...")
            print(f"    Length diff: {len(aa_seq) - len(three_di_seq)}")

        # Create aligned pairs with filtering
        aligned_pairs = []
        stats = {
            'total_common': len(common_indices),
            'length_filtered': 0,
            'char_filtered': 0,
            'length_mismatch': 0,
            'valid_pairs': 0
        }

        from utils.aa2fold_trainer import REAL_FOLDSEEK_3DI_ALPHABET, AA_TO_IDX
        valid_3di_chars = set(REAL_FOLDSEEK_3DI_ALPHABET)

        count = 0
        for idx in sorted(common_indices):
            if count >= max_pairs:
                break

            aa_seq = self.aa_sequences_dict[idx]
            three_di_seq = self.three_di_sequences_dict[idx]

            # Apply quality filters
            if not (30 <= len(aa_seq) <= 500):
                stats['length_filtered'] += 1
                continue

            if len(aa_seq) != len(three_di_seq):
                stats['length_mismatch'] += 1
                continue

            # Check valid characters
            if any(aa not in AA_TO_IDX for aa in aa_seq):
                stats['char_filtered'] += 1
                continue

            if any(char not in valid_3di_chars for char in three_di_seq):
                stats['char_filtered'] += 1
                continue

            # Valid pair
            aligned_pairs.append((aa_seq, three_di_seq))
            stats['valid_pairs'] += 1
            count += 1

        print(f"\n📋 Correct alignment results:")
        print(f"  ✅ Total common indices: {stats['total_common']:,}")
        print(f"  ✅ Valid pairs: {stats['valid_pairs']:,}")
        print(f"  ⚠️  Length filtered: {stats['length_filtered']:,}")
        print(f"  ⚠️  Length mismatched: {stats['length_mismatch']:,}")
        print(f"  ⚠️  Character filtered: {stats['char_filtered']:,}")

        # Calculate improvement over our previous attempts
        previous_best = 52032  # Our best previous result
        if stats['valid_pairs'] > previous_best:
            improvement = stats['valid_pairs'] / previous_best
            print(f"\n🚀 MAJOR IMPROVEMENT:")
            print(f"  Previous best: {previous_best:,} pairs")
            print(f"  Current result: {stats['valid_pairs']:,} pairs")
            print(f"  Improvement: {improvement:.1f}x ({improvement*100-100:+.0f}%)")
        else:
            print(f"\n📊 Comparison with previous:")
            print(f"  Previous: {previous_best:,}, Current: {stats['valid_pairs']:,}")

        self.aligned_pairs = aligned_pairs
        return aligned_pairs

def main():
    """Run correct sequence alignment"""

    aligner = CorrectSequenceAligner()
    aligned_pairs = aligner.align_sequences_by_index(max_pairs=100000)

    if aligned_pairs:
        print(f"\n🎯 SUCCESS: Found {len(aligned_pairs):,} correctly aligned pairs!")

        # Save the correctly aligned pairs
        import pickle
        with open("/tmp/correctly_aligned_pairs.pkl", 'wb') as f:
            pickle.dump(aligned_pairs, f)

        print(f"💾 Saved correctly aligned pairs to /tmp/correctly_aligned_pairs.pkl")

        # Show first few examples
        print(f"\n🔬 First 3 correctly aligned pairs:")
        for i, (aa, three_di) in enumerate(aligned_pairs[:3]):
            print(f"  {i+1}. AA ({len(aa)}):  {aa[:60]}...")
            print(f"     3Di ({len(three_di)}): {three_di[:60]}...")
            print(f"     Match: {len(aa) == len(three_di)}")
    else:
        print(f"❌ No correctly aligned pairs found!")

    print(f"\n" + "=" * 60)
    print(f"🔧 Correct Sequence Alignment Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()