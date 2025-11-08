#============================================================
# Data Quality Investigator
# Analyze why we only get ~6k valid pairs from 550k sequences
#============================================================

import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm

# Import our constants
from sequence_to_3di import AA_TO_IDX, FOLDSEEK_3DI_TO_IDX

print("🔍 Data Quality Investigation")
print("=" * 60)

class DataQualityInvestigator:
    """Investigate data quality issues in the 550k PDB dataset"""

    def __init__(self):
        self.aa_sequences_list = []
        self.three_di_sequences_list = []

    def load_and_analyze_data(self):
        """Load data and perform comprehensive analysis"""

        print("📖 Loading data for analysis...")

        # Load amino acid sequences
        aa_sequences = []
        with open("/tmp/aa_sequences.fasta", 'r') as f:
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        aa_sequences.append(''.join(current_seq))
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                aa_sequences.append(''.join(current_seq))

        # Load 3Di sequences
        three_di_sequences = []
        with open("/tmp/3di_sequences.tsv", 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    three_di_sequences.append(parts[1])

        self.aa_sequences_list = aa_sequences
        self.three_di_sequences_list = three_di_sequences

        print(f"✅ Loaded {len(self.aa_sequences_list):,} AA sequences")
        print(f"✅ Loaded {len(self.three_di_sequences_list):,} 3Di sequences")

    def analyze_length_distributions(self):
        """Analyze the length distributions to understand mismatches"""

        print(f"\n📊 Length Distribution Analysis")
        print("=" * 40)

        # Sample sequences for analysis
        sample_size = min(10000, len(self.aa_sequences_list), len(self.three_di_sequences_list))

        aa_lengths = [len(seq) for seq in self.aa_sequences_list[:sample_size]]
        three_di_lengths = [len(seq) for seq in self.three_di_sequences_list[:sample_size]]

        print(f"📈 AA sequence lengths (first {sample_size:,}):")
        print(f"  Min: {min(aa_lengths)}, Max: {max(aa_lengths)}")
        print(f"  Mean: {np.mean(aa_lengths):.1f}, Median: {np.median(aa_lengths):.1f}")
        print(f"  Std: {np.std(aa_lengths):.1f}")

        print(f"📈 3Di sequence lengths (first {sample_size:,}):")
        print(f"  Min: {min(three_di_lengths)}, Max: {max(three_di_lengths)}")
        print(f"  Mean: {np.mean(three_di_lengths):.1f}, Median: {np.median(three_di_lengths):.1f}")
        print(f"  Std: {np.std(three_di_lengths):.1f}")

        # Check exact length matches
        length_matches = sum(1 for i in range(sample_size)
                            if len(self.aa_sequences_list[i]) == len(self.three_di_sequences_list[i]))

        print(f"📊 Length match rate: {length_matches}/{sample_size} = {length_matches/sample_size*100:.1f}%")

        return length_matches / sample_size

    def analyze_filter_effects(self, max_analyze: int = 50000):
        """Analyze the effect of each filter step"""

        print(f"\n🔍 Filter Analysis (first {max_analyze:,} sequences)")
        print("=" * 50)

        stats = {
            'total_processed': 0,
            'length_30_500': 0,
            'length_match': 0,
            'valid_aa_chars': 0,
            'valid_3di_chars': 0,
            'all_filters_passed': 0
        }

        valid_pairs = []
        valid_3di_chars = set(FOLDSEEK_3DI_TO_IDX.keys())

        min_length = min(len(self.aa_sequences_list), len(self.three_di_sequences_list), max_analyze)

        with tqdm(range(min_length), desc="Analyzing filters") as pbar:
            for i in pbar:
                stats['total_processed'] += 1

                aa_seq = self.aa_sequences_list[i]
                three_di_seq = self.three_di_sequences_list[i]

                # Filter 1: Length range 30-500
                if 30 <= len(aa_seq) <= 500:
                    stats['length_30_500'] += 1

                    # Filter 2: Length match
                    if len(aa_seq) == len(three_di_seq):
                        stats['length_match'] += 1

                        # Filter 3: Valid amino acids
                        if not any(aa not in AA_TO_IDX for aa in aa_seq):
                            stats['valid_aa_chars'] += 1

                            # Filter 4: Valid 3Di chars
                            if not any(char not in valid_3di_chars for char in three_di_seq):
                                stats['valid_3di_chars'] += 1
                                stats['all_filters_passed'] += 1
                                valid_pairs.append((aa_seq, three_di_seq))

                pbar.set_postfix(valid=stats['all_filters_passed'])

        print(f"\n📋 Filter Results:")
        print(f"  Total processed: {stats['total_processed']:,}")
        print(f"  ✅ Length 30-500: {stats['length_30_500']:,} ({stats['length_30_500']/stats['total_processed']*100:.1f}%)")
        print(f"  ✅ Length match: {stats['length_match']:,} ({stats['length_match']/stats['total_processed']*100:.1f}%)")
        print(f"  ✅ Valid AA chars: {stats['valid_aa_chars']:,} ({stats['valid_aa_chars']/stats['total_processed']*100:.1f}%)")
        print(f"  ✅ Valid 3Di chars: {stats['valid_3di_chars']:,} ({stats['valid_3di_chars']/stats['total_processed']*100:.1f}%)")
        print(f"  🎯 All filters passed: {stats['all_filters_passed']:,} ({stats['all_filters_passed']/stats['total_processed']*100:.1f}%)")

        return valid_pairs, stats

    def suggest_relaxed_filters(self, stats: Dict):
        """Suggest relaxed filtering strategies to get more pairs"""

        print(f"\n💡 Relaxed Filtering Suggestions")
        print("=" * 40)

        total = stats['total_processed']

        # Estimate gains from relaxing each filter
        print(f"Current yield: {stats['all_filters_passed']} pairs from {total:,} sequences")

        # Relaxing length range
        potential_length_relaxed = stats['length_match'] - stats['length_30_500']
        if potential_length_relaxed > 0:
            print(f"📏 Relaxing length range (20-1000): +{potential_length_relaxed:,} pairs")

        # Show bottlenecks
        bottleneck = min(stats['length_30_500'], stats['length_match'],
                        stats['valid_aa_chars'], stats['valid_3di_chars'])

        if stats['length_match'] == bottleneck:
            print(f"🚨 MAJOR BOTTLENECK: Length mismatches ({stats['length_match']}/{total})")
            print(f"   Suggests AA and 3Di sequences are not properly aligned!")

        if stats['valid_3di_chars'] < stats['valid_aa_chars']:
            print(f"🔤 3Di character validation is stricter than AA validation")

        print(f"\n🎯 Recommendations:")
        if stats['length_match'] < stats['length_30_500'] * 0.5:
            print(f"  1. Investigate AA ↔ 3Di sequence alignment")
            print(f"  2. Consider fuzzy matching or sliding window approach")
        print(f"  3. Relax length constraints to 20-1000 amino acids")
        print(f"  4. Consider approximate length matching (±5 residues)")


def main():
    """Run the data quality investigation"""

    investigator = DataQualityInvestigator()

    # Load and analyze data
    investigator.load_and_analyze_data()

    # Analyze length distributions
    match_rate = investigator.analyze_length_distributions()

    # Analyze filter effects
    valid_pairs, stats = investigator.analyze_filter_effects(max_analyze=50000)

    # Suggest improvements
    investigator.suggest_relaxed_filters(stats)

    print(f"\n🔬 Sample of valid pairs found:")
    for i, (aa, three_di) in enumerate(valid_pairs[:3]):
        print(f"  {i+1}. AA ({len(aa)}): {aa[:50]}...")
        print(f"      3Di({len(three_di)}): {three_di[:50]}...")

    print(f"\n✅ Investigation complete! Found {len(valid_pairs):,} valid pairs in analysis.")


if __name__ == "__main__":
    main()