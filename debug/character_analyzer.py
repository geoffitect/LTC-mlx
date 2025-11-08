#============================================================
# Character Set Analysis
# Find out what 3Di characters are actually in the data
#============================================================

from collections import Counter
from sequence_to_3di import FOLDSEEK_3DI_TO_IDX

print("🔤 3Di Character Analysis")
print("=" * 50)

def analyze_3di_characters():
    """Analyze actual characters in the 3Di sequences"""

    print("📖 Loading and analyzing 3Di character usage...")

    all_3di_chars = []

    # Sample first 10k sequences to understand character distribution
    with open("/tmp/3di_sequences.tsv", 'r') as f:
        for i, line in enumerate(f):
            if i >= 10000:  # Analyze first 10k sequences
                break

            parts = line.strip().split('\t')
            if len(parts) >= 2:
                three_di_seq = parts[1]
                all_3di_chars.extend(list(three_di_seq))

    # Count character usage
    char_counts = Counter(all_3di_chars)

    print(f"📊 Character analysis from {i+1:,} sequences:")
    print(f"  Total 3Di characters found: {len(all_3di_chars):,}")
    print(f"  Unique characters: {len(char_counts)}")

    print(f"\n📋 Character frequency (top 30):")
    for char, count in char_counts.most_common(30):
        print(f"  '{char}': {count:,} ({count/len(all_3di_chars)*100:.1f}%)")

    # Compare with our expected character set
    expected_chars = set(FOLDSEEK_3DI_TO_IDX.keys())
    actual_chars = set(char_counts.keys())

    print(f"\n🔍 Character Set Comparison:")
    print(f"  Expected 3Di characters: {len(expected_chars)}")
    print(f"  Expected: {sorted(expected_chars)}")
    print(f"  Actual characters found: {len(actual_chars)}")
    print(f"  Actual: {sorted(actual_chars)}")

    # Find missing and unexpected characters
    missing_chars = expected_chars - actual_chars
    unexpected_chars = actual_chars - expected_chars

    if missing_chars:
        print(f"  ❌ Missing from data: {sorted(missing_chars)}")
    if unexpected_chars:
        print(f"  ⚠️  Unexpected in data: {sorted(unexpected_chars)}")

    overlap = expected_chars & actual_chars
    print(f"  ✅ Valid overlap: {len(overlap)}/{len(expected_chars)} chars")

    # Estimate how many sequences would be valid with different character sets
    valid_with_current = 0
    valid_with_actual = 0

    print(f"\n🧮 Validation simulation (first 1000 sequences):")
    with open("/tmp/3di_sequences.tsv", 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break

            parts = line.strip().split('\t')
            if len(parts) >= 2:
                three_di_seq = parts[1]

                # Current validation
                if not any(char not in expected_chars for char in three_di_seq):
                    valid_with_current += 1

                # Relaxed validation (using actual character set)
                if not any(char not in actual_chars for char in three_di_seq):
                    valid_with_actual += 1

    print(f"  Current strict validation: {valid_with_current}/1000 ({valid_with_current/10:.1f}%)")
    print(f"  Using actual char set: {valid_with_actual}/1000 ({valid_with_actual/10:.1f}%)")

    return actual_chars, char_counts

if __name__ == "__main__":
    actual_chars, char_counts = analyze_3di_characters()

    print(f"\n💡 Recommendations:")
    print(f"  1. Update FOLDSEEK_3DI_TO_IDX to include actual characters")
    print(f"  2. This should increase valid pairs from ~10.8% to ~{len(char_counts)/10:.1f}%")
    print(f"  3. Potential 8-9x increase in available training data!")