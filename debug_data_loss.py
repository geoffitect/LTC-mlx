#!/usr/bin/env python3
"""
Debug script to understand why 80% of data is being filtered out
"""

import random
from enhanced_real_trainer_fixed import REAL_FOLDSEEK_3DI_ALPHABET, REAL_FOLDSEEK_3DI_TO_IDX
from spline.sequence_to_3di import AA_TO_IDX

print("🔍 DEBUGGING DATA LOSS - Why are we losing 80% of sequences?")
print("=" * 80)

# Load data
aa_sequences = []
struct_sequences = []

# Load amino acid sequences from FASTA
print("📖 Loading amino acid sequences...")
with open("aa_sequences.fasta", 'r') as f:
    current_seq = ""
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                aa_sequences.append(current_seq)
            current_seq = ""
        else:
            current_seq += line
    if current_seq:
        aa_sequences.append(current_seq)

# Load 3Di sequences from TSV
print("📖 Loading 3Di sequences...")
with open("3di_sequences.tsv", 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            struct_seq = parts[1].strip()
            struct_sequences.append(struct_seq)

print(f"✅ Loaded {len(aa_sequences):,} AA sequences")
print(f"✅ Loaded {len(struct_sequences):,} 3Di sequences")

# Sample for analysis
sample_size = 1000
indices = random.sample(range(min(len(aa_sequences), len(struct_sequences))), sample_size)

print(f"\n🔬 Analyzing {sample_size} random pairs:")
print("=" * 50)

stats = {
    'total': 0,
    'empty_aa': 0,
    'empty_struct': 0,
    'length_mismatch': 0,
    'invalid_3di_chars': 0,
    'invalid_aa_chars': 0,
    'valid': 0
}

failed_examples = {
    'length_mismatch': [],
    'invalid_3di_chars': [],
    'invalid_aa_chars': []
}

for i in indices:
    aa_seq = aa_sequences[i]
    struct_seq = struct_sequences[i]
    stats['total'] += 1

    # Check each validation step
    if len(aa_seq) == 0:
        stats['empty_aa'] += 1
        continue

    if len(struct_seq) == 0:
        stats['empty_struct'] += 1
        continue

    if len(aa_seq) != len(struct_seq):
        stats['length_mismatch'] += 1
        if len(failed_examples['length_mismatch']) < 5:
            failed_examples['length_mismatch'].append({
                'aa_len': len(aa_seq),
                'struct_len': len(struct_seq),
                'aa_seq': aa_seq[:50] + '...' if len(aa_seq) > 50 else aa_seq,
                'struct_seq': struct_seq[:50] + '...' if len(struct_seq) > 50 else struct_seq
            })
        continue

    # Check 3Di characters
    invalid_3di = [char for char in struct_seq if char not in REAL_FOLDSEEK_3DI_TO_IDX]
    if invalid_3di:
        stats['invalid_3di_chars'] += 1
        if len(failed_examples['invalid_3di_chars']) < 5:
            unique_invalid = list(set(invalid_3di))
            failed_examples['invalid_3di_chars'].append({
                'invalid_chars': unique_invalid[:10],  # First 10 unique
                'count': len(invalid_3di),
                'struct_seq': struct_seq[:100] + '...' if len(struct_seq) > 100 else struct_seq
            })
        continue

    # Check AA characters
    invalid_aa = [char for char in aa_seq if char.upper() not in AA_TO_IDX]
    if invalid_aa:
        stats['invalid_aa_chars'] += 1
        if len(failed_examples['invalid_aa_chars']) < 5:
            unique_invalid = list(set(invalid_aa))
            failed_examples['invalid_aa_chars'].append({
                'invalid_chars': unique_invalid[:10],
                'count': len(invalid_aa),
                'aa_seq': aa_seq[:100] + '...' if len(aa_seq) > 100 else aa_seq
            })
        continue

    stats['valid'] += 1

print("📊 VALIDATION STATISTICS:")
print(f"  Total analyzed: {stats['total']:,}")
print(f"  Valid pairs: {stats['valid']:,} ({stats['valid']/stats['total']*100:.1f}%)")
print(f"  Empty AA sequences: {stats['empty_aa']:,}")
print(f"  Empty struct sequences: {stats['empty_struct']:,}")
print(f"  Length mismatches: {stats['length_mismatch']:,} ({stats['length_mismatch']/stats['total']*100:.1f}%)")
print(f"  Invalid 3Di characters: {stats['invalid_3di_chars']:,} ({stats['invalid_3di_chars']/stats['total']*100:.1f}%)")
print(f"  Invalid AA characters: {stats['invalid_aa_chars']:,} ({stats['invalid_aa_chars']/stats['total']*100:.1f}%)")

print(f"\n🔍 FAILURE EXAMPLES:")
print("=" * 50)

if failed_examples['length_mismatch']:
    print(f"\n📏 LENGTH MISMATCHES (showing {len(failed_examples['length_mismatch'])} examples):")
    for i, ex in enumerate(failed_examples['length_mismatch']):
        print(f"  {i+1}. AA:{ex['aa_len']} vs 3Di:{ex['struct_len']}")
        print(f"     AA:  {ex['aa_seq']}")
        print(f"     3Di: {ex['struct_seq']}")

if failed_examples['invalid_3di_chars']:
    print(f"\n🔤 INVALID 3DI CHARACTERS (showing {len(failed_examples['invalid_3di_chars'])} examples):")
    for i, ex in enumerate(failed_examples['invalid_3di_chars']):
        print(f"  {i+1}. Invalid chars: {ex['invalid_chars']} (count: {ex['count']})")
        print(f"     3Di: {ex['struct_seq']}")

if failed_examples['invalid_aa_chars']:
    print(f"\n🧬 INVALID AA CHARACTERS (showing {len(failed_examples['invalid_aa_chars'])} examples):")
    for i, ex in enumerate(failed_examples['invalid_aa_chars']):
        print(f"  {i+1}. Invalid chars: {ex['invalid_chars']} (count: {ex['count']})")
        print(f"     AA: {ex['aa_seq']}")

print(f"\n🎯 EXPECTED SUCCESS RATE: {stats['valid']/stats['total']*100:.1f}%")
print("   (This should match the ~20% we're seeing)")

print(f"\n🔬 3DI ALPHABET CHECK:")
print(f"   Expected: {sorted(REAL_FOLDSEEK_3DI_ALPHABET)}")
print(f"   Expected count: {len(REAL_FOLDSEEK_3DI_ALPHABET)}")