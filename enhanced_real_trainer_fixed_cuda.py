#============================================================
# Enhanced Real Data Trainer - CUDA Version
# CUDA-optimized constants and utilities for protein training
#============================================================

# Real Foldseek 3Di alphabet (corrected version)
REAL_FOLDSEEK_3DI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
REAL_FOLDSEEK_3DI_TO_IDX = {char: i for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}
REAL_IDX_TO_FOLDSEEK_3DI = {i: char for i, char in enumerate(REAL_FOLDSEEK_3DI_ALPHABET)}

print("[DNA] Enhanced Real Data Trainer - CUDA Version")
print("="*70)
print(f"[OK] Using corrected 3Di alphabet: {REAL_FOLDSEEK_3DI_ALPHABET}")