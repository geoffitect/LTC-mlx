#!/usr/bin/env python3
"""
Extract backbone coordinates from SwissProt PDB tar file
Matches to existing sequence IDs from our AA/3Di datasets
"""

import tarfile
import gzip
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle
import os

print("🧬 Extracting Backbone Coordinates from SwissProt PDB")
print("=" * 60)

def parse_pdb_backbone(pdb_content: str) -> Optional[np.ndarray]:
    """
    Extract CA (alpha carbon) coordinates from PDB content
    Returns: numpy array of shape (seq_len, 3) for backbone coordinates
    """
    ca_coords = []

    for line in pdb_content.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            # PDB format: ATOM serial name altLoc resName chainID resSeq iCode x y z occupancy tempFactor
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ca_coords.append([x, y, z])
            except (ValueError, IndexError):
                continue

    if len(ca_coords) == 0:
        return None

    return np.array(ca_coords)

def extract_pdb_id_from_filename(filename: str) -> str:
    """Extract PDB ID from filename like 'AF-P00001-F1-model_v6.pdb.gz'"""
    # Pattern: AF-{UNIPROT_ID}-F1-model_v6.pdb.gz
    match = re.match(r'AF-([A-Z0-9]+)-F1-model_v6\.pdb\.gz', filename)
    if match:
        return match.group(1)
    return filename.replace('.pdb.gz', '').replace('.pdb', '')

def load_existing_sequence_ids() -> List[str]:
    """Load sequence IDs from our existing dataset files"""
    sequence_ids = set()

    # Load from AA sequences
    print("📖 Loading sequence IDs from aa_sequences.fasta...")
    try:
        with open("aa_sequences.fasta", 'r') as f:
            for line in f:
                if line.startswith('>'):
                    # Extract ID from header like '>AF-P00001-F1-model_v4'
                    header = line.strip()[1:]  # Remove '>'
                    seq_id = header.split()[0]  # First part before space
                    if seq_id.startswith('AF-'):
                        # Extract UniProt ID: AF-P00001-F1-model_v4 -> P00001
                        parts = seq_id.split('-')
                        if len(parts) >= 2:
                            uniprot_id = parts[1]
                            sequence_ids.add(uniprot_id)
                    else:
                        sequence_ids.add(seq_id)
    except FileNotFoundError:
        print("⚠️  aa_sequences.fasta not found")

    # Load from 3Di sequences for cross-validation
    print("📖 Loading sequence IDs from 3di_sequences.tsv...")
    try:
        with open("3di_sequences.tsv", 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    seq_id = parts[0].strip()
                    if seq_id.startswith('AF-'):
                        # Extract UniProt ID: AF-P00001-F1-model_v4 -> P00001
                        parts = seq_id.split('-')
                        if len(parts) >= 2:
                            uniprot_id = parts[1]
                            sequence_ids.add(uniprot_id)
                    else:
                        sequence_ids.add(seq_id)
    except FileNotFoundError:
        print("⚠️  3di_sequences.tsv not found")

    sequence_list = list(sequence_ids)
    print(f"✅ Found {len(sequence_list)} unique sequence IDs")
    return sequence_list

def extract_coordinates_from_tar():
    """Extract coordinates from the tar file matching our sequence IDs"""

    # Load sequence IDs we need
    target_ids = set(load_existing_sequence_ids())
    print(f"🎯 Target sequence IDs: {len(target_ids)}")

    # Output data
    coordinate_data = {}
    processed_count = 0
    matched_count = 0

    tar_path = "huge/swissprot_pdb_v6.tar"

    if not os.path.exists(tar_path):
        print(f"❌ Tar file not found: {tar_path}")
        return

    print(f"📦 Opening tar file: {tar_path}")

    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        print(f"📁 Found {len(members)} files in tar")

        # Process files with progress bar
        for member in tqdm(members, desc="Processing PDB files"):
            if member.isfile() and member.name.endswith('.pdb.gz'):
                processed_count += 1

                # Extract PDB ID from filename
                filename = os.path.basename(member.name)
                pdb_id = extract_pdb_id_from_filename(filename)

                # Check if this ID is in our target set
                if pdb_id in target_ids:
                    try:
                        # Extract and decompress file content
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            # Decompress gzipped content
                            compressed_data = file_obj.read()
                            pdb_content = gzip.decompress(compressed_data).decode('utf-8')

                            # Parse backbone coordinates
                            coords = parse_pdb_backbone(pdb_content)

                            if coords is not None and len(coords) > 0:
                                coordinate_data[pdb_id] = coords
                                matched_count += 1

                                if matched_count % 1000 == 0:
                                    print(f"  ✅ Matched {matched_count} coordinates so far...")

                    except Exception as e:
                        print(f"⚠️  Error processing {filename}: {e}")
                        continue

                # Progress update
                if processed_count % 10000 == 0:
                    print(f"  📊 Processed {processed_count} PDB files, matched {matched_count}")

    print(f"\n📊 Extraction complete!")
    print(f"  🔬 Processed: {processed_count} PDB files")
    print(f"  ✅ Matched: {matched_count} coordinates")
    print(f"  📈 Match rate: {matched_count/len(target_ids)*100:.1f}%")

    return coordinate_data

def save_coordinate_data(coordinate_data: Dict[str, np.ndarray]):
    """Save extracted coordinates to files"""

    if not coordinate_data:
        print("❌ No coordinate data to save")
        return

    print(f"\n💾 Saving {len(coordinate_data)} coordinate sets...")

    # Save as pickle for fast loading
    pickle_path = "backbone_coordinates.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(coordinate_data, f)
    print(f"✅ Saved to {pickle_path}")

    # Save as text file for inspection
    txt_path = "backbone_coordinates.txt"
    with open(txt_path, 'w') as f:
        f.write("# Backbone Coordinates Data\n")
        f.write("# Format: ID\tLength\tFirst_3_coords\n")

        for seq_id, coords in coordinate_data.items():
            first_coords = coords[:3].flatten()  # First 3 CA atoms
            coord_str = '\t'.join([f"{x:.3f}" for x in first_coords])
            f.write(f"{seq_id}\t{len(coords)}\t{coord_str}\n")

    print(f"✅ Saved summary to {txt_path}")

    # Statistics
    lengths = [len(coords) for coords in coordinate_data.values()]
    print(f"\n📊 Coordinate Statistics:")
    print(f"  📏 Average length: {np.mean(lengths):.1f} residues")
    print(f"  📏 Min length: {np.min(lengths)} residues")
    print(f"  📏 Max length: {np.max(lengths)} residues")
    print(f"  📏 Median length: {np.median(lengths):.1f} residues")

def test_coordinate_data():
    """Test loading and using the coordinate data"""

    if not os.path.exists("backbone_coordinates.pkl"):
        print("❌ No coordinate data found. Run extraction first.")
        return

    print("\n🧪 Testing coordinate data...")

    # Load coordinates
    with open("backbone_coordinates.pkl", 'rb') as f:
        coordinate_data = pickle.load(f)

    print(f"✅ Loaded {len(coordinate_data)} coordinate sets")

    # Test a few examples
    sample_ids = list(coordinate_data.keys())[:3]

    for seq_id in sample_ids:
        coords = coordinate_data[seq_id]
        print(f"\n📊 Sample: {seq_id}")
        print(f"  Length: {len(coords)} residues")
        print(f"  Shape: {coords.shape}")
        print(f"  First CA: ({coords[0][0]:.2f}, {coords[0][1]:.2f}, {coords[0][2]:.2f})")
        print(f"  Last CA:  ({coords[-1][0]:.2f}, {coords[-1][1]:.2f}, {coords[-1][2]:.2f})")

        # Test coordinate range
        coord_range = np.max(coords) - np.min(coords)
        print(f"  Coordinate range: {coord_range:.1f} Å")

def main():
    """Main execution"""

    print("🚀 Starting coordinate extraction...")

    # Extract coordinates from tar
    coordinate_data = extract_coordinates_from_tar()

    if coordinate_data:
        # Save the data
        save_coordinate_data(coordinate_data)

        # Test the data
        test_coordinate_data()

        print("\n🎉 Coordinate extraction complete!")
        print("📁 Files created:")
        print("  - backbone_coordinates.pkl (binary data)")
        print("  - backbone_coordinates.txt (text summary)")
        print("\n🚀 Ready for 3Di→coords training!")
    else:
        print("❌ No coordinates extracted")

if __name__ == "__main__":
    main()