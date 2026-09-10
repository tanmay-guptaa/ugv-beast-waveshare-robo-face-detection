"""
view_database.py
================
Quick viewer script to inspect all enrolled persons, metadata,
and raw 128-D face embeddings stored in the database.

Usage:
    python view_database.py
"""

import os
import json
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "dataset", "database.pkl")
JSON_PATH = os.path.join(BASE_DIR, "database.json")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "dataset", "known_faces")

print("=" * 65)
print("             UGV BEAST - FACE EMBEDDINGS DATABASE")
print("=" * 65)

# 1. Metadata JSON
print(f"\n[1] METADATA FILE: {JSON_PATH}")
if os.path.exists(JSON_PATH):
    try:
        with open(JSON_PATH, "r") as f:
            metadata = json.load(f)
        print(f"    Total Registered Profiles: {len(metadata)}")
        for i, person in enumerate(metadata, 1):
            print(f"    {i}. Name: {person.get('name')} | Role: {person.get('role')} | Access: {person.get('access_level')}")
            print(f"       Info: {person.get('info')} | Enrolled: {person.get('first_seen')}")
    except Exception as e:
        print(f"    Error reading JSON: {e}")
else:
    print("    File does not exist yet.")

# 2. Embeddings PKL
print(f"\n[2] EMBEDDINGS DATABASE: {PKL_PATH}")
if os.path.exists(PKL_PATH):
    try:
        with open(PKL_PATH, "rb") as f:
            embeddings_db = pickle.load(f)
        
        print(f"    Total People with Embeddings: {len(embeddings_db)}")
        for name, vectors in embeddings_db.items():
            print("\n" + "-" * 55)
            print(f"    [Person]: {name.upper()}")
            print(f"       Total Embedding Vectors: {len(vectors)}")
            for idx, vec in enumerate(vectors, 1):
                if isinstance(vec, np.ndarray):
                    # Flatten vector to 1D if shape is (1, 128)
                    flat = vec.flatten()
                    sample_vals = [f"{v:.4f}" for v in flat[:8]]
                    print(f"       Vector #{idx}: shape={vec.shape}, dtype={vec.dtype}")
                    print(f"         First 8 dims : [{', '.join(sample_vals)}, ...]")
                    print(f"         Vector Norm  : {np.linalg.norm(flat):.4f} (L2-normalized)")
            print("-" * 55)
    except Exception as e:
        print(f"    Error reading PKL: {e}")
else:
    print("    File does not exist yet.")

# 3. Reference Images
print(f"\n[3] ARCHIVED FACE IMAGES: {KNOWN_FACES_DIR}")
if os.path.isdir(KNOWN_FACES_DIR):
    folders = [d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]
    for folder in folders:
        fpath = os.path.join(KNOWN_FACES_DIR, folder)
        imgs = [f for f in os.listdir(fpath) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"    - {folder}/: {len(imgs)} photos ({', '.join(imgs[:5])}{'...' if len(imgs) > 5 else ''})")
else:
    print("    Folder does not exist yet.")

print("\n" + "=" * 65)
