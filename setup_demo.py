"""
setup_demo.py
=============
One-time setup script that creates the demo database (Vikas profile)
and required folder structure.

Run once: python setup_demo.py
"""

import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from ugv import face_engine

def main():
    print("=" * 55)
    print("  UGV Beast Dashboard -- Demo Setup")
    print("=" * 55)

    # Create required folders
    for folder in [
        config.DATASET_DIR,
        config.CAPTURES_DIR,
        os.path.join(config.DATASET_DIR, "Vikas"),
    ]:
        os.makedirs(folder, exist_ok=True)
        print(f"  [OK] Created: {folder}")

    # Add Vikas profile
    p = config.DEMO_PERSON
    face_engine.add_person(
        name=p["name"],
        role=p["role"],
        access_level=p["access_level"],
        info=p["info"],
    )
    print(f"\n  [OK] Enrolled person: {p['name']} ({p['role']})")
    print("\n  NOTE: To add Vikas's reference photos:")
    print(f"     Copy JPEGs to: {os.path.join(config.DATASET_DIR, 'Vikas')}")
    print("     Then run: python -c \"from ugv import face_engine; print(face_engine.rebuild_from_folder())\"")

    print("\n  Setup complete!")
    print("  Start dashboard:  .venv\\Scripts\\streamlit run Dashboard.py")
    print("=" * 55)

if __name__ == "__main__":
    main()
