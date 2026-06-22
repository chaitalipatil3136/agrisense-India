"""
AgriSense India — PlantVillage Dataset Organizer (Day 7)
File: notebooks/04_organize_plantvillage.py

Splits PlantVillage images 70% train / 20% val / 10% test.
Handles both folder structures:
  A) data/raw/plantvillage/color/  (38 class folders inside color/)
  B) data/raw/plantvillage/        (38 class folders directly)

Run ONCE on Day 7. Takes 5–20 minutes depending on disk speed.

Run: python notebooks/04_organize_plantvillage.py
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────
SPLITS      = {"train": 0.70, "val": 0.20, "test": 0.10}
RANDOM_SEED = 42
VALID_EXT   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
OUT_BASE    = Path("data/plantvillage")

print("=" * 58)
print("AgriSense India — PlantVillage Organizer")
print("=" * 58)

# ── Step 1: Find the raw PlantVillage folder ──────────────
CANDIDATES = [
    Path("data/raw/plantvillage/color"),      # Structure A
    Path("data/raw/plantvillage"),            # Structure B
    Path("data/raw/Plant_leave_diseases_dataset_without_augmentation"),
    Path("data/raw/PlantVillage"),
]

raw_base = None
for c in CANDIDATES:
    if c.exists():
        # Check it actually has image subfolders
        subfolders = [d for d in c.iterdir() if d.is_dir()]
        if len(subfolders) >= 10:
            raw_base = c
            break

if raw_base is None:
    print("\nERROR: PlantVillage raw folder not found.")
    print("Expected one of:")
    for c in CANDIDATES:
        print(f"  {c}")
    print("\nMake sure the Kaggle download is complete and extracted.")
    print("Dataset: kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    exit(1)

print(f"\nFound raw data at: {raw_base}")

# ── Step 2: Discover all class folders ────────────────────
class_folders = sorted([
    d for d in raw_base.iterdir()
    if d.is_dir() and not d.name.startswith(".")
])

print(f"Found {len(class_folders)} disease classes")

if len(class_folders) < 10:
    print("ERROR: Too few class folders. Check your extraction.")
    print("Expected 38 folders, one per disease class.")
    exit(1)

# Count images per class
total_images = 0
class_info = []
for cf in class_folders:
    images = [f for f in cf.iterdir() if f.suffix in VALID_EXT]
    class_info.append({"class": cf.name, "total": len(images)})
    total_images += len(images)
    print(f"  {cf.name[:55]:55s} — {len(images):5d} images")

print(f"\nTotal images found: {total_images:,}")

if total_images < 1000:
    print("WARNING: Very few images found. Extraction may be incomplete.")

# ── Step 3: Create output directories ─────────────────────
print("\nCreating output split directories...")
for split in SPLITS:
    for cf in class_folders:
        (OUT_BASE / split / cf.name).mkdir(parents=True, exist_ok=True)

# ── Step 4: Split and copy ────────────────────────────────
print(f"\nSplitting {total_images:,} images ({int(SPLITS['train']*100)}/{int(SPLITS['val']*100)}/{int(SPLITS['test']*100)} train/val/test)...")
print("This takes 5–20 minutes. Do not interrupt.\n")

random.seed(RANDOM_SEED)
results = []

for i, cf in enumerate(class_folders, 1):
    images = [f for f in cf.iterdir() if f.suffix in VALID_EXT]
    random.shuffle(images)

    n     = len(images)
    n_tr  = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    n_te  = n - n_tr - n_val

    assignments = (
        ["train"] * n_tr +
        ["val"]   * n_val +
        ["test"]  * n_te
    )

    counts = {"class": cf.name, "total": n, "train": 0, "val": 0, "test": 0}

    for img_path, split in zip(images, assignments):
        dest = OUT_BASE / split / cf.name / img_path.name
        shutil.copy2(img_path, dest)
        counts[split] += 1

    results.append(counts)
    pct = i / len(class_folders) * 100
    print(f"  [{pct:5.1f}%] {cf.name[:45]:45s} "
          f"train={counts['train']} val={counts['val']} test={counts['test']}")

# ── Step 5: Save summary ──────────────────────────────────
results_df = pd.DataFrame(results)
summary_path = OUT_BASE / "split_counts.csv"
results_df.to_csv(summary_path, index=False)

print("\n" + "=" * 58)
print("PlantVillage Split Complete!")
print("=" * 58)
print(f"  Classes     : {len(results_df)}")
print(f"  Train images: {results_df['train'].sum():,}")
print(f"  Val images  : {results_df['val'].sum():,}")
print(f"  Test images : {results_df['test'].sum():,}")
print(f"  Total       : {results_df['total'].sum():,}")
print(f"\nSaved split counts: {summary_path}")
print("\nReady for CNN training in Week 3!")
print("Week 1 is officially complete. Commit to GitHub now.")
