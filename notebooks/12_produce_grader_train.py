
Copy

"""
AgriSense India — Produce Quality Grader Training
File: notebooks/12_produce_grader_train.py
 
Trains EfficientNetB0 on produce quality dataset (Healthy vs Rotten).
Maps healthy → Grade A/B and rotten/diseased → Grade C.
 
Dataset: Kaggle — "Fruit and Vegetable Disease (Healthy vs Rotten)"
Download: kaggle datasets download -d muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
Extract to: data/raw/produce_quality/
 
If dataset not available: uses PlantVillage split as fallback
(healthy classes → Grade A, diseased classes → Grade C)
 
Run: python notebooks/12_produce_grader_train.py
Expected time: 25-45 min CPU · 8-12 min Colab T4 GPU
Output: models/produce_grader.h5 · assets/grader_training_history.png
"""
 
import os
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
 
os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("data/produce_split/train/Grade_A", exist_ok=True)
os.makedirs("data/produce_split/train/Grade_B", exist_ok=True)
os.makedirs("data/produce_split/train/Grade_C", exist_ok=True)
os.makedirs("data/produce_split/val/Grade_A",   exist_ok=True)
os.makedirs("data/produce_split/val/Grade_B",   exist_ok=True)
os.makedirs("data/produce_split/val/Grade_C",   exist_ok=True)
 
print("=" * 60)
print("AgriSense India — Produce Quality Grader Training")
print("=" * 60)
 
# ── Step 1: Find dataset ─────────────────────────────────────
# Try produce quality dataset first, fall back to PlantVillage
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
VALID_EXTS = IMG_EXTS
 
PRODUCE_CANDIDATES = [
    "data/raw/produce_quality",
    "data/raw/fruit_vegetable_disease",
    "data/raw/Fruit_and_Vegetable_Disease",
]
 
PLANTVILLAGE_CANDIDATES = [
    "data/plantvillage/train",
    "data/raw/plantvillage/color",
]
 
source_type = None
source_path = None
 
# Check produce dataset
for c in PRODUCE_CANDIDATES:
    if os.path.exists(c):
        subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
        if len(subdirs) >= 4:
            source_type = "produce"
            source_path = c
            print(f"\nFound produce quality dataset: {c}")
            print(f"Classes found: {len(subdirs)}")
            break
 
# Check PlantVillage fallback
if source_type is None:
    for c in PLANTVILLAGE_CANDIDATES:
        if os.path.exists(c):
            subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
            if len(subdirs) >= 10:
                source_type = "plantvillage"
                source_path = c
                print(f"\nProduce dataset not found. Using PlantVillage fallback: {c}")
                print(f"  Healthy classes → Grade A · Diseased classes → Grade C")
                print(f"  Classes found: {len(subdirs)}")
                break
 
if source_type is None:
    print("\nERROR: No suitable dataset found.")
    print("Option 1: Download produce quality dataset:")
    print("  kaggle datasets download -d muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten")
    print("  Extract to: data/raw/produce_quality/")
    print("\nOption 2: Make sure PlantVillage split exists:")
    print("  python notebooks/04_organize_plantvillage.py")
    sys.exit(1)
 
 
# ── Step 2: Prepare split ─────────────────────────────────────
import random
random.seed(42)
 
print("\nPreparing train/val split...")
 
def copy_images_to_grade(src_folder, grade, split, max_imgs=800):
    """Copy images from source folder to grade folder."""
    dest = f"data/produce_split/{split}/Grade_{grade}"
    imgs = [
        os.path.join(src_folder, f)
        for f in os.listdir(src_folder)
        if os.path.splitext(f)[1] in IMG_EXTS
    ]
    random.shuffle(imgs)
    imgs = imgs[:max_imgs]
    for img in imgs:
        shutil.copy2(img, os.path.join(dest, os.path.basename(img)))
    return len(imgs)
 
 
if source_type == "produce":
    # Map class names to grades
    # healthy → Grade A, slightly damaged → Grade B, rotten/diseased → Grade C
    all_classes = sorted([
        d for d in os.listdir(source_path)
        if os.path.isdir(os.path.join(source_path, d))
    ])
 
    grade_a_classes = [c for c in all_classes if "healthy" in c.lower() or "fresh" in c.lower()]
    grade_c_classes = [c for c in all_classes if "rotten" in c.lower() or "disease" in c.lower() or "bad" in c.lower()]
    grade_b_classes = [c for c in all_classes if c not in grade_a_classes and c not in grade_c_classes]
 
    # If no B classes found, create B from random 30% of A images (slightly augmented)
    if not grade_b_classes:
        grade_b_classes = grade_a_classes[:max(1, len(grade_a_classes)//3)]
        print("  Note: Grade B synthesised from Grade A (partial healthy images)")
 
    print(f"  Grade A classes: {grade_a_classes[:5]}")
    print(f"  Grade B classes: {grade_b_classes[:3]}")
    print(f"  Grade C classes: {grade_c_classes[:5]}")
 
    count_a_train = 0
    for cls in grade_a_classes:
        folder = os.path.join(source_path, cls)
        all_imgs = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1] in IMG_EXTS]
        random.shuffle(all_imgs)
        split_idx = int(len(all_imgs) * 0.8)
        for img in all_imgs[:split_idx][:600]:
            shutil.copy2(img, f"data/produce_split/train/Grade_A/{cls}_{os.path.basename(img)}")
            count_a_train += 1
        for img in all_imgs[split_idx:][:150]:
            shutil.copy2(img, f"data/produce_split/val/Grade_A/{cls}_{os.path.basename(img)}")
 
    count_b_train = 0
    for cls in grade_b_classes:
        folder = os.path.join(source_path, cls)
        all_imgs = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1] in IMG_EXTS]
        random.shuffle(all_imgs)
        split_idx = int(len(all_imgs) * 0.8)
        for img in all_imgs[:split_idx][:400]:
            shutil.copy2(img, f"data/produce_split/train/Grade_B/{cls}_{os.path.basename(img)}")
            count_b_train += 1
        for img in all_imgs[split_idx:][:100]:
            shutil.copy2(img, f"data/produce_split/val/Grade_B/{cls}_{os.path.basename(img)}")
 
    count_c_train = 0
    for cls in grade_c_classes:
        folder = os.path.join(source_path, cls)
        all_imgs = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1] in IMG_EXTS]
        random.shuffle(all_imgs)
        split_idx = int(len(all_imgs) * 0.8)
        for img in all_imgs[:split_idx][:600]:
            shutil.copy2(img, f"data/produce_split/train/Grade_C/{cls}_{os.path.basename(img)}")
            count_c_train += 1
        for img in all_imgs[split_idx:][:150]:
            shutil.copy2(img, f"data/produce_split/val/Grade_C/{cls}_{os.path.basename(img)}")
 
    print(f"  Train — A:{count_a_train} B:{count_b_train} C:{count_c_train}")
 
else:
    # PlantVillage fallback
    all_classes = sorted([
        d for d in os.listdir(source_path)
        if os.path.isdir(os.path.join(source_path, d))
    ])
 
    healthy_classes  = [c for c in all_classes if "healthy" in c.lower()]
    diseased_classes = [c for c in all_classes if "healthy" not in c.lower()]
 
    print(f"  Healthy classes ({len(healthy_classes)}) → Grade A")
    print(f"  Diseased classes ({len(diseased_classes)}) → Grade C")
    print("  Grade B = 40% subsample of healthy (simulates minor defects)")
 
    count = {"A": 0, "B": 0, "C": 0}
 
    for cls in healthy_classes:
        folder = os.path.join(source_path, cls)
        imgs   = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1] in IMG_EXTS]
        random.shuffle(imgs)
        n = min(len(imgs), 300)
        # 60% → A train, 10% → A val, 20% → B train, 10% → B val (simulated)
        for img in imgs[:int(n*0.6)]:
            shutil.copy2(img, f"data/produce_split/train/Grade_A/{cls}_{os.path.basename(img)}")
            count["A"] += 1
        for img in imgs[int(n*0.6):int(n*0.7)]:
            shutil.copy2(img, f"data/produce_split/val/Grade_A/{cls}_{os.path.basename(img)}")
        for img in imgs[int(n*0.7):int(n*0.9)]:
            shutil.copy2(img, f"data/produce_split/train/Grade_B/{cls}_{os.path.basename(img)}")
            count["B"] += 1
        for img in imgs[int(n*0.9):]:
            shutil.copy2(img, f"data/produce_split/val/Grade_B/{cls}_{os.path.basename(img)}")
 
    for cls in diseased_classes:
        folder = os.path.join(source_path, cls)
        imgs   = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1] in IMG_EXTS]
        random.shuffle(imgs)
        n = min(len(imgs), 200)
        for img in imgs[:int(n*0.8)]:
            shutil.copy2(img, f"data/produce_split/train/Grade_C/{cls}_{os.path.basename(img)}")
            count["C"] += 1
        for img in imgs[int(n*0.8):]:
            shutil.copy2(img, f"data/produce_split/val/Grade_C/{cls}_{os.path.basename(img)}")
 
    print(f"  Train — A:{count['A']} B:{count['B']} C:{count['C']}")
 
 
# ── Step 3: Import TensorFlow ─────────────────────────────────
print("\nImporting TensorFlow...")
try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
 
    print(f"TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs: {len(gpus)} ({'GPU' if gpus else 'CPU — will take 25-45 min'})")
 
except ImportError:
    print("ERROR: TensorFlow not installed.")
    print("Fix: pip install tensorflow-cpu")
    sys.exit(1)
 
 
# ── Step 4: Data generators ───────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 15
 
train_gen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.75, 1.25],
    shear_range=0.1,
    fill_mode="nearest",
)
val_gen = ImageDataGenerator(rescale=1.0/255)
 
train_data = train_gen.flow_from_directory(
    "data/produce_split/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)
val_data = val_gen.flow_from_directory(
    "data/produce_split/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)
 
NUM_CLASSES = len(train_data.class_indices)
print(f"\nClasses: {train_data.class_indices}")
print(f"Train samples: {train_data.samples}")
print(f"Val samples:   {val_data.samples}")
 
 
# ── Step 5: Build EfficientNetB0 model ───────────────────────
print("\nBuilding EfficientNetB0 model...")
 
base = EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False   # Freeze pretrained weights
 
x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
 
model = Model(inputs=base.input, outputs=out)
 
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
 
trainable = sum(tf.keras.backend.count_params(p) for p in model.trainable_weights)
total     = model.count_params()
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,} (top layers only)")
 
 
# ── Step 6: Phase 1 — train top layers ───────────────────────
print(f"\nPhase 1: Training top layers ({EPOCHS} epochs)...")
 
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=4,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/produce_grader.h5",
                    monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                      patience=2, min_lr=1e-6, verbose=1),
]
 
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=callbacks,
    verbose=1,
)
 
best_val = max(history.history["val_accuracy"])
print(f"\nPhase 1 best val_accuracy: {best_val:.4f}")
 
 
# ── Step 7: Phase 2 — fine-tune top 20 base layers ───────────
if best_val < 0.85:
    print("\nPhase 2: Fine-tuning top 20 layers of EfficientNetB0...")
 
    for layer in base.layers[-20:]:
        layer.trainable = True
 
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
 
    history2 = model.fit(
        train_data,
        epochs=8,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=1,
    )
 
    # Merge histories
    for k in history.history:
        if k in history2.history:
            history.history[k].extend(history2.history[k])
 
    best_val = max(history.history.get("val_accuracy", [best_val]))
    print(f"Phase 2 best val_accuracy: {best_val:.4f}")
 
 
# ── Step 8: Training history chart ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
axes[0].plot(history.history["accuracy"],
             color="#1D9E75", label="Train accuracy", linewidth=2)
axes[0].plot(history.history["val_accuracy"],
             color="#7F77DD", label="Val accuracy", linewidth=2, linestyle="--")
axes[0].axhline(y=0.85, color="#D85A30", linewidth=0.8, linestyle=":",
                label="85% target")
axes[0].set_title("Accuracy per epoch", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.25)
 
axes[1].plot(history.history["loss"],
             color="#1D9E75", label="Train loss", linewidth=2)
axes[1].plot(history.history["val_loss"],
             color="#7F77DD", label="Val loss", linewidth=2, linestyle="--")
axes[1].set_title("Loss per epoch", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.25)
 
fig.suptitle(
    f"EfficientNetB0 Produce Grader — Best val_accuracy: {best_val*100:.2f}%",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig("assets/grader_training_history.png", dpi=200, bbox_inches="tight")
plt.close()
 
 
# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Produce Grader Training Complete!")
print("=" * 60)
print(f"  Model saved:      models/produce_grader.h5")
print(f"  Training history: assets/grader_training_history.png")
print(f"  Best val accuracy: {best_val*100:.2f}%")
print(f"  Classes: {train_data.class_indices}")
 
if best_val >= 0.85:
    print("\n  Target (≥85%) ACHIEVED!")
else:
    print(f"\n  Target not reached ({best_val*100:.1f}%). Try:")
    print("  1. More training data")
    print("  2. Increase EPOCHS to 20")
    print("  3. Use Google Colab T4 GPU for faster fine-tuning")
 
print("\nNext: Open pages/12_Produce_Grader.py in the Streamlit app")
print("      The page automatically loads models/produce_grader.h5")