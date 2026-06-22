"""
AgriSense India — Plant Disease Detector CNN (Days 11–12)
File: notebooks/08_disease_cnn.py

Builds and trains a MobileNetV2 transfer learning model
on the PlantVillage dataset (38 disease classes, 14 crops).

Architecture:
  MobileNetV2 (pretrained ImageNet) → frozen base
  → GlobalAveragePooling2D
  → Dense(128, relu) + Dropout(0.3)
  → Dense(38, softmax)

Target: ≥85% validation accuracy after 10 epochs.

COLAB TIP: If CPU training is too slow, use Google Colab:
  1. Upload this script to Colab
  2. Upload data/plantvillage/ to Google Drive
  3. Mount Drive, set PLANTVILLAGE_PATH to Drive path
  4. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
  5. Download disease_model.h5 after training

Run: python notebooks/08_disease_cnn.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)

print("=" * 60)
print("AgriSense India — Plant Disease CNN (Day 11)")
print("=" * 60)

# ── Config ───────────────────────────────────────────────────
IMG_SIZE   = 224        # MobileNetV2 input size
BATCH_SIZE = 32         # reduce to 16 if OOM error
EPOCHS     = 10
LEARNING_RATE = 0.001

# ── Find PlantVillage split ──────────────────────────────────
PV_CANDIDATES = [
    "data/plantvillage",
    "data/raw/plantvillage_split",
]
PV_BASE = None
for c in PV_CANDIDATES:
    if os.path.exists(os.path.join(c, "train")):
        PV_BASE = c
        break

if PV_BASE is None:
    print("\nERROR: PlantVillage split not found.")
    print("Run first: python notebooks/04_organize_plantvillage.py")
    print("\nExpected structure:")
    print("  data/plantvillage/train/  (38 class folders)")
    print("  data/plantvillage/val/    (38 class folders)")
    print("  data/plantvillage/test/   (38 class folders)")
    sys.exit(1)

TRAIN_DIR = os.path.join(PV_BASE, "train")
VAL_DIR   = os.path.join(PV_BASE, "val")
TEST_DIR  = os.path.join(PV_BASE, "test")

# Count classes
classes = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])
NUM_CLASSES = len(classes)
print(f"\nPlantVillage path : {PV_BASE}")
print(f"Classes found     : {NUM_CLASSES}")
print(f"Sample classes    : {classes[:5]}")

# Count images
train_count = sum(
    len(os.listdir(os.path.join(TRAIN_DIR, c)))
    for c in classes if os.path.isdir(os.path.join(TRAIN_DIR, c))
)
val_count = sum(
    len(os.listdir(os.path.join(VAL_DIR, c)))
    for c in classes if os.path.isdir(os.path.join(VAL_DIR, c))
)
print(f"Train images      : {train_count:,}")
print(f"Val images        : {val_count:,}")


# ── Import TensorFlow ────────────────────────────────────────
print("\nImporting TensorFlow...")
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import (
        GlobalAveragePooling2D, Dense, Dropout, Input
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available    : {len(gpus)} ({'GPU training' if gpus else 'CPU training — will be slow'})")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
except ImportError:
    print("ERROR: TensorFlow not installed.")
    print("Fix: pip install tensorflow==2.13.0")
    print("     or on Windows: pip install tensorflow-cpu")
    sys.exit(1)


# ── Data generators ──────────────────────────────────────────
print("\nCreating data generators...")

train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest",
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)
val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# Save class names
class_names = list(train_data.class_indices.keys())
joblib.dump(class_names, "models/class_names.pkl")
print(f"Classes saved     : {len(class_names)}")


# ── Build MobileNetV2 model ──────────────────────────────────
print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False   # Freeze pretrained weights

# Custom head
inputs  = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = GlobalAveragePooling2D()(x)
x       = Dense(128, activation="relu")(x)
x       = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

total_params     = model.count_params()
trainable_params = sum(
    tf.keras.backend.count_params(p) for p in model.trainable_weights
)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print(f"Frozen parameters    : {total_params - trainable_params:,}")


# ── Callbacks ────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        "models/disease_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]


# ── Train ─────────────────────────────────────────────────────
print(f"\nTraining for up to {EPOCHS} epochs...")
print("(EarlyStopping will halt if val_accuracy stops improving)")
print("─" * 50)

steps_per_epoch  = max(1, train_count // BATCH_SIZE)
validation_steps = max(1, val_count   // BATCH_SIZE)

history = model.fit(
    train_data,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
)

print("\nTraining complete!")
best_val_acc = max(history.history["val_accuracy"])
best_epoch   = history.history["val_accuracy"].index(best_val_acc) + 1
print(f"Best val_accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at epoch {best_epoch}")


# ── Training history chart ────────────────────────────────────
print("\nGenerating training history chart...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["accuracy"],     color="#1D9E75", label="Train accuracy", linewidth=2)
axes[0].plot(history.history["val_accuracy"], color="#7F77DD", label="Val accuracy",   linewidth=2, linestyle="--")
axes[0].set_title("Model accuracy per epoch", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(alpha=0.25)
axes[0].axhline(y=0.85, color="#D85A30", linewidth=0.8, linestyle=":", label="85% target")

axes[1].plot(history.history["loss"],     color="#1D9E75", label="Train loss", linewidth=2)
axes[1].plot(history.history["val_loss"], color="#7F77DD", label="Val loss",   linewidth=2, linestyle="--")
axes[1].set_title("Model loss per epoch", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(alpha=0.25)

fig.suptitle(f"MobileNetV2 — PlantVillage Disease Detector\nBest val accuracy: {best_val_acc*100:.2f}%",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/cnn_training_history.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: assets/cnn_training_history.png")


# ── Test evaluation ───────────────────────────────────────────
print("\nEvaluating on test set...")
test_data = val_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)
test_loss, test_acc = model.evaluate(test_data, verbose=0)
print(f"Test accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test loss      : {test_loss:.4f}")


# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Day 11 complete!")
print("=" * 60)
print(f"  disease_model.h5 saved   : {'YES' if os.path.exists('models/disease_model.h5') else 'NO'}")
print(f"  class_names.pkl saved    : {'YES' if os.path.exists('models/class_names.pkl') else 'NO'}")
print(f"  training_history.png     : {'YES' if os.path.exists('assets/cnn_training_history.png') else 'NO'}")
print(f"  Best val accuracy        : {best_val_acc*100:.2f}%")
if best_val_acc >= 0.85:
    print("  Target (≥85%) ACHIEVED")
else:
    print("  Target not reached — try fine-tuning (see Day 12 script)")
print("\nNext: python notebooks/09_disease_info.py")
