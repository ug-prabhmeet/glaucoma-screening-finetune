"""
config.py — Central configuration for Glaucoma Screening.

All hyperparameters, paths, and constants live here so that
train.py and evaluate.py import from a single source of truth.
"""

import os

# ─── Hyperparameters ──────────────────────────────────────────────

CONFIG = {
    # Model architecture
    "input_size": (224, 224, 3),
    "backbone": "efficientnet-b0",

    # Phase 1: Frozen backbone (feature‑extraction)
    "phase1_batch_size": 64,
    "phase1_lr": 3e-4,
    "phase1_epochs": 12,

    # Phase 2: Fine‑tuning (unfreeze late blocks)
    "phase2_batch_size": 32,
    "phase2_lr": 1e-5,
    "phase2_epochs": 25,

    # Optimizer
    "weight_decay": 1e-5,

    # Focal Loss
    "focal_alpha": 0.65,
    "focal_gamma": 2.0,
    "label_smoothing": 0.05,

    # CBAM attention
    "cbam_reduction": 16,
}

# ─── Paths (override via CLI args in train.py / evaluate.py) ─────

DATA_DIR = os.environ.get(
    "GLAUCOMA_DATA_DIR",
    "./data/full-fundus",
)

SAVE_DIR = os.environ.get(
    "GLAUCOMA_SAVE_DIR",
    "./checkpoints",
)

# ─── Run Identification ──────────────────────────────────────────

RUN_ID = "9_may_glaucoma"
RUN_NAME = f"glaucoma_b0_{RUN_ID}"

# ─── Class Weights (optional) ────────────────────────────────────

USE_CLASS_WEIGHTS = False

CLASS_WEIGHT = {
    0: 1.0,   # normal
    1: 1.25,  # glaucoma
}

# ─── Label Aliases ────────────────────────────────────────────────

POSITIVE_CLASS_ALIASES = {"glaucoma", "positive", "abnormal"}
NEGATIVE_CLASS_ALIASES = {"normal", "negative"}

# ─── Threshold Selection ─────────────────────────────────────────

TARGET_SENSITIVITY = 0.90  # default for targeted‑sensitivity method
