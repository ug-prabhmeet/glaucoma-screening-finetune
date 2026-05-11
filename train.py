#!/usr/bin/env python3
"""
train.py — Two-phase training pipeline for GlaucomaDetector v3.

Usage:  python train.py --data-dir ./data/full-fundus --save-dir ./checkpoints
"""

import argparse, json, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW

from config import CLASS_WEIGHT, CONFIG, RUN_NAME, TARGET_SENSITIVITY, USE_CLASS_WEIGHTS
from data import (build_eval_pipeline, build_train_pipeline, collect_probs_and_labels,
                  init_label_mapping, load_dataset)
from losses import FocalLoss
from model import build_glaucoma_model, load_trained_model


def _setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def _set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def iter_all_layers(layer_or_model, seen=None):
    if seen is None:
        seen = set()
    if id(layer_or_model) in seen:
        return
    seen.add(id(layer_or_model))
    yield layer_or_model
    if hasattr(layer_or_model, "layers"):
        for sub in layer_or_model.layers:
            yield from iter_all_layers(sub, seen)


def select_threshold_for_sensitivity(y_true, y_prob, target_sensitivity=0.90):
    from sklearn.metrics import auc, roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    valid = np.where(tpr >= target_sensitivity)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmax(1 - fpr[valid])]
    else:
        best_idx = int(np.argmax(tpr))
    return float(thresholds[best_idx]), float(roc_auc), float(tpr[best_idx]), float(1 - fpr[best_idx])


def _compile_model(model, lr_schedule, wd_factor=1.0):
    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=CONFIG["weight_decay"] * wd_factor, clipnorm=1.0),
        loss=FocalLoss(alpha=CONFIG["focal_alpha"], gamma=CONFIG["focal_gamma"], label_smoothing=CONFIG["label_smoothing"]),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def run_phase1(model, train_ds_raw, val_ds_raw, save_dir, run_name):
    train_ds = build_train_pipeline(train_ds_raw, CONFIG["phase1_batch_size"])
    val_ds = build_eval_pipeline(val_ds_raw, CONFIG["phase1_batch_size"])
    steps = max(tf.data.experimental.cardinality(train_ds).numpy(), 100)
    lr = tf.keras.optimizers.schedules.CosineDecay(CONFIG["phase1_lr"], steps * CONFIG["phase1_epochs"])
    _compile_model(model, lr)

    best_path = os.path.join(save_dir, f"{run_name}_p1_best.keras")
    callbacks = [
        ModelCheckpoint(best_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(save_dir, f"{run_name}_p1_log.csv")),
    ]
    print("=" * 60 + "\n PHASE 1: Frozen Backbone\n" + "=" * 60)
    history = model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["phase1_epochs"],
                        callbacks=callbacks, verbose=1, class_weight=CLASS_WEIGHT if USE_CLASS_WEIGHTS else None)
    model.save(os.path.join(save_dir, f"{run_name}_p1_resume.keras"))
    print(f"\n✓ Phase 1 complete!")
    return history


def prepare_phase2(model, save_dir, run_name):
    best = os.path.join(save_dir, f"{run_name}_p1_best.keras")
    resume = os.path.join(save_dir, f"{run_name}_p1_resume.keras")
    model = load_trained_model(best if os.path.exists(best) else resume)

    for layer in iter_all_layers(model):
        if layer is not model:
            layer.trainable = False

    for layer in iter_all_layers(model):
        lname = layer.name.lower()
        if any(k in lname for k in ("block6", "block7", "top_conv", "top_bn", "top_activation")):
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        if any(k in lname for k in ("cbam", "gap", "dense", "dropout", "output")):
            layer.trainable = True

    for layer in iter_all_layers(model):
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.trainable = True
    if not model.trainable_weights:
        raise RuntimeError("No trainable weights found.")
    return model


def run_phase2(model, train_ds_raw, val_ds_raw, save_dir, run_name):
    train_ds = build_train_pipeline(train_ds_raw, CONFIG["phase2_batch_size"])
    val_ds = build_eval_pipeline(val_ds_raw, CONFIG["phase2_batch_size"])
    steps = max(tf.data.experimental.cardinality(train_ds).numpy(), 100)
    lr = tf.keras.optimizers.schedules.CosineDecay(CONFIG["phase2_lr"], steps * CONFIG["phase2_epochs"])
    _compile_model(model, lr, wd_factor=0.5)

    best_path = os.path.join(save_dir, f"{run_name}_p2_best.keras")
    callbacks = [
        ModelCheckpoint(best_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_auc", mode="max", patience=7, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(save_dir, f"{run_name}_p2_log.csv")),
    ]
    print("=" * 60 + "\n PHASE 2: Fine-tuning\n" + "=" * 60)
    history = model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["phase2_epochs"],
                        callbacks=callbacks, verbose=1, class_weight=CLASS_WEIGHT if USE_CLASS_WEIGHTS else None)
    print("\n✓ Phase 2 complete!")
    return history


def quick_test_eval(model, test_ds_raw, threshold, pos_name, neg_name, save_dir, run_name):
    from sklearn.metrics import auc, average_precision_score, classification_report, confusion_matrix, roc_curve

    test_ds = build_eval_pipeline(test_ds_raw, CONFIG["phase2_batch_size"])
    y_true, y_prob = collect_probs_and_labels(model, test_ds)
    y_pred = (y_prob >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_prob)

    print(f"\nTest CM:\n{cm}\nSens: {sens:.4f}  Spec: {spec:.4f}  ROC AUC: {roc_auc:.4f}  PR AUC: {pr_auc:.4f}")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=[neg_name, pos_name], digits=4))

    final_path = os.path.join(save_dir, f"{run_name}_final.keras")
    weights_path = os.path.join(save_dir, f"{run_name}_weights.weights.h5")
    model.save(final_path)
    model.save_weights(weights_path)

    with open(os.path.join(save_dir, f"{run_name}_test_results.json"), "w") as f:
        json.dump({"threshold": threshold, "confusion_matrix": cm.tolist(),
                    "sensitivity": sens, "specificity": spec, "roc_auc": roc_auc, "pr_auc": pr_auc,
                    "positive_class": pos_name, "negative_class": neg_name}, f, indent=2)
    print(f"✓ Saved model, weights, and results to {save_dir}")


def main():
    p = argparse.ArgumentParser(description="Train GlaucomaDetector v3")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    data_dir = args.data_dir or "./data/full-fundus"
    save_dir = args.save_dir or "./checkpoints"
    run_name = f"glaucoma_b0_{args.run_id}" if args.run_id else RUN_NAME
    os.makedirs(save_dir, exist_ok=True)

    _setup_gpu()
    _set_seeds()
    print(f"TF {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")

    train_ds_raw = load_dataset(os.path.join(data_dir, "train"), shuffle=True)
    val_ds_raw = load_dataset(os.path.join(data_dir, "val"), shuffle=False)
    test_ds_raw = load_dataset(os.path.join(data_dir, "test"), shuffle=False)

    pos_name, neg_name = init_label_mapping(list(train_ds_raw.class_names))
    print(f"Positive: {pos_name} → 1 | Negative: {neg_name} → 0")

    model = build_glaucoma_model(CONFIG)
    run_phase1(model, train_ds_raw, val_ds_raw, save_dir, run_name)

    model = prepare_phase2(model, save_dir, run_name)
    run_phase2(model, train_ds_raw, val_ds_raw, save_dir, run_name)

    model = load_trained_model(os.path.join(save_dir, f"{run_name}_p2_best.keras"))

    # Threshold selection
    val_ds = build_eval_pipeline(val_ds_raw, CONFIG["phase2_batch_size"])
    y_val_true, y_val_prob = collect_probs_and_labels(model, val_ds)
    threshold, val_auc, val_sens, val_spec = select_threshold_for_sensitivity(y_val_true, y_val_prob, TARGET_SENSITIVITY)
    print(f"\nThreshold: {threshold:.4f} | Val AUC: {val_auc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f}")

    with open(os.path.join(save_dir, f"{run_name}_threshold.json"), "w") as f:
        json.dump({"optimal_threshold": threshold, "validation_auc": val_auc,
                    "validation_sensitivity": val_sens, "validation_specificity": val_spec,
                    "selection_method": f"target_sensitivity_{TARGET_SENSITIVITY:.2f}"}, f, indent=2)

    quick_test_eval(model, test_ds_raw, threshold, pos_name, neg_name, save_dir, run_name)
    print("\n✓ Training pipeline complete!")


if __name__ == "__main__":
    main()
