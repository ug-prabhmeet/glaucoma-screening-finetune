#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation for GlaucomaDetector v3.

Usage:
    python evaluate.py --model-path ./checkpoints/model_p2_best.keras \\
                       --data-dir ./data/full-fundus \\
                       --save-dir ./results
"""

import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix,
    classification_report, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from config import CONFIG, RUN_NAME, TARGET_SENSITIVITY
from data import (build_eval_pipeline, collect_probs_and_labels,
                  init_label_mapping, load_dataset, remap_to_binary)
from model import load_trained_model


# ─── Threshold Selection ─────────────────────────────────────────


def select_threshold_for_sensitivity(y_true, y_prob, target_sensitivity=0.90):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    valid = np.where(tpr >= target_sensitivity)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmax(1 - fpr[valid])]
    else:
        best_idx = int(np.argmax(tpr))
    return float(thresholds[best_idx]), float(roc_auc), float(tpr[best_idx]), float(1 - fpr[best_idx])


# ─── Test Evaluation ──────────────────────────────────────────────


def evaluate_test(model, test_ds_raw, threshold, pos_name, neg_name, save_dir, run_name):
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

    print(f"\nTest CM:\n{cm}")
    print(f"Sensitivity: {sens:.4f}  Specificity: {spec:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}  PR AUC: {pr_auc:.4f}")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=[neg_name, pos_name], digits=4))

    # Save results
    with open(os.path.join(save_dir, f"{run_name}_test_results.json"), "w") as f:
        json.dump({"threshold": threshold, "confusion_matrix": cm.tolist(),
                    "sensitivity": sens, "specificity": spec,
                    "roc_auc": roc_auc, "pr_auc": pr_auc,
                    "positive_class": pos_name, "negative_class": neg_name}, f, indent=2)

    return y_true, y_prob


# ─── Training History Plot ────────────────────────────────────────


def plot_training_history(save_dir, run_name):
    import pandas as pd

    p1_log = os.path.join(save_dir, f"{run_name}_p1_log.csv")
    p2_log = os.path.join(save_dir, f"{run_name}_p2_log.csv")

    if not os.path.exists(p1_log) or not os.path.exists(p2_log):
        print("⚠ Training CSV logs not found, skipping history plot.")
        return

    df1 = pd.read_csv(p1_log)
    df2 = pd.read_csv(p2_log)
    history_df = pd.concat([df1, df2], ignore_index=True)

    metrics = [m for m in ["loss", "accuracy", "auc"] if m in history_df.columns]
    if not metrics:
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    phase1_epochs = len(df1)
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(history_df[metric], label="Train", linewidth=2)
        val_key = f"val_{metric}"
        if val_key in history_df.columns:
            ax.plot(history_df[val_key], label="Val", linewidth=2, linestyle="--")
        ax.axvline(x=phase1_epochs - 0.5, color="gray", linestyle=":")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"{run_name}_history.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ History plot saved to: {path}")


# ─── ROC & PR Curves ─────────────────────────────────────────────


def plot_curves(y_true, y_prob, threshold, save_dir, run_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    y_pred = (y_prob >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    op_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    op_tpr = tp / (tp + fn) if (tp + fn) else 0.0

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(fpr, tpr, linewidth=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="Random")
    ax1.scatter(op_fpr, op_tpr, marker="o", s=90, color="black",
                label=f"Threshold ({threshold:.2f})", zorder=5)
    ax1.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax1.legend(loc="lower right")

    baseline = np.mean(y_true)
    ax2.hlines(baseline, 0, 1, linestyles="--", linewidth=2, label=f"Baseline ({baseline:.3f})")
    ax2.plot(recall_vals, precision_vals, linewidth=2.5, label=f"PR (AUC = {pr_auc:.4f})")
    ax2.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision", title="PR Curve")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{run_name}_curves.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Curves saved to: {path}")

    with open(os.path.join(save_dir, f"{run_name}_curve_data.json"), "w") as f:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                    "precision": precision_vals.tolist(), "recall": recall_vals.tolist(),
                    "roc_auc": roc_auc, "pr_auc": pr_auc, "threshold": threshold}, f, indent=2)


# ─── Confusion Matrix Plot ───────────────────────────────────────


def plot_confusion_matrix(y_true, y_prob, threshold, pos_name, neg_name, save_dir, run_name):
    y_pred = (y_prob >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[neg_name, pos_name], yticklabels=[neg_name, pos_name],
                annot_kws={"size": 16, "weight": "bold"})
    plt.title(f"Confusion Matrix (Threshold: {threshold:.2f})", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("True Diagnosis", fontsize=12)
    plt.xlabel("AI Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{run_name}_confusion.png"), dpi=300, bbox_inches="tight")
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    roc_auc = roc_auc_score(y_true, y_prob)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    print("\n" + "=" * 60)
    print(" MEDICAL EVALUATION REPORT ".center(60))
    print("=" * 60)
    print(f"Threshold   : {threshold:.4f}")
    print(f"Sensitivity : {sens:.4f}  (% glaucoma caught)")
    print(f"Specificity : {spec:.4f}  (% normal cleared)")
    print(f"ROC AUC     : {roc_auc:.4f}")


# ─── Bootstrap Confidence Intervals ──────────────────────────────


def bootstrap_ci(y_true, y_prob, threshold, save_dir, run_name, n_bootstraps=2000):
    y_pred = (y_prob >= threshold).astype(np.int32)

    point = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    point["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0

    rng = np.random.default_rng(42)
    n = len(y_true)
    boot = {k: [] for k in point}

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        yhat = (yp >= threshold).astype(np.int32)
        if len(np.unique(yt)) < 2:
            continue
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
        boot["roc_auc"].append(roc_auc_score(yt, yp))
        boot["pr_auc"].append(average_precision_score(yt, yp))
        boot["accuracy"].append(accuracy_score(yt, yhat))
        boot["precision"].append(precision_score(yt, yhat, zero_division=0))
        boot["recall_sensitivity"].append(recall_score(yt, yhat, zero_division=0))
        boot["specificity"].append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) else 0.0)
        boot["f1"].append(f1_score(yt, yhat, zero_division=0))

    def _ci(arr, alpha=0.05):
        a = np.asarray(arr, dtype=np.float64)
        return {"mean": float(np.mean(a)), "lower": float(np.quantile(a, alpha / 2)),
                "upper": float(np.quantile(a, 1 - alpha / 2)), "n_boot": len(a)}

    ci_metrics = {k: _ci(v) for k, v in boot.items()}

    print("=" * 70)
    print(" BOOTSTRAP 95% CONFIDENCE INTERVALS ".center(70))
    print("=" * 70)
    for k in point:
        c = ci_metrics[k]
        print(f"{k:22s}: {point[k]:.4f}  (95% CI {c['lower']:.4f} – {c['upper']:.4f})")

    with open(os.path.join(save_dir, f"{run_name}_bootstrap_ci.json"), "w") as f:
        json.dump({"optimal_threshold": threshold, "point_metrics": {k: float(v) for k, v in point.items()},
                    "bootstrap_ci": ci_metrics, "n_bootstraps_requested": n_bootstraps}, f, indent=2)
    print(f"✓ Bootstrap CI saved")


# ─── Calibration ─────────────────────────────────────────────────


def plot_calibration(y_true, y_prob, save_dir, run_name):
    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2.5, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="Perfect")
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Observed Frequency", fontsize=12)
    plt.title(f"Calibration Curve\nBrier Score = {brier:.4f}", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{run_name}_calibration.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Brier Score: {brier:.6f}")

    with open(os.path.join(save_dir, f"{run_name}_calibration.json"), "w") as f:
        json.dump({"brier_score": brier, "prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}, f, indent=2)


# ─── Error Visualization ─────────────────────────────────────────


def plot_error_analysis(model, test_ds_raw, y_prob, threshold, save_dir, run_name):
    raw_vis_ds = test_ds_raw.map(remap_to_binary).batch(CONFIG["phase2_batch_size"])
    images_all, labels_all = [], []
    for bx, by in raw_vis_ds:
        images_all.append(bx.numpy())
        labels_all.append(by.numpy().reshape(-1))
    images_all = np.concatenate(images_all)
    labels_all = np.concatenate(labels_all)

    pred_labels = (y_prob >= threshold).astype(np.int32)
    fp_idx = np.where((labels_all == 0) & (pred_labels == 1))[0]
    fn_idx = np.where((labels_all == 1) & (pred_labels == 0))[0]
    print(f"False Positives: {len(fp_idx)}  |  False Negatives: {len(fn_idx)}")

    N = 6
    fig, axes = plt.subplots(2, N, figsize=(3 * N, 6))
    fig.suptitle("Model Error Analysis", fontsize=18, fontweight="bold")

    for i in range(N):
        ax = axes[0, i]
        if i < len(fp_idx):
            ax.imshow(images_all[fp_idx[i]].astype(np.uint8))
            ax.set_title(f"FP\nP={y_prob[fp_idx[i]]:.3f}", fontsize=10)
        ax.axis("off")

    for i in range(N):
        ax = axes[1, i]
        if i < len(fn_idx):
            ax.imshow(images_all[fn_idx[i]].astype(np.uint8))
            ax.set_title(f"FN\nP={y_prob[fn_idx[i]]:.3f}", fontsize=10)
        ax.axis("off")

    axes[0, 0].set_ylabel("False Positives", fontsize=12)
    axes[1, 0].set_ylabel("False Negatives", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{run_name}_error_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Error analysis saved")


# ─── Main ─────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Evaluate GlaucomaDetector v3")
    p.add_argument("--model-path", required=True, help="Path to .keras model checkpoint")
    p.add_argument("--data-dir", default="./data/full-fundus")
    p.add_argument("--save-dir", default="./results")
    p.add_argument("--run-name", default=None)
    p.add_argument("--target-sensitivity", type=float, default=TARGET_SENSITIVITY)
    args = p.parse_args()

    save_dir = args.save_dir
    run_name = args.run_name or RUN_NAME
    os.makedirs(save_dir, exist_ok=True)

    # GPU
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    np.random.seed(42)
    tf.random.set_seed(42)

    print(f"TF {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_trained_model(args.model_path)
    print("✓ Model loaded")

    # Load datasets
    val_ds_raw = load_dataset(os.path.join(args.data_dir, "val"))
    test_ds_raw = load_dataset(os.path.join(args.data_dir, "test"))
    pos_name, neg_name = init_label_mapping(list(val_ds_raw.class_names))
    print(f"Positive: {pos_name} → 1 | Negative: {neg_name} → 0")

    # Threshold selection
    print("\n" + "=" * 60 + "\n VALIDATION THRESHOLD SELECTION\n" + "=" * 60)
    val_ds = build_eval_pipeline(val_ds_raw, CONFIG["phase2_batch_size"])
    y_val_true, y_val_prob = collect_probs_and_labels(model, val_ds)
    threshold, val_auc, val_sens, val_spec = select_threshold_for_sensitivity(
        y_val_true, y_val_prob, args.target_sensitivity)
    print(f"Threshold: {threshold:.4f} | AUC: {val_auc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f}")

    with open(os.path.join(save_dir, f"{run_name}_threshold.json"), "w") as f:
        json.dump({"optimal_threshold": threshold, "validation_auc": val_auc,
                    "validation_sensitivity": val_sens, "validation_specificity": val_spec,
                    "selection_method": f"target_sensitivity_{args.target_sensitivity:.2f}"}, f, indent=2)

    # Test evaluation
    y_true, y_prob = evaluate_test(model, test_ds_raw, threshold, pos_name, neg_name, save_dir, run_name)
    y_true = np.asarray(y_true).reshape(-1).astype(np.int32)
    y_prob = np.asarray(y_prob).reshape(-1).astype(np.float32)

    # Plots & analysis
    plot_training_history(save_dir, run_name)
    plot_curves(y_true, y_prob, threshold, save_dir, run_name)
    plot_confusion_matrix(y_true, y_prob, threshold, pos_name, neg_name, save_dir, run_name)
    bootstrap_ci(y_true, y_prob, threshold, save_dir, run_name)
    plot_calibration(y_true, y_prob, save_dir, run_name)
    plot_error_analysis(model, test_ds_raw, y_prob, threshold, save_dir, run_name)

    print("\n✓ Full evaluation pipeline complete!")


if __name__ == "__main__":
    main()
