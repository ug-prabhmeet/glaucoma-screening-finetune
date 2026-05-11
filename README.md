# Fine-Tuning for Real-Time Glaucoma Screening

EfficientNetB0 + CBAM attention for binary glaucoma detection from full-fundus images. Two-phase training with focal loss and targeted-sensitivity threshold selection.

## Architecture

```
EfficientNetB0 (ImageNet) → CBAM → GAP → Dense(512) → Dense(256) + Residual → Dense(128) → Sigmoid
```

- **Backbone**: EfficientNetB0 (frozen in Phase 1, block6/7 unfrozen in Phase 2)
- **Attention**: CBAM (Channel + Spatial) with reduction ratio 16
- **Loss**: Focal Loss (α=0.65, γ=2.0, label smoothing=0.05)
- **Optimizer**: AdamW with cosine decay LR schedule
- **Threshold**: Selected on validation set targeting ≥90% sensitivity

## Dataset Structure

```
data/full-fundus/
├── train/
│   ├── glaucoma/
│   └── normal/
├── val/
│   ├── glaucoma/
│   └── normal/
└── test/
    ├── glaucoma/
    └── normal/
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data-dir ./data/full-fundus --save-dir ./checkpoints --run-id my_run
```

This runs:
1. **Phase 1** — Frozen backbone, lr=3e-4, 12 epochs
2. **Phase 2** — Fine-tune block6/7 + head, lr=1e-5, 25 epochs
3. Validation threshold selection (≥90% sensitivity)
4. Quick test evaluation

### Evaluation

```bash
python evaluate.py \
    --model-path ./checkpoints/glaucoma_b0_my_run_p2_best.keras \
    --data-dir ./data/full-fundus \
    --save-dir ./results
```

This generates:
- ROC & Precision-Recall curves
- Confusion matrix heatmap
- Bootstrap 95% confidence intervals (2000 iterations)
- Calibration curve + Brier score
- False positive / false negative visualization

## Project Structure

| File | Description |
|---|---|
| `config.py` | Hyperparameters, paths, constants |
| `data.py` | Dataset loading, augmentation, tf.data pipelines |
| `cbam.py` | CBAM attention layers (Channel + Spatial) |
| `losses.py` | Focal Loss with label smoothing |
| `model.py` | Model architecture + loading utilities |
| `train.py` | Two-phase training pipeline (CLI) |
| `evaluate.py` | Full evaluation pipeline (CLI) |

## Results

| Metric | Value |
|---|---|
| ROC AUC | See `*_test_results.json` |
| Sensitivity | Tuned ≥ 0.90 on validation |
| Brier Score | See `*_calibration.json` |

## License

BTP Final Year Project — IIT Delhi
