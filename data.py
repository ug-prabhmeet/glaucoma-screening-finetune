"""
data.py — Dataset loading, label resolution, augmentation, and tf.data pipelines.

Handles the full‑fundus folder layout:
    data_dir/
        train/  glaucoma/  normal/
        val/    glaucoma/  normal/
        test/   glaucoma/  normal/
"""

import tensorflow as tf
from tensorflow.keras import layers

from config import CONFIG, POSITIVE_CLASS_ALIASES, NEGATIVE_CLASS_ALIASES

AUTOTUNE = tf.data.AUTOTUNE

# ─── Label Resolution ────────────────────────────────────────────


def _normalize_class_name(name: str) -> str:
    """Lower‑case, strip, and replace spaces/hyphens with underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_binary_class_indices(class_names: list[str]):
    """
    Return (positive_idx, negative_idx) from the folder names
    discovered by ``image_dataset_from_directory``.
    """
    normalized = [_normalize_class_name(c) for c in class_names]

    pos_hits = [i for i, c in enumerate(normalized) if c in POSITIVE_CLASS_ALIASES]
    neg_hits = [i for i, c in enumerate(normalized) if c in NEGATIVE_CLASS_ALIASES]

    if len(pos_hits) != 1:
        raise ValueError(
            f"Could not resolve exactly one positive class from: {class_names}"
        )
    if len(neg_hits) != 1:
        raise ValueError(
            f"Could not resolve exactly one negative class from: {class_names}"
        )

    return pos_hits[0], neg_hits[0]


# ─── Dataset Loading ─────────────────────────────────────────────


def load_dataset(directory: str, shuffle: bool = False):
    """
    Load an image dataset from ``directory`` using Keras.

    Returns an *unbatched* ``tf.data.Dataset`` with ``(image, int_label)`` pairs
    and attaches ``.class_names`` for downstream label resolution.
    """
    img_size = CONFIG["input_size"][:2]  # (224, 224)

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=None,
        color_mode="rgb",
        shuffle=shuffle,
        seed=42 if shuffle else None,
    )
    return ds


# ─── Binary Remapping ────────────────────────────────────────────

# Module‑level cache filled by ``init_label_mapping()``.
_positive_class_idx: int | None = None


def init_label_mapping(class_names: list[str]):
    """
    Must be called once after loading any dataset so that
    ``remap_to_binary`` knows which folder index is positive.

    Returns ``(positive_class_name, negative_class_name)``.
    """
    global _positive_class_idx
    pos_idx, neg_idx = resolve_binary_class_indices(class_names)
    _positive_class_idx = pos_idx
    return class_names[pos_idx], class_names[neg_idx]


def remap_to_binary(image, label):
    """Map dataset labels so that positive → 1, negative → 0."""
    label = tf.cast(label, tf.int32)
    label = tf.cast(tf.equal(label, _positive_class_idx), tf.float32)
    return image, tf.expand_dims(label, axis=-1)


# ─── Augmentation Pipeline ───────────────────────────────────────

_rotation_factor = 10 / 360.0

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=_rotation_factor),
        layers.RandomBrightness(factor=0.08),
        layers.RandomContrast(factor=0.15),
    ],
    name="augmentation_pipeline",
)


def _process_train(image, label):
    image = data_augmentation(image, training=True)
    return image, label


# ─── Pipeline Builders ───────────────────────────────────────────


def build_train_pipeline(ds_raw, batch_size: int):
    """Remap → shuffle → batch → augment → prefetch."""
    return (
        ds_raw
        .map(remap_to_binary, num_parallel_calls=AUTOTUNE)
        .shuffle(2000, seed=42, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .map(_process_train, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )


def build_eval_pipeline(ds_raw, batch_size: int):
    """Remap → batch → prefetch (no augmentation)."""
    return (
        ds_raw
        .map(remap_to_binary, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )


# ─── Inference Helper ────────────────────────────────────────────


def collect_probs_and_labels(model, ds):
    """
    Run inference on a batched dataset and return
    ``(y_true, y_prob)`` as flat numpy arrays.
    """
    import numpy as np

    y_true, y_prob = [], []
    for x_batch, y_batch in ds:
        probs = model(x_batch, training=False)
        y_true.append(y_batch.numpy().reshape(-1))
        y_prob.append(probs.numpy().reshape(-1))

    return np.concatenate(y_true), np.concatenate(y_prob)
