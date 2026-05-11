"""
model.py — GlaucomaDetector v3 architecture and loading utilities.

Architecture:
    EfficientNetB0 (ImageNet) → CBAM → GAP → Dense head with residual → sigmoid
"""

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

from cbam import CBAM, ChannelAttention, SpatialAttention
from losses import FocalLoss


def build_glaucoma_model(config: dict) -> Model:
    """
    Build the GlaucomaDetector_v3 model.

    Parameters
    ----------
    config : dict
        Must contain ``input_size`` and ``cbam_reduction``.

    Returns
    -------
    keras.Model with a single sigmoid output.
    """
    inputs = layers.Input(shape=config["input_size"], name="input_image")

    # ── Backbone ──
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling=None,
    )
    backbone.trainable = False

    x = backbone.output  # (None, 7, 7, 1280)

    # ── CBAM ──
    x = CBAM(
        reduction_ratio=config["cbam_reduction"],
        kernel_size=7,
        name="cbam",
    )(x)

    # ── Classification Head ──
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Block 1
    x1 = layers.Dense(512, kernel_initializer="he_normal")(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation("swish")(x1)
    x1 = layers.Dropout(0.4)(x1)

    # Block 2
    x2 = layers.Dense(256, kernel_initializer="he_normal")(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation("swish")(x2)
    x2 = layers.Dropout(0.3)(x2)

    # Residual connection (project GAP output to 256‑d)
    res = layers.Dense(256, kernel_initializer="he_normal")(x)
    x = layers.Add()([x2, res])

    # Block 3
    x = layers.Dense(128, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.2)(x)

    # Output (float32 even under mixed precision)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="output",
    )(x)

    return Model(inputs, outputs, name="GlaucomaDetector_v3")


# ─── Custom Objects for Model Loading ─────────────────────────────


def get_custom_objects() -> dict:
    """Return the dict needed by ``keras.models.load_model(custom_objects=...)``."""
    return {
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "FocalLoss": FocalLoss,
    }


def load_trained_model(path: str, compile: bool = False):
    """Load a saved ``.keras`` checkpoint with all custom objects registered."""
    from tensorflow import keras

    return keras.models.load_model(
        path,
        custom_objects=get_custom_objects(),
        compile=compile,
    )
