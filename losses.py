"""
losses.py — Custom loss functions for glaucoma screening.

Implements Focal Loss with optional label smoothing to handle
the class imbalance typical in medical screening datasets.
"""

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """
    Binary focal loss with optional label smoothing.

    α  weights the positive class (default 0.65 for glaucoma).
    γ  down‑weights easy examples (default 2.0).
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        label_smoothing=0.0,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name="focal_loss",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Optional label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # p_t = p if y=1 else (1−p)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # Alpha weighting
        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)

        # Focal modulation
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        # Per‑sample loss
        loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.squeeze(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing,
        })
        return config
