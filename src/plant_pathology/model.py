"""Keras VGG16 transfer-learning model — used by the training notebook."""
from __future__ import annotations

from .config import CLASS_NAMES, IMG_SIZE


def build_vgg16(n_classes: int = len(CLASS_NAMES), img_size: int = IMG_SIZE,
                freeze_backbone: bool = True):
    """VGG16 backbone (ImageNet weights) → GAP → Dense head."""
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required to build VGG16. Run the training notebook "
            "in Colab where TF is pre-installed."
        ) from e

    base = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = not freeze_backbone

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax",
                                     name="class_proba")(x)

    model = tf.keras.Model(inputs, outputs, name="vgg16_plant_pathology")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
