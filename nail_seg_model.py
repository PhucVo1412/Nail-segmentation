"""
Shared model architecture + loss/metric definitions for the nail segmentation
model. Used by train.py, evaluate.py, and export_tflite.py so all three agree
on exactly the same network shape and custom objects (mirrors how nail_lib.py
is the single source of truth for crop/landmark logic, see training_plan.md).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_unet(input_shape=(128, 128, 3), base_filters=8):
    """Mini U-Net, 3 pooling levels. ~120K params at base_filters=8 -- sized
    for a few hundred training crops (see training_plan.md Option A)."""
    inputs = layers.Input(input_shape)

    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, base_filters * 8)

    u3 = layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(c4)
    u3 = layers.Concatenate()([u3, c3])
    d3 = conv_block(u3, base_filters * 4)

    u2 = layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, base_filters * 2)

    u1 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = conv_block(u1, base_filters)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return Model(inputs, outputs, name="mini_unet")


def dice_loss(y_true, y_pred, smooth=1e-6):
    axes = [1, 2, 3]
    inter = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    return 1 - tf.reduce_mean((2 * inter + smooth) / (union + smooth))


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


CUSTOM_OBJECTS = {"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss}


def per_image_iou(y_true, y_pred, threshold=0.5, eps=1e-6):
    """Per-image IoU for a batch, shape (N,H,W,1) -> (N,). Used by evaluate.py
    / export_tflite.py for the actual mIoU gate number (training_plan.md
    stage 4), as opposed to the pixel-aggregated BinaryIoU used during
    training for early stopping."""
    y_true_bin = (y_true > 0.5).astype("float32")
    y_pred_bin = (y_pred > threshold).astype("float32")
    axes = (1, 2, 3)
    inter = (y_true_bin * y_pred_bin).sum(axis=axes)
    union = y_true_bin.sum(axis=axes) + y_pred_bin.sum(axis=axes) - inter
    return (inter + eps) / (union + eps)
