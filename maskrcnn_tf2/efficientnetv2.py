import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfl
import keras


efficientnet_v2_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    include_preprocessing=True
)

print(efficientnet_v2_model.trainable)

