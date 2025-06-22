from tensorflow import keras
from tensorflow.keras import layers

"""
    Builds and compiles a simple neural network for image classification.
    The model consists of:
    - Flatten layer: Converts 28x28 images to a 1D array
    - Dense layer: 128 neurons with ReLU activation
    - Output layer: 10 neurons (one for each class)
"""

def build_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model
