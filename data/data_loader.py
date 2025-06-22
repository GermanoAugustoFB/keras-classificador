from tensorflow.keras.datasets import fashion_mnist
import numpy as np

def load_data():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normaliza os dados (valores de pixel entre 0 e 1)
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    return (train_images, train_labels), (test_images, test_labels)
