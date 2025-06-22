import matplotlib.pyplot as plt
import numpy as np
from config import CLASS_NAMES

"""
    Plots training and validation accuracy and loss.
    Plota a acurácia e a perda de treino e validação.
"""

def plot_sample_images(images, labels, num=25):
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(CLASS_NAMES[labels[i]])
    plt.tight_layout()
    plt.show()
