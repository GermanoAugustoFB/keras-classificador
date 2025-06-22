from config import EPOCHS
import matplotlib.pyplot as plt

def train(model, train_images, train_labels, test_images, test_labels):
    history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_split=0.1)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nAcurácia no teste: {test_acc:.4f}')

    # (Opcional) Mostrar gráfico de treinamento
    plot_history(history)

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Treino')
    plt.plot(epochs_range, val_acc, label='Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino')
    plt.plot(epochs_range, val_loss, label='Validação')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.tight_layout()
    plt.show()
