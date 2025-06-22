from data.data_loader import load_data
from model.model_builder import build_model
from train.train_model import train
from config import CLASS_NAMES

"""
    Main function to run the training pipeline.
    Função principal para executar o pipeline de treinamento.
"""

def main():

     # Load and normalize data / Carrega e normaliza os dados
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = build_model()

    # Train and evaluate the model / Treina e avalia o modelo
    train(model, train_images, train_labels, test_images, test_labels)

if __name__ == "__main__":
    main()
