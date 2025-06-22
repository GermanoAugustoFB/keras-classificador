from data.data_loader import load_data
from model.model_builder import build_model
from train.train_model import train
from config import CLASS_NAMES

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = build_model()
    train(model, train_images, train_labels, test_images, test_labels)

if __name__ == "__main__":
    main()
