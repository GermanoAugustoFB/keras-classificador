# Image Classifier (CNN) - Fashion MNIST

## 🇬🇧 English

This project is a simple image classifier using TensorFlow and Keras.  
It uses the Fashion MNIST dataset, which contains grayscale images of clothing items.  
The model is a basic neural network (not a convolutional one) for educational purposes.

### Features

- Loads and normalizes Fashion MNIST data
- Builds and trains a neural network model
- Evaluates accuracy on test data
- Visualizes training progress

### How to Run

1. **Install dependencies**  
   Activate your virtual environment and run:

 ```shell  
   pip install -r requirements.txt
   python main.py
 ```
 
2. **Train and evaluate the model**  

    python main.py

### File Structure

- `data/data_loader.py` – Loads and normalizes the dataset
- `model/model_builder.py` – Builds the neural network model
- `train/train_model.py` – Trains and evaluates the model
- `utils/plot_utils.py` – (Optional) Functions for plotting images
- `config.py` – Configuration (class names, epochs, etc.)
- `main.py` – Main script to run everything

### Output

After training, the project displays a graph showing the accuracy and loss curves for both training and validation sets.  
This helps you visualize how well the model is learning and if it is overfitting.

---

## 🇧🇷 Português

Este projeto é um classificador de imagens simples usando TensorFlow e Keras.  
Utiliza o dataset Fashion MNIST, que contém imagens em tons de cinza de peças de roupa.  
O modelo é uma rede neural básica (não convolucional), ideal para fins didáticos.

### Funcionalidades

- Carrega e normaliza os dados do Fashion MNIST
- Constrói e treina um modelo de rede neural
- Avalia a acurácia nos dados de teste
- Visualiza o progresso do treinamento

### Como executar

1. **Instale as dependências**  
Ative seu ambiente virtual e rode: 

 ```shell  
   pip install -r requirements.txt
   python main.py
 ``` 

2. **Treine e avalie o modelo**  

    python main.py

### Estrutura dos arquivos

- `data/data_loader.py` – Carrega e normaliza o dataset
- `model/model_builder.py` – Constrói o modelo de rede neural
- `train/train_model.py` – Treina e avalia o modelo
- `utils/plot_utils.py` – (Opcional) Funções para plotar imagens
- `config.py` – Configurações (nomes das classes, épocas, etc.)
- `main.py` – Script principal para rodar tudo

### Saída

Após o treinamento, o projeto exibe um gráfico mostrando as curvas de acurácia e perda (loss) para os conjuntos de treino e validação.  
Isso ajuda a visualizar o desempenho do modelo e identificar possíveis sinais de overfitting.

---

## License

MIT License
