from statistical_approach.src.models.detect_language import identify_language
from statistical_approach.src.models.train_model import train_model

if __name__ == '__main__':
    text = "Gazowana woda jest zdecydowanie lepsza od zwykłej.".lower()
    models = train_model('../../data/raw/x_train.txt', '../../data/raw/y_train.txt', n_vals=range(1, 4))

    print(f"Test text: {text}")
    print(f"Identified language: {identify_language(text, models, n_vals=range(1, 4))}")
