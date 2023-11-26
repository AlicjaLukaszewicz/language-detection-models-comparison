import numpy as np

from data.global_variables import LANGUAGE_LABELS
from data.load_data import load_data
from statistical_approach.src.models.build_model import build_model


def train_model(x_train_file: str, y_train_file: str, n_vals: list[int]):
    """
    Builds a language model for each language in the training data.

    :param n_vals: The n values to use for the language models.
    :param x_train_file: The path to the file containing the training sentences.
    :param y_train_file: The path to the file containing the corresponding labels.
    :return: A dictionary containing the language models.
    """
    sentences_to_languages = load_data(x_train_file, y_train_file)

    models = {}
    for language in LANGUAGE_LABELS:
        # Filter the sentences that correspond to the current language
        language_sentences = np.array([sent for sent, lang in sentences_to_languages.items() if lang == language])
        # Build a language model from the filtered sentences
        models[language] = build_model(text=' '.join(language_sentences), n_vals=n_vals)
    return models
