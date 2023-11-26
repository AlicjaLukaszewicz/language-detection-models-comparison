from collections import Counter
from typing import List, Dict


def build_model(text: str, n_vals: List[int]) -> Dict[str, int]:
    """
    Build a language model from a given text.

    :param text: the text from which to extract utils
    :param n_vals: the sizes of n-grams to extract
    :return: a dictionary with n-grams as keys and their probabilities as values
    """
    model = Counter(extract_ngrams(text, n_vals))
    total = sum(model.values())
    for key in model:
        model[key] /= total
    return model


def extract_ngrams(text: str, n_vals: List[int]) -> List[str]:
    """
    Extract n-grams from a given text.

    :param text: the text from which to extract n-grams
    :param n_vals: the sizes of n-grams to extract
    :return: a list of n-grams
    """
    return [text[i:i + n] for n in n_vals for i in range(len(text) - n + 1)]
