import math
from typing import Dict, List

from statistical_approach.src.models.build_model import build_model


def calculate_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Calculate the cosine similarity between two dictionaries of n-grams

    :param a: a dictionary of n-grams and their probabilities
    :param b: a dictionary of n-grams and their probabilities
    :return: the cosine similarity between the two dictionaries
    """
    numerator = sum([a[k] * b[k] for k in a if k in b])
    denominator = (math.sqrt(sum([a[k] ** 2 for k in a])) * math.sqrt(sum([b[k] ** 2 for k in b])))
    return numerator / denominator


def identify_language(text: str, language_models: Dict[str, Dict[str, float]], n_vals: List[int]) -> str:
    """
    Identify the language of a given text.

    :param text:  the text to identify
    :param language_models:  a dictionary of language models, where each key is a language name and each value is a dictionary of ngram: probability pairs
    :param n_vals:  a list of n_gram sizes to extract to build a model of the test
    :return:  language model whose n_gram probabilities best match those of the test text
    """
    text_model = build_model(text, n_vals)
    language = ""
    max_cosine = 0
    for model in language_models:
        cosine = calculate_cosine(language_models[model], text_model)
        if cosine > max_cosine:
            max_cosine = cosine
            language = model
    return language
