import re

from data.global_variables import LANGUAGE_LABELS


def load_data(sentences_file: str, languages_file: str) -> dict[str, str]:
    """
    Load sentences and their corresponding languages from two text files.

    :param sentences_file: path to the file containing sentences
    :param languages_file: path to the file containing corresponding languages
    :return: a dictionary mapping each sentence to its corresponding language
    """
    # Read sentences and their corresponding LANGUAGES from two text files
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    with open(languages_file, 'r', encoding='utf-8') as f:
        languages = f.readlines()

    # Delete newline characters
    sentences = [re.sub(r'\n', '', sent) for sent in sentences]
    languages = [re.sub(r'\n', '', lang) for lang in languages]
    # Delete special characters and numbers
    sentences = [re.sub(r'[^a-zA-Z ]', '', sent) for sent in sentences]
    # Convert to lowercase
    sentences = [sent.lower() for sent in sentences]

    # Build a dictionary to map each sentence to its corresponding language
    sentences_to_languages = {sentences[i].strip(): languages[i].strip() for i in range(len(sentences))}

    # Delete instances that are not present in list LANGUAGE_LABELS
    sentences_to_languages = {sent: lang for sent, lang in sentences_to_languages.items() if lang in LANGUAGE_LABELS}

    return sentences_to_languages
