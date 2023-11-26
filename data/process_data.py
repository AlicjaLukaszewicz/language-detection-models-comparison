# Split sentences into multiple parts with different lengths and save to a file
import random

from data.global_variables import LANGUAGE_LABELS


def split_sentences(sentences_file: str, labels_file: str, n_vals: list, output_sentences_file: str,
                    output_labels_file: str):
    """
    Splits sentences into parts with different numbers of words and creates new files with split sentences and labels.

    :param sentences_file: The path to the file containing the sentences.
    :param labels_file: The path to the file containing the corresponding labels.
    :param n_vals: A list of n-gram values to split the sentences.
    :param output_sentences_file: The path to the output file for the split sentences.
    :param output_labels_file: The path to the output file for the corresponding labels.
    """
    with open(sentences_file, 'r', encoding='utf-8') as f_sentences, open(labels_file, 'r',
                                                                          encoding='utf-8') as f_labels, open(
        output_sentences_file, 'w', encoding='utf-8') as f_output_sentences, open(output_labels_file, 'w',
                                                                                  encoding='utf-8') as f_output_labels:
        sentences = f_sentences.readlines()
        labels = f_labels.readlines()

        for sentence, label in zip(sentences, labels):
            sentence = sentence.strip()
            words = sentence.split()
            # If it is a train data delete numbers and special characters
            if output_sentences_file == "processed/x_train.txt":
                words = [word for word in words if word.isalpha()]

            for n in n_vals:
                split_parts = [words[i:i + n] for i in range(len(words) - n + 1)]
                split_sentences = [' '.join(part) for part in split_parts]

                for split_sentence in split_sentences:
                    f_output_sentences.write(f"{split_sentence}\n")
                    f_output_labels.write(f"{label}")


# Split sentences into multiple parts with different lengths specified in list n_vals and save to a file
def process_sentences(sentences_file: str, labels_file: str, n_vals: list, output_sentences_file: str,
                      output_labels_file: str):
    with open(sentences_file, 'r', encoding='utf-8') as f_sentences, open(labels_file, 'r') as f_labels:
        sentences = f_sentences.readlines()
        labels = f_labels.readlines()

    # Group sentences by language
    language_sentences = {}
    for sentence, label in zip(sentences, labels):
        label = label.strip()
        if label in LANGUAGE_LABELS:
            if label not in language_sentences:
                language_sentences[label] = []
            # Delete numbers and special characters
            sentence = sentence.lower()
            sentence = sentence.strip()
            sentence = ' '.join([word for word in sentence.split() if word.isalpha()])
            language_sentences[label].append(sentence.strip())

    # For each language in language_sentences split sentences into parts with different lengths and save to a file
    with open(output_sentences_file, 'w', encoding='utf-8') as f_output, open(output_labels_file, 'w') as f_labels:
        for language, sentences in language_sentences.items():
            for n in n_vals:
                num_sentences = len(sentences)
                num_random_sentences = int(num_sentences * 1 / len(n_vals))
                # Get random sentences from the language that have at least n words
                filtered_sentences = [sentence for sentence in sentences if len(sentence.split()) >= n]
                random_sentences = random.sample(filtered_sentences, num_random_sentences)

                for sentence in random_sentences:
                    words = sentence.split()
                    # Get random n words from the sentence
                    random_words = random.sample(words, n)
                    # Write the selected words to the output file
                    f_output.write(f"{' '.join(random_words)}\n")
                    f_labels.write(f"{language}\n")


if __name__ == '__main__':
    process_sentences("raw/x_train.txt", "raw/y_train.txt",
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      "processed/x_train.txt", "processed/y_train.txt")
    process_sentences("raw/x_test.txt", "raw/y_test.txt",
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      "processed/x_test.txt", "processed/y_test.txt")
