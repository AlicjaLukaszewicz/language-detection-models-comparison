import matplotlib.pyplot as plt
import seaborn as sns

from statistical_approach.src.models.detect_language import identify_language
from statistical_approach.src.models.train_model import train_model


# For each language and sentence length, calculate the accuracy of the model
def evaluate_accuracy_by_sentence_length(sentences_file: str, labels_file: str, model):
    with open(sentences_file, 'r', encoding='utf-8') as f_sentences:
        sentences = f_sentences.readlines()
        sentences = [sentence.strip() for sentence in sentences]
    with open(labels_file, 'r', encoding='utf-8') as f_labels:
        labels = f_labels.readlines()
        labels = [label.strip() for label in labels]

    # Group sentences by language
    language_sentences = {}
    for sentence, label in zip(sentences, labels):
        if label not in language_sentences:
            language_sentences[label] = []
        language_sentences[label].append(sentence)

    accuracy_by_length = {}
    for language, sentences in language_sentences.items():
        accuracy_by_length[language] = {}
        for sentence in sentences:
            length = len(sentence.split())
            if length not in accuracy_by_length[language]:
                accuracy_by_length[language][length] = {'num_correct': 0, 'num_total': 0}
            num_correct = accuracy_by_length[language][length]['num_correct']
            num_total = accuracy_by_length[language][length]['num_total']
            num_total += 1
            prediction = identify_language(sentence, model, range(1, 2))
            if prediction == language:
                num_correct += 1
            accuracy_by_length[language][length]['num_correct'] = num_correct
            accuracy_by_length[language][length]['num_total'] = num_total

    return accuracy_by_length


if __name__ == '__main__':
    sentences_file = "../../../data/processed/x_test.txt"
    labels_file = "../../../data/processed/y_test.txt"
    model = train_model('../../../data/processed/x_train.txt', '../../../data/processed/y_train.txt', range(1, 2))

    accuracy_by_length = evaluate_accuracy_by_sentence_length(sentences_file, labels_file, model)

    for language, accuracy in accuracy_by_length.items():
        x = list(accuracy.keys())
        y = [value['num_correct'] / value['num_total'] for value in accuracy.values()]

        # Set font style and grid
        sns.set_style("whitegrid", {'font.family': 'serif', 'font.serif': 'Times New Roman'})

        # Create bar plot
        ax = sns.barplot(x=x, y=y, capsize=1, saturation=0.4, width=0.5)

        plt.ylim(0, 1)

        for i, v in enumerate(y):
            # Make text small enough to fit in the bar
            ax.text(i, v + 0.01, str(round(v, 2)), color='black', ha='center', va='bottom', fontsize=7)

        plt.xlabel('Sentence length')
        plt.ylabel('Accuracy')
        plt.title(language, pad=20)
        # Save the plot
        plt.savefig('../../results/png/accuracy_by_sentence_length_' + language + '.png')
        plt.show()
        # Clear the plot for the next iteration
        plt.clf()

    for language, accuracy in accuracy_by_length.items():
        x = list(accuracy.keys())
        y = [value['num_correct'] / value['num_total'] for value in accuracy.values()]
        plt.plot(x, y, label=language)
    plt.margins(0.1)
    plt.title('Accuracy by sentence length')
    plt.xlabel('Sentence length')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../../results/png/accuracy_by_sentence_length.png')
    plt.show()

    overall_accuracy = {}
    # Calculate overall accuracy for each language
    for language, accuracy_by_length in accuracy_by_length.items():
        num_correct = 0
        num_total = 0
        for length, accuracy in accuracy_by_length.items():
            num_correct += accuracy['num_correct']
            num_total += accuracy['num_total']
        overall_accuracy[language] = num_correct / num_total

    # Create bar plot
    ax = sns.barplot(x=list(overall_accuracy.keys()), y=list(overall_accuracy.values()), capsize=1, saturation=0.4)
    plt.ylim(0, 1)
    for i, v in enumerate(list(overall_accuracy.values())):
        # Make text small enough to fit in the bar
        ax.text(i, v + 0.01, str(round(v, 2)), color='black', ha='center', va='bottom', fontsize=10)
    plt.xlabel('Language')
    plt.ylabel('Accuracy')
    plt.title('Overall accuracy')
    # Save the plot
    plt.savefig('../../results/png/overall_accuracy.png')
    plt.show()

