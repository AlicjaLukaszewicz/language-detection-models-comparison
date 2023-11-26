import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from data.global_variables import LANGUAGE_LABELS

# Load and preprocess train data
train_data = pd.read_csv('train_data.csv')
train_data = train_data.sample(frac=1).reset_index(drop=True)
x_train = train_data['text']
y_train = train_data['lang']

# Get 80% of the data for training and 20% for validation
x_valid = x_train[int(len(x_train) * 0.8):]
y_valid = y_train[int(len(y_train) * 0.8):]
x_train = x_train[:int(len(x_train) * 0.8)]
y_train = y_train[:int(len(y_train) * 0.8)]

# Load and preprocess test data
test_data = pd.read_csv('test_data.csv')
x_test = test_data['text']
y_test = test_data['lang']

# Tokenize and pad sequences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)

training_sequences = tokenizer.texts_to_sequences(x_train)
training_padded = pad_sequences(training_sequences, maxlen=400)

validating_sequences = tokenizer.texts_to_sequences(x_valid)
validating_padded = pad_sequences(validating_sequences, maxlen=400)

testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences, maxlen=400)

# Convert labels to indices
encoder = LabelEncoder()
encoder.fit(LANGUAGE_LABELS)

training_labels = encoder.transform(y_train)
validating_labels = encoder.transform(y_valid)

# Build and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 128, input_length=400),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(LANGUAGE_LABELS), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
num_epochs = 20
history = model.fit(
    training_padded, training_labels, epochs=num_epochs,
    validation_data=(validating_padded, validating_labels), verbose=2
)

# Save the model
model.save('results/models/test.h5')

# Get predictions on test set
labels = model.predict(testing_padded)
labels = [list(l).index(max(l)) for l in labels]
predictions = encoder.inverse_transform(labels)

# Add predictions column to the test DataFrame
test_data['predictions'] = predictions

# Measure accuracy for each sentence length
accuracy_by_length = test_data.groupby('sentence_length').apply(lambda x: accuracy_score(x['lang'], x['predictions']))

# Overall accuracy for each sentence length
overall_accuracy_by_length = accuracy_by_length.mean()

# Plot the accuracy for each language by sentence length
for lang in LANGUAGE_LABELS:
    lang_accuracy_by_length = test_data[test_data['lang'] == lang].groupby('sentence_length').apply(
        lambda x: accuracy_score(x['lang'], x['predictions']))
    plt.plot(lang_accuracy_by_length.index, lang_accuracy_by_length.values, label=lang)

plt.title('Accuracy by sentence length')
plt.xlabel('Sentence Length')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/png/accuracy_by_length.png')
plt.show()

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
conf_matrix_df = pd.DataFrame(conf_matrix, columns=LANGUAGE_LABELS, index=LANGUAGE_LABELS)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
sns.set(font_scale=1.5)
sns.heatmap(conf_matrix_df, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('results/png/confusion_matrix.png')
plt.show()

# Plot accuracy for each language
plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
sns.set(font_scale=1.5)
sns.barplot(x=LANGUAGE_LABELS,
            y=[round(conf_matrix[i][i] / sum(conf_matrix[i]) * 100, 2) for i in range(len(LANGUAGE_LABELS))])
for i, v in enumerate([round(conf_matrix[i][i] / sum(conf_matrix[i]) * 100, 2) for i in range(len(LANGUAGE_LABELS))]):
    plt.text(i, v + 1, str(v) + '%', fontsize=16, color='black', ha='center')
plt.title('Accuracy by Language', fontsize=22)
plt.xlabel('Language', fontsize=22)
plt.ylabel('Accuracy (%)', fontsize=22)
plt.savefig('results/png/accuracy_by_language.png')
plt.show()

# Plot accuracy based on models history
plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
sns.set(font_scale=1.5)
sns.lineplot(x=range(1, num_epochs + 1), y=history.history['accuracy'])
plt.xlabel('Epoch', fontsize=22)
plt.ylabel('Accuracy (%)', fontsize=22)
plt.savefig('results/png/model_accuracy.png')
plt.show()

# Plot loss based on models history
plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
sns.set(font_scale=1.5)
sns.lineplot(x=range(1, num_epochs + 1), y=history.history['loss'])
plt.xlabel('Epoch', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.savefig('results/png/model_loss.png')
plt.show()

# For each language plot a graph with accuracy for each sentence length
for lang in LANGUAGE_LABELS:
    lang_accuracy_by_length = test_data[test_data['lang'] == lang].groupby('sentence_length').apply(
        lambda x: accuracy_score(x['lang'], x['predictions']))
    plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
    for i, v in enumerate(lang_accuracy_by_length.values):
        plt.text(i - 0.1, v + 1, str(round(v, 2)) + '%', fontsize=16)
    plt.plot(lang_accuracy_by_length.index, lang_accuracy_by_length.values, label=lang)
    sns.set(font_scale=1.5)
    plt.xlabel('Sentence Length', fontsize=22)
    plt.ylabel('Accuracy (%)', fontsize=22)
    plt.title('Accuracy by Sentence Length for ' + lang, fontsize=22)
    plt.legend()
    plt.savefig('results/png/accuracy_by_length_' + lang + '.png')
    plt.show()
    plt.clf()
