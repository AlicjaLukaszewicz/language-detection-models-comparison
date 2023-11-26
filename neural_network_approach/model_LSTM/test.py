import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.metrics import ConfusionMatrixDisplay

from data.global_variables import LANGUAGE_LABELS

train = pd.read_csv('train_data.csv')
x = train['text']
y = train['lang']

# Get 80% of data for training and 20% for validation
x_train = x[0: int(len(x) * 0.8)]
y_train = y[0: int(len(y) * 0.8)]

x_valid = x[int(len(x) * 0.8):]
y_valid = y[int(len(y) * 0.8):]

test = pd.read_csv('test_data.csv')
x_test = test['text']
y_test = test['lang']

x_train_tmp = []
y_train_tmp = []

x_valid_tmp = []
y_valid_tmp = []

x_test_tmp = []
y_test_tmp = []

for line in x_train:
    text_line = ""
    for c in line.lower():
        text_line = text_line + ' ' + c
    x_train_tmp.append(text_line)
for line in y_train:
    y_train_tmp.append(LANGUAGE_LABELS.index(line))

for line in x_valid:
    text_line = ""
    for c in line.lower():
        text_line = text_line + ' ' + c
    x_valid_tmp.append(text_line)
for line in y_valid:
    y_valid_tmp.append(LANGUAGE_LABELS.index(line))

for line in x_test:
    text_line = ""
    for c in line.lower():
        text_line = text_line + ' ' + c
    x_test_tmp.append(text_line)
for line in y_test:
    y_test_tmp.append(LANGUAGE_LABELS.index(line))

max_len = 0
for line in x_train_tmp:
    if len(line) > max_len:
        max_len = len(line)

print("Initializing Tokenizer and padding results")
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(x_train_tmp)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(x_train_tmp)
training_padded = pad_sequences(training_sequences, maxlen=max_len)

validating_sequences = tokenizer.texts_to_sequences(x_valid_tmp)
validating_padded = pad_sequences(validating_sequences, maxlen=max_len)

testing_sequences = tokenizer.texts_to_sequences(x_test_tmp)
testing_padded = pad_sequences(testing_sequences, maxlen=max_len)

training_padded = np.array(training_padded)
training_labels = np.array(y_train_tmp)
validating_padded = np.array(validating_padded)
validating_labels = np.array(y_valid_tmp)
testing_padded = np.array(testing_padded)
testing_labels = np.array(y_test_tmp)

print("Initializing Model")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 128, input_length=max_len),
    tf.keras.layers.LSTM(14, return_sequences=True),
    tf.keras.layers.LSTM(14),
    tf.keras.layers.Dense(len(LANGUAGE_LABELS), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Loading Model")
model.load_weights('test.h5')
model.summary()

print("Evaluate on test data")
results = model.evaluate(testing_padded, testing_labels, batch_size=128)
print("test loss, test acc:", results)

classifications = model.predict(testing_padded)
classifications = classifications.tolist()

confusion = np.zeros((len(LANGUAGE_LABELS), len(LANGUAGE_LABELS)))

for i in range(len(testing_padded)):
    res = classifications[i]
    res = res.index(max(res))
    confusion[testing_labels[i]][res] += 1

for i in range(len(LANGUAGE_LABELS)):
    confusion[i] = confusion[i] / confusion[i].sum()

for i in range(len(LANGUAGE_LABELS)):
    print(f"JEZ:{LANGUAGE_LABELS[i]} WYN: {confusion[i][i]}")

disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=LANGUAGE_LABELS)
disp.plot(xticks_rotation='vertical', include_values=False, )
plt.tight_layout()
plt.savefig("conf.png", pad_inches=5)
plt.show()
print(confusion)
print("Evaluate on test data")
results = model.evaluate(testing_padded, testing_labels, batch_size=128)
print("test loss, test acc:", results)

classifications = model.predict(testing_padded)
classifications = classifications.tolist()

confusion = np.zeros((len(LANGUAGE_LABELS), len(LANGUAGE_LABELS)))

for i in range(len(testing_padded)):
    res = classifications[i]
    res = res.index(max(res))
    confusion[testing_labels[i]][res] += 1

for i in range(len(LANGUAGE_LABELS)):
    confusion[i] = confusion[i] / confusion[i].sum()

for i in range(len(LANGUAGE_LABELS)):
    print(f"JEZ:{LANGUAGE_LABELS[i]} WYN: {confusion[i][i]}")

disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=LANGUAGE_LABELS)
disp.plot(xticks_rotation='vertical', include_values=False, )
plt.tight_layout()
plt.savefig("conf.png", pad_inches=5)
plt.show()
print(confusion)
