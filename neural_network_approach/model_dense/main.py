import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from data.global_variables import LANGUAGE_LABELS
from neural_network_approach.model_dense.functions import get_trigrams, encode


# Read in x_train file
with open('../../data/processed/x_train.txt', 'r', encoding='utf-8') as f:
    x_train = [s.strip() for s in f.readlines()]

# Read in y_train file
with open('../../data/processed/y_train.txt', 'r', encoding='utf-8') as f:
    y_train = [s.strip() for s in f.readlines()]

# Create DataFrame from x_train and y_train
data = pd.DataFrame({'lang': y_train, 'text': x_train})
data['sentence_length'] = data['text'].apply(lambda x: len(x.split()))

# Shuffle and save data
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv('train_data.csv', index=False)

# Split data into train and valid
train = data.iloc[:int(len(data) * 0.8)]
valid = data.iloc[int(len(data) * 0.8):]

# Read in x_test file
with open('../../data/processed/x_test.txt', 'r', encoding='utf-8') as f:
    x_test = [s.strip() for s in f.readlines()]

# Read in y_test file
with open('../../data/processed/y_test.txt', 'r', encoding='utf-8') as f:
    y_test = [s.strip() for s in f.readlines()]

# Create DataFrame from x_test and y_test
test = pd.DataFrame({'lang': y_test, 'text': x_test})
test['sentence_length'] = test['text'].apply(lambda x: len(x.split()))

# Shuffle and save data
test = test.sample(frac=1).reset_index(drop=True)
test.to_csv('test_data.csv', index=False)

# Extract trigrams for each language
features = {}
features_set = set()

for language in LANGUAGE_LABELS:
    corpus = train[train.lang == language]['text']
    trigrams = get_trigrams(corpus)
    features[language] = trigrams
    features_set.update(trigrams)

# Create vocabulary list using feature set
vocab = {f: i for i, f in enumerate(features_set)}

# Train count vectorizer using vocabulary
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3), vocabulary=vocab)

# Create feature matrix for training set
corpus = train['text']
X_train = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
train_feat = pd.DataFrame(data=X_train.toarray(), columns=feature_names)

# Scale feature matrix
train_min = train_feat.min()
train_max = train_feat.max()
train_feat = (train_feat - train_min) / (train_max - train_min)

# Add target variable
train_feat['lang'] = list(train['lang'])
train_feat['sentence_length'] = list(train['sentence_length'])

# Create feature matrix for validation set
corpus = valid['text']
X_valid = vectorizer.transform(corpus)
valid_feat = pd.DataFrame(data=X_valid.toarray(), columns=feature_names)
valid_feat = (valid_feat - train_min) / (train_max - train_min)
valid_feat['lang'] = list(valid['lang'])
valid_feat['sentence_length'] = list(valid['sentence_length'])

# Create feature matrix for test set
corpus = test['text']
X_test = vectorizer.transform(corpus)
test_feat = pd.DataFrame(data=X_test.toarray(), columns=feature_names)
test_feat = (test_feat - train_min) / (train_max - train_min)
test_feat['lang'] = list(test['lang'])
test_feat['sentence_length'] = list(test['sentence_length'])

# Fit encoder
encoder = LabelEncoder()
encoder.fit(LANGUAGE_LABELS)

# Get training data
x_train = train_feat.drop(['lang', 'sentence_length'], axis=1)
y_train = encode(encoder, train_feat['lang'])

x_valid = valid_feat.drop(['lang', 'sentence_length'], axis=1)
y_valid = encode(encoder, valid_feat['lang'])


# Define model
model = Sequential()
model.add(Dense(250, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(len(LANGUAGE_LABELS), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
num_epochs = 100
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=20,
                    validation_data=(x_valid, y_valid))

# Save model
model.save('results/models/model.h5')

# Get test data
x_test = test_feat.drop(['lang', 'sentence_length'], axis=1)
y_test = test_feat['lang']

# Get predictions on test set
labels = model.predict(x_test)

# Flatten predictions
labels = [list(l).index(max(l)) for l in labels]
predictions = encoder.inverse_transform(labels)

# Add predictions column to the test DataFrame
test['predictions'] = predictions

# Measure accuracy for each sentence length
accuracy_by_length = test.groupby('sentence_length').apply(lambda x: accuracy_score(x['lang'], x['predictions']))

# Overall accuracy for each sentence length
overall_accuracy_by_length = accuracy_by_length.mean()

# Plot the accuracy for each language by sentence length
for lang in LANGUAGE_LABELS:
    lang_accuracy_by_length = test[test['lang'] == lang].groupby('sentence_length').apply(
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
    lang_accuracy_by_length = test[test['lang'] == lang].groupby('sentence_length').apply(
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
