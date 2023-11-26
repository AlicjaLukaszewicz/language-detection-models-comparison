from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer


def get_trigrams(_corpus, n_feat=200):
    # fit the n-gram model
    _vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3), max_features=n_feat)

    _X = _vectorizer.fit_transform(_corpus)

    # Get model feature names
    _feature_names = _vectorizer.get_feature_names_out()

    return _feature_names


def encode(encoder, y):
    y_encoded = encoder.transform(y)
    y_dummy = np_utils.to_categorical(y_encoded)

    return y_dummy