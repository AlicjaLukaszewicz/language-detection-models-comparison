LANGUAGE_LABELS = ['deu', 'eng', 'spa', 'fra', 'ita', 'pol', 'por', 'nld']
BERTA_LANGUAGES = ['de', 'en', 'es', 'fr', 'it', 'pl', 'pt', 'nl']

LANGUAGES_FULL = ['German', 'English', 'Spanish', 'French',
                  'Italian', 'Polish', 'Portuguese', 'Romanian', 'Dutch']

# map BERTA_LANGUAGES to LANGUAGE_LABELS
BERTA_TO_LABELS = dict(zip(BERTA_LANGUAGES, LANGUAGE_LABELS))
LABELS_TO_BERTA = dict(zip(LANGUAGE_LABELS, BERTA_LANGUAGES))

BERTA_TO_FULL = dict(zip(BERTA_LANGUAGES, LANGUAGES_FULL))
LABELS_TO_FULL = dict(zip(LANGUAGE_LABELS, LANGUAGES_FULL))