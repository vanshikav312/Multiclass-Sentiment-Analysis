import re
import string

import nltk
nltk.download('punkt',                      quiet=True)
nltk.download('stopwords',                  quiet=True)
nltk.download('wordnet',                    quiet=True)
nltk.download('omw-1.4',                    quiet=True)
nltk.download('punkt_tab',                  quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

STOPWORDS  = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

CONTRACTIONS = {
    "can't":"cannot",    "won't":"will not",    "don't":"do not",
    "doesn't":"does not","didn't":"did not",    "isn't":"is not",
    "aren't":"are not",  "wasn't":"was not",    "weren't":"were not",
    "haven't":"have not","hasn't":"has not",    "hadn't":"had not",
    "wouldn't":"would not","couldn't":"could not","shouldn't":"should not",
    "i'm":"i am",        "i've":"i have",       "i'll":"i will",
    "i'd":"i would",     "you're":"you are",    "you've":"you have",
    "he's":"he is",      "she's":"she is",      "it's":"it is",
    "we're":"we are",    "they're":"they are",  "they've":"they have",
    "that's":"that is",  "there's":"there is",  "what's":"what is",
    "let's":"let us",    "who's":"who is",      "ain't":"am not",
    "gonna":"going to",  "wanna":"want to",     "gotta":"got to",
    "kinda":"kind of",   "outta":"out of",
}

SLANG = {
    "tbh":"to be honest",   "idk":"i do not know",  "imo":"in my opinion",
    "omg":"oh my god",      "lol":"laughing",        "lmao":"laughing",
    "ngl":"not going to lie","irl":"in real life",   "rn":"right now",
    "abt":"about",          "bc":"because",          "cuz":"because",
    "tho":"though",         "thru":"through",        "ur":"your",
    "pls":"please",         "plz":"please",          "sry":"sorry",
    "ugh":"frustrated",     "smh":"shaking my head", "fml":"my life is terrible",
    "afaik":"as far as i know","btw":"by the way",
}

EMOTICONS = {
    ":)":"happy",    ":-)":"happy",   ":D":"happy",
    ":(":"sad",      ":-(":"sad",     ":'(":"crying sad",
    ":o":"surprised",":O":"surprised",
    ">:(":"angry",   "<3":"love",     "</3":"heartbreak",
    "^_^":"happy",   "-_-":"tired",   "T_T":"crying",
}


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN


def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Step 2 — Emoticons (must happen BEFORE punctuation removal)
    for emoticon, word in EMOTICONS.items():
        text = text.replace(emoticon, f' {word} ')

    # Step 3 — Lowercase
    text = text.lower()

    # Step 4 — Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)

    # Step 5 — Expand slang
    words = text.split()
    words = [SLANG.get(w, w) for w in words]
    text  = ' '.join(words)

    # Step 6 — Remove URLs and HTML
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)

    # Step 7 — Normalize repeated characters (sooooo → so, feeel → feel)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Step 8 — NEGATION HANDLING
    negation_words = {
        'not','no','never','neither','nor','nobody','nothing',
        'nowhere','without','cannot','hardly','barely','scarcely'
    }
    word_list    = text.split()
    result_words = []
    negate       = False
    negate_count = 0

    for word in word_list:
        if word in negation_words:
            negate       = True
            negate_count = 0
            result_words.append(word)
        elif negate and negate_count < 3:
            result_words.append(f'not_{word}')
            negate_count += 1
            if negate_count >= 3:
                negate = False
        else:
            negate = False
            result_words.append(word)

    text = ' '.join(result_words)

    # Step 9 — Punctuation removal (keep underscore to preserve not_word tokens)
    punct = string.punctuation.replace('_', '')
    text  = text.translate(str.maketrans('', '', punct))

    # Step 10 — Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)

    # Step 11 — Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 12 — Tokenize
    tokens = word_tokenize(text)

    # Step 13 — Stopword removal (preserve not_ negation tokens)
    tokens = [
        t for t in tokens
        if (t.startswith('not_') or t not in STOPWORDS) and len(t) > 1
    ]

    # Step 14 — POS-aware lemmatization (skip not_ tokens)
    pos_tagged = pos_tag(tokens)
    tokens = [
        LEMMATIZER.lemmatize(token, get_wordnet_pos(tag))
        if not token.startswith('not_')
        else token
        for token, tag in pos_tagged
    ]

    # Step 15 — Remove short tokens
    tokens = [t for t in tokens if len(t) > 2 or t.startswith('not_')]

    return ' '.join(tokens)


# ── Pipeline-compatible transformers (must live here so pickle can find them) ──

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class PreprocessedTfidf(BaseEstimator, TransformerMixin):
    """Applies preprocess_text() then TF-IDF. Takes raw text as input."""
    def __init__(self, **tfidf_kwargs):
        self.tfidf_kwargs = tfidf_kwargs
        self._tfidf = TfidfVectorizer(**tfidf_kwargs)

    def fit(self, X, y=None):
        self._tfidf.fit([preprocess_text(t) for t in X])
        return self

    def transform(self, X):
        return self._tfidf.transform([preprocess_text(t) for t in X])

    def get_params(self, deep=True):
        return {'tfidf_kwargs': self.tfidf_kwargs}

    def set_params(self, **params):
        if 'tfidf_kwargs' in params:
            self.tfidf_kwargs = params['tfidf_kwargs']
            self._tfidf = TfidfVectorizer(**self.tfidf_kwargs)
        return self


class VaderTransformer(BaseEstimator, TransformerMixin):
    """Extracts VADER sentiment scores from raw text: [compound, pos, neg, neu]."""
    def __init__(self):
        self._analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            s = self._analyzer.polarity_scores(str(text))
            rows.append([s['compound'], s['pos'], s['neg'], s['neu']])
        return sp.csr_matrix(np.array(rows, dtype=np.float32))
