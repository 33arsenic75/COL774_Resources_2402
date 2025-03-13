import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
from sklearn.feature_extraction.text import CountVectorizer

MAX_NGRAM = 5

# Download stopwords if not already available
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def simple_split(text):
    return text.split()

def stemming_stopword_removal(text):
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize (split into words)
    words = text.split()

    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in words if word not in stop_words]

    return tokens

def unigram_bigram_tokenizer(text):
    tokens = stemming_stopword_removal(text)
    unigrams = tokens
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    return unigrams + bigrams

def bigrams(text):
    tokens = stemming_stopword_removal(text)
    unigrams = tokens
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    return bigrams

def unigrams(text):
    tokens = stemming_stopword_removal(text)
    unigrams = tokens
    return unigrams

def skipgrams(tokens, k=2):
    return [" ".join([tokens[i], tokens[i+j]]) for i in range(len(tokens)) for j in range(2, k+1) if i+j < len(tokens)]

def char_ngrams(text, n=3):
    return [text[i:i+n] for i in range(len(text)-n+1)]

def multigram_tokenizer(text, max_n=MAX_NGRAM):
    tokens = stemming_stopword_removal(text)
    unigrams = tokens
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    rev_bigrams = [" ".join([tokens[i+1], tokens[i]]) for i in range(len(tokens)-1)]
    rev_trigrams = [" ".join([tokens[i+2], tokens[i+1], tokens[i]]) for i in range(len(tokens)-2)]
    skip_bigrams = skipgrams(tokens, k=2)
    char_trigrams = char_ngrams(" ".join(tokens), n=3)
    window_phrases = [" ".join(tokens[i:i+4]) for i in range(len(tokens)-3)]
    
    return unigrams + bigrams + trigrams + rev_bigrams + rev_trigrams + skip_bigrams + char_trigrams + window_phrases


