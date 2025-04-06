import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from nltk.tokenize import word_tokenize

MAX_NGRAM = 5

nltk.download('stopwords')

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



def multigram_tokenizer(text, max_n=MAX_NGRAM):
    tokens = stemming_stopword_removal(text)
    unigrams = tokens
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    quadgrams = [" ".join(tokens[i:i+4]) for i in range(len(tokens)-3)]
    alt_grams = [" ".join([tokens[i],tokens[i+2]]) for i in range(len(tokens)-2)]
    return unigrams + bigrams + quadgrams + alt_grams


