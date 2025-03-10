import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
from sklearn.feature_extraction.text import CountVectorizer

# Download stopwords if not already available
nltk.download('stopwords')
# Initialize stopwords and stemmer
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


