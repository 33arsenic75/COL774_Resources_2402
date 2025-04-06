import numpy as np
import pandas as pd
from collections import defaultdict


class NaiveBayes:
    def __init__(self):
        self.class_prior: dict = {}
        self.word_likelihoods = defaultdict(lambda: defaultdict(float))  # P(w|C)
        self.vocab = set()
        self.classes: list = []
        self.smoothening: float = 1.0
        self.class_counts: dict = {}
        self.word_counts: dict = {}
        self.total_words_per_class = defaultdict(int)
        self.vocab_size: int = 0
        pass
        
    def fit(self, df, smoothening, class_col = "Class Index", text_col = "Tokenized Description"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        self.smoothening = smoothening
        self.class_counts = df[class_col].value_counts().to_dict()
        total_samples = len(df)
        self.classes = sorted(self.class_counts.keys())

        for cls in self.classes:
            self.class_prior[cls] = np.log(self.class_counts[cls] / total_samples)

        self.word_counts = {cls: defaultdict(int) for cls in self.classes}
        self.total_words_per_class = defaultdict(int)

        for _, row in df.iterrows():
            cls = row[class_col]
            words = row[text_col]
            for word in words:
                self.word_counts[cls][word] += 1
                self.total_words_per_class[cls] += 1
                self.vocab.add(word)
        
        self.vocab_size = len(self.vocab)
        
        # Compute likelihoods P(w|C) with Laplace smoothing
        for cls in self.classes:
            for word in self.vocab:
                self.word_likelihoods[cls][word] = np.log(
                    (self.word_counts[cls][word] + smoothening) / (self.total_words_per_class[cls] + self.smoothening * self.vocab_size))
    
    def predict(self, df, text_col = "Tokenized Description", predicted_col = "Predicted", log = False, log_col = "Log Likelihood"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        predictions = []
        if log:
            log_likelihoods = {cls: [] for cls in self.classes}
        for _, row in df.iterrows():
            words = row[text_col]
            class_scores = {}
            
            for cls in self.classes:
                class_scores[cls] = self.class_prior[cls]
                for word in words:
                    if word in self.vocab:
                        class_scores[cls] += self.word_likelihoods[cls][word]
                    else:
                        class_scores[cls] += np.log(self.smoothening / (self.total_words_per_class[cls] + self.smoothening * self.vocab_size))
            
            predictions.append(max(class_scores, key=class_scores.get))
        
        df[predicted_col] = predictions
        if log:
            for cls in self.classes:
                df[f"Log Likelihood {cls}"] = log_likelihoods[cls]
        return df