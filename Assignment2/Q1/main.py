import pandas as pd
import numpy as np
from naive_bayes import *
from PIL import Image 
import matplotlib.pyplot as plt #
from wordcloud import WordCloud, STOPWORDS
from tokenizer import *
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, accuracy_score
from tabulate import tabulate
import seaborn as sns

def train(df_train, df_test, smoothening, tokenizer, column_name=["Description"], dump = True):
    df_train["Concatenated Data"] = df_train[column_name].astype(str).agg(" ".join, axis=1)
    df_test["Concatenated Data"] = df_test[column_name].astype(str).agg(" ".join, axis=1)
    
    df_train[f"Tokenized Data"] = df_train["Concatenated Data"].apply(tokenizer)
    df_test[f"Tokenized Data"] = df_test["Concatenated Data"].apply(tokenizer)
    
    model = NaiveBayes()
    model.fit(df_train, smoothening=smoothening
                , class_col="Class Index", text_col=f"Tokenized Data")

    model.predict(df_test, text_col=f"Tokenized Data", predicted_col="Predicted")
    num_correct = (df_test["Class Index"] == df_test["Predicted"]).sum()
    accuracy = num_correct / len(df_test)
    precision = precision_score(df_test["Class Index"], df_test["Predicted"], average="weighted")
    recall = recall_score(df_test["Class Index"], df_test["Predicted"], average="weighted")
    f1 = f1_score(df_test["Class Index"], df_test["Predicted"], average="weighted")
    
    if dump:
        print("--"*20)
        print(f"Tokenizer: {tokenizer.__name__}")
        print(f"Column: {column_name}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1-score: {f1:.2%}")
        print("--"*20)
    return accuracy, precision, recall, f1

def word_cloud(df_train, df_test, smoothening, tokenizer, qnum = "1_1b", column_name = ["Description"]):
    df_train["Concatenated Data"] = df_train[column_name].astype(str).agg(" ".join, axis=1)
    df_test["Concatenated Data"] = df_test[column_name].astype(str).agg(" ".join, axis=1)
    
    df_train[f"Tokenized Data"] = df_train["Concatenated Data"].apply(tokenizer)
    df_test[f"Tokenized Data"] = df_test["Concatenated Data"].apply(tokenizer)

    model = NaiveBayes()
    model.fit(df_train, smoothening=smoothening
                , class_col="Class Index", text_col=f"Tokenized Data")
    
    for cls in model.classes:
        log_word_freq = model.word_likelihoods[cls]  # Log-likelihoods P(w|C)
        
        # Convert log-likelihoods back to normal probabilities
        word_freq = {word: np.exp(log_prob) for word, log_prob in log_word_freq.items()}
        
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        # Plot the word cloud
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud for Class {cls}")
        plt.savefig(f"q{qnum}_word_cloud_{cls}.png")
        print(f"Saved q{qnum}_word_cloud_{cls}.png")

def compare_methods(df_train, df_test, column_name ,smoothening=1, qnum = "1_4"):
    results = []
    tokenizers = [simple_split, stemming_stopword_removal, unigram_bigram_tokenizer]  # Add other tokenizers if needed
    for tokenizer in tokenizers:
        accuracy, precision, recall, f1 = train(df_train, df_test, smoothening, tokenizer, column_name = column_name, dump=False)
        results.append([tokenizer.__name__, accuracy, precision, recall, f1])
        word_cloud(df_train, df_test, smoothening, tokenizer, qnum = f"{tokenizer.__name__}_{qnum}", column_name = ["Title"])
    
    # Convert to DataFrame and display as table
    results_df = pd.DataFrame(results, columns=["Tokenizer", "Accuracy", "Precision", "Recall", "F1-score"])
    print(column_name)
    print(tabulate(results_df, headers="keys", tablefmt="grid"))
    
    return results_df

def q1_1a(df_train, df_test, smoothening = 1):
    train(df_train, df_test, smoothening, tokenizer=simple_split)

def q1_1b(df_train, df_test, smoothening = 1):
    word_cloud(df_train, df_test, smoothening, tokenizer=simple_split, qnum="1_1b")

def q1_2a(df_train, df_test, smoothening = 1):
    train(df_train, df_test, smoothening, tokenizer=stemming_stopword_removal)

def q1_2b(df_train, df_test, smoothening = 1):
    word_cloud(df_train, df_test, smoothening, tokenizer=stemming_stopword_removal, qnum="1_2b")

def q1_3(df_train, df_test, smoothening = 1):
    train(df_train, df_test, smoothening, tokenizer=unigram_bigram_tokenizer)
    word_cloud(df_train, df_test, smoothening, tokenizer=unigram_bigram_tokenizer, qnum="1_3")

def q1_4(df_train, df_test, smoothening=1):
    compare_methods(df_train, df_test, column_name=["Description"], smoothening=1)

def q1_5(df_train, df_test, smoothening = 1):
    compare_methods(df_train, df_test, column_name=["Title"], smoothening=1)

def q1_6a(df_train, df_test, smoothening = 1):
    train(df_train, df_test, smoothening = 1, tokenizer = unigram_bigram_tokenizer, column_name=["Description", "Title"])
    word_cloud(df_train, df_test, smoothening, tokenizer = unigram_bigram_tokenizer, qnum="1_6a", column_name=["Description", "Title"])

def q1_6b(df_train, df_test, smoothening = 1):
    column_name=["Description", "Title"]
    tokenizer = unigram_bigram_tokenizer

    for col in column_name:    
        df_train[f"Tokenized {col}"] = df_train[col].apply(tokenizer)
        df_test[f"Tokenized {col}"] = df_test[col].apply(tokenizer)
        df_train[f"Cleaned {col}"] = [" ".join([s + f"_{col}" for s in lst]) for lst in df_train[f"Tokenized {col}"]]
        df_test[f"Cleaned {col}"] = [" ".join([s + f"_{col}" for s in lst]) for lst in df_test[f"Tokenized {col}"]]

    
    df_train["Tokenized Data"] = df_train[column_name].apply(lambda row: tokenizer(" ".join(str(row[col]) for col in column_name)), axis=1)
    df_test["Tokenized Data"] = df_test[column_name].apply(lambda row: tokenizer(" ".join(str(row[col]) for col in column_name)), axis=1)

    model = NaiveBayes()
    model.fit(df_train, smoothening=smoothening
                , class_col="Class Index", text_col=f"Tokenized Data")

    model.predict(df_test, text_col=f"Tokenized Data", predicted_col="Predicted")
    num_correct = (df_test["Class Index"] == df_test["Predicted"]).sum()
    accuracy = num_correct / len(df_test)
    precision = precision_score(df_test["Class Index"], df_test["Predicted"], average="weighted")
    recall = recall_score(df_test["Class Index"], df_test["Predicted"], average="weighted")
    f1 = f1_score(df_test["Class Index"], df_test["Predicted"], average="weighted")
    
    print("--"*20)
    print(f"Tokenizer: {tokenizer.__name__}")
    print(f"Column: {column_name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-score: {f1:.2%}")
    print("--"*20)
    df_test.to_csv("q1_6b_test.csv", index=False)
    df_train.to_csv("q1_6b_train.csv", index=False)

    qnum = "1_6b"
    for cls in model.classes:
        log_word_freq = model.word_likelihoods[cls]  # Log-likelihoods P(w|C)
        
        # Convert log-likelihoods back to normal probabilities
        word_freq = {word: np.exp(log_prob) for word, log_prob in log_word_freq.items()}
        
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        # Plot the word cloud
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud for Class {cls}")
        plt.savefig(f"q{qnum}_word_cloud_{cls}.png")
        print(f"Saved q{qnum}_word_cloud_{cls}.png")
    return accuracy, precision, recall, f1

def q1_7(df_train, df_test, smoothening = 1):
    class_col = "Class Index"
    num_classes = df_test[class_col].nunique()
    random_guess_acc = 1 / num_classes  # Random prediction accuracy
    
    # Most common class prediction accuracy
    majority_class = df_test[class_col].value_counts().idxmax()
    majority_class_acc = (df_test[class_col] == majority_class).sum() / len(df_test)
    print(f"Random Guess Accuracy: {random_guess_acc:.2%}")
    print(f"Majority Class Accuracy: {majority_class_acc:.2%}")

    return random_guess_acc, majority_class_acc

def q1_8(df_train, df_test, smoothening = 1):
    column_name=["Description", "Title"]
    tokenizer = unigram_bigram_tokenizer

    for col in column_name:    
        df_train[f"Tokenized {col}"] = df_train[col].apply(tokenizer)
        df_test[f"Tokenized {col}"] = df_test[col].apply(tokenizer)
        df_train[f"Cleaned {col}"] = [" ".join([s + f"_{col}" for s in lst]) for lst in df_train[f"Tokenized {col}"]]
        df_test[f"Cleaned {col}"] = [" ".join([s + f"_{col}" for s in lst]) for lst in df_test[f"Tokenized {col}"]]

    
    df_train["Tokenized Data"] = df_train[column_name].apply(lambda row: tokenizer(" ".join(str(row[col]) for col in column_name)), axis=1)
    df_test["Tokenized Data"] = df_test[column_name].apply(lambda row: tokenizer(" ".join(str(row[col]) for col in column_name)), axis=1)

    model = NaiveBayes()
    model.fit(df_train, smoothening=smoothening
                , class_col="Class Index", text_col=f"Tokenized Data")

    model.predict(df_test, text_col=f"Tokenized Data", predicted_col="Predicted")
    num_correct = (df_test["Class Index"] == df_test["Predicted"]).sum()
    accuracy = num_correct / len(df_test)
    precision = precision_score(df_test["Class Index"], df_test["Predicted"], average="weighted")
    recall = recall_score(df_test["Class Index"], df_test["Predicted"], average="weighted")
    f1 = f1_score(df_test["Class Index"], df_test["Predicted"], average="weighted")
    
    print("--"*20)
    print(f"Tokenizer: {tokenizer.__name__}")
    print(f"Column: {column_name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-score: {f1:.2%}")
    print("--"*20)
    cm = confusion_matrix(df_test["Class Index"], df_test["Predicted"])
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df_test["Class Index"].unique()), yticklabels=sorted(df_test["Class Index"].unique()))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("q1_8_confusion_matrix.png")

def q1_9(df_train, df_test, smoothening = 1):
    column_name=["Description", "Title"]
    tokenizer = multigram_tokenizer
    
    for col in column_name:    
        df_train[f"Tokenized {col}"] = df_train[col].apply(tokenizer)
        df_test[f"Tokenized {col}"] = df_test[col].apply(tokenizer)
        df_train[f"Cleaned {col}"] = [" ".join([s + f"_{col}" for s in lst]) for lst in df_train[f"Tokenized {col}"]]
        df_test[f"Cleaned {col}"] = [" ".join([s + f"_{col}" for s in lst]) for lst in df_test[f"Tokenized {col}"]]
    

    df_train["Tokenized Data"] = df_train[column_name].apply(lambda row: tokenizer(" ".join(str(row[col]) for col in column_name)), axis=1)
    df_test["Tokenized Data"] = df_test[column_name].apply(lambda row: tokenizer(" ".join(str(row[col]) for col in column_name)), axis=1)

    vectorizer = TfidfVectorizer()
    X_train_combined = vectorizer.fit_transform(df_train["Tokenized Data"].apply(lambda x: " ".join(x)))
    X_test_combined = vectorizer.transform(df_test["Tokenized Data"].apply(lambda x: " ".join(x)))
    
    model = MultinomialNB()
    model.fit(X_train_combined, df_train["Class Index"])  # Assuming you have a "Label" column
    y_pred = model.predict(X_test_combined)

    # Evaluate Performance
    accuracy = accuracy_score(df_test["Class Index"], y_pred)
    precision = precision_score(df_test["Class Index"], y_pred, average="weighted")
    recall = recall_score(df_test["Class Index"], y_pred, average="weighted")
    f1 = f1_score(df_test["Class Index"], y_pred, average="weighted")
    
    print("--"*20)
    print(f"Tokenizer: {tokenizer.__name__}")
    print(f"Column: {column_name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-score: {f1:.2%}")
    print("--"*20)
    cm = confusion_matrix(df_test["Class Index"], y_pred)
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df_test["Class Index"].unique()), yticklabels=sorted(df_test["Class Index"].unique()))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("q1_8_confusion_matrix.png")



df_train = pd.read_csv('../data/Q1/train.csv', header=0)
df_test = pd.read_csv('../data/Q1/test.csv', header=0)


# q1_1a(df_train, df_test)
# q1_1b(df_train, df_test)
# q1_2a(df_train, df_test)
# q1_2b(df_train, df_test)
# q1_3(df_train,df_test)
# q1_4(df_train, df_test)
# q1_5(df_train, df_test)
# q1_6a(df_train, df_test)
# q1_6b(df_train, df_test)
# q1_7(df_train, df_test)
# q1_8(df_train, df_test)
q1_9(df_train, df_test)
