"""
Chatbot that integrates different models to parse the text and 
get the most probable answers.

Instruction to run:
    python chatbot.py
"""

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText, Word2Vec, KeyedVectors, Doc2Vec
import pandas as pd
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath("glove.txt")
tmp_file = get_tmpfile("word2vec.txt")
_, _ = glove2word2vec(glove_file, tmp_file)

# Constants
nltk.download("wordnet")
tokenizer = RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = text.lower()
    token1 = tokenizer.tokenize(text)
    token = []
    for x in token1:
        if x not in stop_words:
            token.append(x)
    lemmatiz = [lemmatizer.lemmatize(tokens) for tokens in token]
    return lemmatiz


def answer_preprocess(file):
    token_files = []
    tokenizer = RegexpTokenizer(r"\w+")
    token_files = tokenizer.tokenize(str(file))
    for i in range(len(token_files)):
        if token_files[i] == "u" or token_files[i] == "U":
            token_files[i] = "you"
        elif token_files[i] == "d":
            token_files[i] = "the"
        elif token_files[i] == "n" or token_files[i] == "nd":
            token_files[i] = "and"
        elif token_files[i] == "hv":
            token_files[i] = "have"
        elif (
            token_files[i] == "bcoz"
            or token_files[i] == "becoz"
            or token_files == "bcz"
        ):
            token_files[i] = "because"
        elif token_files[i] == "ur" or token_files[i] == "Ur":
            token_files[i] = "your"
        elif token_files[i] == "thru":
            token_files[i] = "through"
    str1 = ""
    for word in token_files:
        str1 = str1 + word + " "
    return str1


def preprocessing(df, df_combined, model_type="fasttext"):
    """
    Pre-process the text for training.
    """
    topics = [[] for _ in range(df.shape[0])]
    para = [[] for _ in range(df.shape[0])]
    for i in range(df_combined.shape[0]):
        topics[i] = preprocess_text(str(df_combined.iloc[i][0]))
        para[i] = preprocess_text(str(df_combined.iloc[i][1]))
    if model_type == "doc2vec":
        model_topic = Doc2Vec(topics, min_count=1)
        model_para = Doc2Vec(para, min_count=1)
    if model_type == "fasttext":
        model_topic = FastText(topics, min_count=1)
        model_para = FastText(para, min_count=1)
    elif model_type == "glove":
        model_para = KeyedVectors.load_word2vec_format(tmp_file)
        model_topic = KeyedVectors.load_word2vec_format(tmp_file)
    else:
        model_topic = Word2Vec(topics, min_count=1)
        model_para = Word2Vec(para, min_count=1)
    model_para.init_sims(replace=True)
    model_topic.init_sims(replace=True)
    return {
        "para": para,
        "topic": topics,
        "model_para": model_para,
        "model_topic": model_topic,
    }


def WMdistance(model, doc1, doc2):
    """
    Compute the Word Mover's distance between two documents.
    """
    return model.wmdistance(doc1, doc2)


def get_answers(df_combined, query1, para, model_para):
    """
    Get the answer from the model.
    """
    query = answer_preprocess(query1)
    q = preprocess_text(query)
    count = 0
    min1 = 1000
    result = ""
    distance = float("inf")
    for i in range(len(df_combined)):
        distance = WMdistance(model_para, q, para[i])
        if distance < min1:
            min1 = distance
            result = df_combined.iloc[i][2]
        count = count + 1
        # print ('distance = %.3f' % distance)
    return result, distance


### initiating chat process
if __name__ == "__main__":
    df = pd.read_csv("Training.csv", encoding="unicode_escape")
    df_combined = df
    # Available options - fasttext, glove, word2vec
    resource = preprocessing(df, df_combined, model_type = "fasttext")
    print(
        "Hello user",
        "How may I help you",
        "For exiting from the chatbot,press 0",
        sep="\n",
    )
    x = 1
    while x != 0:
        print("\n\nEnter query")
        query = input()
        q = preprocess_text(query)
        q1 = ""
        for d in q:
            q1 = q1 + d + " "
        res, dist = get_answers(df, q1, resource["para"], resource["model_para"])
        if dist < 0.7:
            print(res)
        else:
            print("Sorry I don't have the answer. Can you please rephrase the query")
        print("If you have any more queries then press 1 else press 0")
        x = int(input())
        if x == 0:
            print("Thank you for using the chatbot.I hope you had a great time")
