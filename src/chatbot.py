"""
Chatbot that integrates different models to parse the text and 
get the most probable answers.

Instruction to run:
    python chatbot.py
"""

import time
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText, Word2Vec, KeyedVectors, Doc2Vec
import pandas as pd
from tqdm import trange
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import gradio as gr  # ui

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


def preprocessing(df, model_type="fasttext"):
    """
    Pre-process the text for training.
    """
    para = [[] for _ in range(df.shape[0])]
    for i in trange(df.shape[0]):
        para[i] = preprocess_text(str(df.iloc[i][0]))

    if model_type == "fasttext":
        model_para = FastText(para, min_count=1)
    else:
        model_para = Word2Vec(para, min_count=1)
    return {
        "para": para,
        "model_para": model_para,
    }


def WMdistance(model, doc1, doc2):
    """
    Compute the Word Mover's distance between two documents.
    """
    return model.wv.wmdistance(doc1, doc2)


def get_answers(df, query1, para, model_para):
    """
    Get the answer from the model.
    """
    query = answer_preprocess(query1)
    q = preprocess_text(query)
    count = 0
    min1 = 1000
    result = ""
    distance = float("inf")
    for i in range(len(df)):
        distance = WMdistance(model_para, q, para[i])
        if distance < min1:
            min1 = distance
            result = df.iloc[i][1]
        count = count + 1
        # print ('distance = %.3f' % distance)
    return result, min1


df = pd.read_csv("./input/q_a.csv", encoding="unicode_escape")
# Available options - fasttext, word2vec
resource = preprocessing(df, model_type="word2vec")


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history):
    query = history[-1][0]
    if query.lower() == "bye":
        bot_message = "Thank you for using the chatbot.I hope you had a great time"
    else:
        q = preprocess_text(query)
        q1 = ""
        for d in q:
            q1 = q1 + d + " "
        res, dist = get_answers(df, q1, resource["para"], resource["model_para"])
        print(dist)
        if dist < 1:
            bot_message = res
        else:
            bot_message = (
                "Sorry I don't have the answer. Can you please rephrase the query"
            )
    history[-1][1] = bot_message
    time.sleep(1)
    return history


### initiating chat process
if __name__ == "__main__":
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            value=[
                ("Hello user", None),
                ("How may I help you", None),
                ("For exiting from the chatbot,just say bye", None),
            ],
            label="College Queries Chatbot",
            elem_id="chatbot",
        ).style(
            height=700,
        )
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    demo.launch()
