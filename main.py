# This is a sample Python script.

import numpy as np
import pandas as pd
import nltk
import re
import networkx as nx

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('stopwords')
# nltk.download('punkt') # 执行一次就可以了


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi():
    # Use a breakpoint in the code line below to debug your script.
    df = pd.read_csv("./tennis_articles_v4.csv")
    sentences = []
    for s in df['article_text']:
        sentences.append(sent_tokenize(s))

    sentences = [y for x in sentences for y in x]  # flatten list

    print(sentences[:8])

    word_embeddings = {}
    f = open('./glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]



    stop_words = stopwords.words('english')

    clean_sentences = [remove_stopwords(r.split(), stop_words) for r in clean_sentences]

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    sim_mat = np.zeros([len(sentences), len(sentences)])

    # 使用余弦相似来计算两个句子间的相似度


    # 使用句子间的相似度初始化矩阵
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = \
                cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # 选取前10个句子作为摘要
    for i in range(1):
        print(ranked_sentences[i][1])



def remove_stopwords(sen, stop_words):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
