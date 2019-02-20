import pickle
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText

def train_word2vec(corpus_file_path, vector_size=100, window_size=5, architecture='cbow', train_algorithm='softmax', sub_sample=0.001):
    """architecture: 'skim-gram' or 'cbow'. train_algorithm: 'softmax' or 'negative'"""
    arch = 1 if architecture=='skip-gram' else 0
    train = 1 if train_algorithm=='softmax' else 0
    model = Word2Vec(corpus_file=corpus_file_path, size=vector_size, window=window_size, min_count=1, workers=4, sg=arch, hs=train, sample=sub_sample)
    return model

def load_word_embeddings_bin(filename, algorithm='fasttext'):
    model = lambda: None
    if(algorithm == 'fasttext'):
        model = FastText.load_fasttext_format(filename, encoding='utf8')
    elif(algorithm == 'word2vec'):
        model = KeyedVectors.load_word2vec_format(filename, encoding='utf8', binary=True)
    return model

def avg_word_emb(tokens, embedding_size, model):
    vec = np.zeros(embedding_size) #word embedding
    #make up for varying lengths with zero-padding
    n = len(tokens)
    for w_i in range(n):
        token = tokens[w_i]
        if (token in model.wv):
            vec += model.wv[token]
    #Average word embeddings
    return vec/n