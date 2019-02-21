import sys, os
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText

word2vec_path = '../data/word2vec/'

def train_save_word2vec(corpus_file_path, vector_size=100, architecture='cbow', train_algorithm='softmax', workers=4):
    """architecture: 'skim-gram' or 'cbow'. train_algorithm: 'softmax' or 'negative'"""
    sentences = []
    with open(corpus_file, 'r', encoding='utf8') as corpus:
        for line in corpus:
            sentences.append(line.rstrip('\n').split(' '))
    arch = 1 if architecture=='skip-gram' else 0
    train = 1 if train_algorithm=='softmax' else 0
    print('Training...')
    model = Word2Vec(sentences=sentences, size=vector_size, min_count=1, workers=workers, sg=arch, hs=train)
    print('Done!')
    filename = "{0}_{1}_{2}_{3}".format(corpus_file_path.split('/')[-1].split('.')[0], vector_size, architecture, train_algorithm)
    print('Saving model and word embeddings in {0}.model and {0}.kv respectively'.format(filename))
    model.save(os.path.join(word2vec_path, "{}.model".format(filename)))
    model.wv.save(os.path.join(word2vec_path, "{}.kv".format(filename)))
    print('Saved!')
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

corpus_file = '../../Data/DSL_Corpus/dsl_sentences.txt'
train_save_word2vec(corpus_file)


