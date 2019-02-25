import sys, os
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText
import argparse

word2vec_path = '../data/word2vec/'
fasttext_path = '../data/fasttext/'

def train_save_word2vec(corpus_file_path, vector_size=100, architecture='cbow', train_algorithm='negative', workers=4):
    """architecture: 'skim-gram' or 'cbow'. train_algorithm: 'softmax' or 'negative'"""
    sentences = []
    with open(corpus_file_path, 'r', encoding='utf8') as corpus:
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

def save_fasttext(path_to_vectors, saved_filename):
    model = load_word_embeddings_bin(path_to_vectors)
    print('Saving word embeddings')
    model.wv.save(os.path.join(fasttext_path, saved_filename))
    print('Done!')
    

def load_saved_word2vec_wv(filepath):
    return KeyedVectors.load(filepath)

def load_word_embeddings_bin(filename, algorithm='fasttext'):
    print('loading model...')
    model = lambda: None
    if(algorithm == 'fasttext'):
        model = FastText.load_fasttext_format(filename, encoding='utf8')
    elif(algorithm == 'word2vec'):
        model = KeyedVectors.load_word2vec_format(filename, encoding='utf8', binary=True)
    print('Done!')
    return model

def avg_word_emb(tokens, embedding_size, wembs):
    vec = np.zeros(embedding_size) #word embedding
    #make up for varying lengths with zero-padding
    n = len(tokens)
    for w_i in range(n):
        token = tokens[w_i]
        if (token in wembs):
            vec += wembs[token]
    #Average word embeddings
    return (vec/n).tolist()


# def main(argv):
#     # arguments setting 
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_file', type=str, help='Input file to train and save model for')
#     parser.add_argument('--vector_size', type=int, default=100, help='the size of a word vector')
#     parser.add_argument('--architecture', type=str, default='cbow', help='the architecture: "skip-gram" or "cbow"')
#     parser.add_argument('--train_algorithm', type=str, default='negative', help='the training algorithm: "softmax" or "negative"')
#     parser.add_argument('--workers', type=int, default=4, help='number of workers')
#     args = parser.parse_args()

#     input_file = args.input_file
#     vector_size = args.vector_size
#     architecture = args.architecture
#     train_algorithm = args.train_algorithm
#     workers = args.workers
#     train_save_word2vec(input_file, vector_size=vector_size, architecture=architecture, train_algorithm=train_algorithm, workers=workers)

if __name__ == "__main__":
    # main(sys.argv[1:])
    save_fasttext(os.path.join(fasttext_path, 'cc.da.300.bin'), 'fasttext_da_300.kv')