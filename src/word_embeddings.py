import sys, os
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText
import argparse

word2vec_path = '../data/word2vec/'
fasttext_path = '../data/fasttext/'
dsl_sentences = '../../Data/DSL_Corpus/dsl_sentences.txt'
wiki_sentences = '../../Data/Wiki_Corpus/wiki_sentences.txt'

# memory friendly iterator
class MySentences:
    def __init__(self, filenames):
        self.filenames = filenames

    def __iter__(self):
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf8') as corpus:
                for line in corpus:
                    line = line.rstrip('\n')
                    if line:
                        yield line.split()

def train_save_word2vec(corpus_file_path, save_model=False, vector_size=100, architecture='cbow', train_algorithm='negative', workers=4):
    """architecture: 'skim-gram' or 'cbow'. train_algorithm: 'softmax' or 'negative'"""
    sentences = MySentences(corpus_file_path)
    arch = 1 if architecture=='skip-gram' else 0
    train = 1 if train_algorithm=='softmax' else 0
    print('Training...')
    model = Word2Vec(sentences=sentences, size=vector_size, workers=workers, sg=arch, hs=train)
    print('Done!')
    s = ''
    for name in corpus_file_path:
        s += name.split('/')[-1].split('.')[0] + '_'
    filename = "{0}_{1}_{2}_{3}".format(s, vector_size, architecture, train_algorithm)
    if save_model:
        print('Saving model in {0}.model'.format(filename))
        model.save(os.path.join(word2vec_path, "{}.model".format(filename)))
    print('Saving word embeddings in {0}.kv'.format(filename))
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
    if (n == 0):
        return vec.tolist()
    for w_i in range(n):
        token = tokens[w_i]
        if (token in wembs):
            vec += wembs[token]
    #Average word embeddings
    return (vec/n).tolist()


def main(argv):
    # arguments setting 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='*', default=[dsl_sentences, wiki_sentences], help='Input file to train and save model for')
    parser.add_argument('--vector_size', type=int, default=100, help='the size of a word vector')
    parser.add_argument('--architecture', type=str, default='cbow', help='the architecture: "skip-gram" or "cbow"')
    parser.add_argument('--train_algorithm', type=str, default='negative', help='the training algorithm: "softmax" or "negative"')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    args = parser.parse_args(argv)

    input_files = args.input_files
    vector_size = args.vector_size
    architecture = args.architecture
    train_algorithm = args.train_algorithm
    workers = args.workers
    train_save_word2vec(input_files, vector_size=vector_size, architecture=architecture, train_algorithm=train_algorithm, workers=workers)

if __name__ == "__main__":
    main(sys.argv[1:])