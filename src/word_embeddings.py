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
datafolder = '../data/'
fasttext_data = os.path.join(datafolder, 'fasttext/fasttext_da_300.kv')
word2vec_data = lambda dim: os.path.join(datafolder, 'word2vec/dsl_sentences_{0}_cbow_negative.kv'.format(dim))

wv_model = None
vector_size = 300

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

def train_save_word2vec(corpus_file_path, word2vec_format=False, save_model=False, 
                        vector_size=100, architecture='cbow', train_algorithm='negative', workers=4):
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
    filename = "{0}{1}_{2}_{3}".format(s, vector_size, architecture, train_algorithm)
    if save_model:
        print('Saving model in {0}.model'.format(filename))
        model.save(os.path.join(word2vec_path, "{}.model".format(filename)))
    if word2vec_format:
        print('Saving word embeddings in original C word2vec (.txt) format in {}.txt'.format(filename))
        model.wv.save_word2vec_format(os.path.join(word2vec_path, "{}.txt".format(filename)))
    else:
        print('Saving word embeddings in {0}.kv'.format(filename))
        model.wv.save(os.path.join(word2vec_path, "{}.kv".format(filename)))
        print('Saved!')
    return model

def save_fasttext(path_to_vectors, saved_filename):
    model = load_word_embeddings_bin(path_to_vectors)
    print('Saving word embeddings')
    model.wv.save(os.path.join(fasttext_path, saved_filename))
    print('Done!')
    

def load_saved_word_embeddings(w2v, fasttext):
    global wv_model
    if w2v:
        wv_model = KeyedVectors.load(word2vec_data(w2v))
        global vector_size
        vector_size = w2v
    elif fasttext:
        wv_model = KeyedVectors.load(fasttext_data)
    return wv_model

def load_word_embeddings_bin(filename, algorithm='fasttext'):
    print('loading model...')
    global wv_model
    if(algorithm == 'fasttext'):
        wv_model = FastText.load_fasttext_format(filename, encoding='utf8')
    elif(algorithm == 'word2vec'):
        wv_model = KeyedVectors.load_word2vec_format(filename, encoding='utf8', binary=True)
    print('Done!')
    return wv_model

def avg_word_emb(tokens):
    global wv_model
    if not wv_model:
        return None
    vec = np.zeros(vector_size) # word embedding
    # make up for varying lengths with zero-padding
    n = len(tokens)
    if n == 0:
        return vec.tolist()
    for w_i in range(n):
        token = tokens[w_i]
        if token in wv_model.vocab:
            vec += wv_model[token]
    # Average word embeddings
    return (vec/n).tolist()

def cosine_similarity(one, other):
    global wv_model
    if not wv_model:
        return None

    # Lookup words in w2c vocab
    words = []
    for token in one:
        if token in wv_model.vocab:  # check that the token exists
            words.append(token)
    other_words = []
    for token in other:
        if token in wv_model.vocab:
            other_words.append(token)

    if len(words) > 0 and len(other_words) > 0:  # make sure there is actually something to compare
        # cosine similarity between two sets of words
        return wv_model.n_similarity(other_words, words)
    else:
        return 0.  # no similarity if one set contains 0 words


def most_similar_word(word):
    global wv_model
    if wv_model and word in wv_model.vocab:
        return wv_model.similar_by_word(word)
    return [(word, 1)]

def main(argv):
    # arguments setting 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='*', default=[dsl_sentences], help='Input file to train and save model for')
    parser.add_argument('-v', '--vector_size', type=int, default=100, help='the size of a word vector')
    parser.add_argument('--architecture', type=str, default='cbow', help='the architecture: "skip-gram" or "cbow"')
    parser.add_argument('--train_algorithm', type=str, default='negative', help='the training algorithm: "softmax" or "negative"')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--word2vec_format', action='store_true', default=False, help='Store in the original C word2vec (.txt) format')
    args = parser.parse_args(argv)

    input_files = args.input_files
    vector_size = args.vector_size
    architecture = args.architecture
    train_algorithm = args.train_algorithm
    word2vec_format = args.word2vec_format
    workers = args.workers
    train_save_word2vec(input_files, word2vec_format=word2vec_format , vector_size=vector_size, 
                        architecture=architecture, train_algorithm=train_algorithm, workers=workers)

if __name__ == "__main__":
    main(sys.argv[1:])