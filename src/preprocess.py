import os, json, csv, sys
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText

from classes.CommentAnnotation import CommentAnnotation

datafolder = '../data/'
preprocessed_folder = os.path.join(datafolder, 'preprocessed/')
fasttext_data = os.path.join(datafolder, 'fasttext/cc.da.300.bin')
word2vec_data = os.path.join(datafolder, 'word2vec/da.bin')

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

swear_words = []
negation_words = []

with open('../data/lexicon/swear_words.txt', "r") as swear_word_file:
    for line in swear_word_file.readlines():
        swear_words.append(line.strip())

with open('../data/lexicon/negation_words.txt', "r") as negation_word_file:
    for line in negation_word_file.readlines():
        negation_words.append(line.strip())

def preprocess(datafolder):
    feature_list = []
    for submission_json in os.listdir(datafolder):
        file_json_path = os.path.join(datafolder, submission_json)
        with open(file_json_path, "r", encoding='utf-8') as file:
            json_list = json.load(file)
            for json_obj in json_list:
                annotation = CommentAnnotation(json_obj)
                feature_list.append(annotation.create_feature_vector(swear_words, negation_words))
    return feature_list

# vec = [x[3] for x in feature_list]
# karma = vec[:6]
# vec [:6] = (karma - karma.min()) / (karma.max() - karma.min())

def write_preprocessed(preprocessed_data, filename):
    with open(preprocessed_folder + filename, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        csv_writer.writerow(['comment_id', 'sdqc_parent', 'sdqc_submission', 'feature_vector'])
        
        for (id, sdqc_p, sdqc_s, vec) in preprocessed_data:
            csv_writer.writerow([id, sdqc_p, sdqc_s, vec])
        
def main(argv):
    # hpv_data_folder = '../data/annotated/hpv/'
    # data = preprocess(hpv_data_folder)
    # write_preprocessed(data, 'preprocessed.csv')
    text = "Jeg gik mig en tur ude i skoven og der så jeg en fugl"
    tokens = text.split()
    model = load_word_embeddings_bin(word2vec_data, algorithm='word2vec')
    vec = avg_word_emb(tokens, 300, model)
    print(vec)

if __name__ == "__main__":
    main(sys.argv[1:])