import os, json, csv, sys
import numpy as np
import getopt
import word_embeddings
import feature_extractor
from classes.CommentAnnotation import CommentAnnotation

datafolder = '../data/'
preprocessed_folder = os.path.join(datafolder, 'preprocessed/')
fasttext_data = os.path.join(datafolder, 'fasttext/cc.da.300.bin')
word2vec_data = os.path.join(datafolder, 'word2vec/dsl_sentences_200_cbow_softmax.kv')

swear_words = []
negation_words = []

with open('../data/lexicon/swear_words.txt', "r") as swear_word_file:
    for line in swear_word_file.readlines():
        swear_words.append(line.strip())

with open('../data/lexicon/negation_words.txt', "r") as negation_word_file:
    for line in negation_word_file.readlines():
        negation_words.append(line.strip())

def loadAnnotations(datafolder):
    if not datafolder:
        return
    annotations = []
    for submission_json in os.listdir(datafolder):
        file_json_path = os.path.join(datafolder, submission_json)
        with open(file_json_path, "r", encoding='utf-8') as file:
            json_list = json.load(file)
            for json_obj in json_list:
                annotation = CommentAnnotation(json_obj)
                annotations.append(annotation)
    return annotations

def preprocess(annotations, wembs=None, emb_dim=100):
    if not annotations:
        return
    feature_list = []
    for annotation in annotations:
        wembs_ = word_embeddings.avg_word_emb(annotation.tokens, emb_dim, wembs) if wembs else []
        feature_list.append(feature_extractor.create_feature_vector(annotation, swear_words, negation_words, wembs_))
    return feature_list

def write_preprocessed(preprocessed_data, filename):
    if not preprocessed_data:
        return
    with open(preprocessed_folder + filename, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        csv_writer.writerow(['comment_id', 'sdqc_parent', 'sdqc_submission', 'feature_vector'])
        
        for (id, sdqc_p, sdqc_s, vec) in preprocessed_data:
            csv_writer.writerow([id, sdqc_p, sdqc_s, vec])
        

def main(argv):
    # data_folder = wembs = emb_dim = None
    # try:
    #     opts, _ = getopt.getopt(argv, "l:d:e:", ["loadw2v=","data=","emb_dim=","help"])
    # except getopt.GetoptError:
    #     print("see: preprocess.py -help")
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt in ('-l', '-loadw2v'):
    #         wembs = word_embeddings.load_saved_word2vec_wv(arg)
    #     elif opt in ('-d', '-data'):
    #         data_folder = arg
    #     elif opt in ('-e', '-emb_dim'):
    #         emb_dim = arg

    wembs = word_embeddings.load_saved_word2vec_wv(word2vec_data)
    data = preprocess("../data/annotated/peter-madsen", wembs, emb_dim=200)
    write_preprocessed(data, 'preprocessed.csv')

if __name__ == "__main__":
    main(sys.argv[1:])