import os, json, csv
import numpy as np
from gensim.models.fasttext import FastText

from classes.CommentAnnotation import CommentAnnotation

datafolder = '../data/'
fasttext_data = os.path.join(datafolder, 'fasttext/cc.da.300.bin')

def load_fasttext(filename):
    return FastText.load_fasttext_format(filename, encoding='utf8')

swear_words = []
negation_words = []

with open('../data/lexicon/swear_words.txt', "r") as swear_word_file:
    for line in swear_word_file.readlines():
        swear_words.append(line.strip())

with open('../data/lexicon/negation_words.txt', "r") as negation_word_file:
    for line in negation_word_file.readlines():
        negation_words.append(line.strip())

hpv_data_folder = '../data/annotated/hpv/'
preprocessed_folder = '../data/preprocessed/'
feature_list = list()

for submission_json in os.listdir(hpv_data_folder):
    file_json_path = os.path.join(hpv_data_folder, submission_json)
    with open(file_json_path, "r", encoding='utf-8') as file:
        json_list = json.load(file)
        for json_obj in json_list:
            annotation = CommentAnnotation(json_obj)
            feature_list.append(annotation.create_feature_vector(swear_words, negation_words))

# vec = [x[3] for x in feature_list]
# karma = vec[:6]
# vec [:6] = (karma - karma.min()) / (karma.max() - karma.min())

with open(preprocessed_folder + "preprocessed.csv", "w+", newline='') as out_file:
    csv_writer = csv.writer(out_file, delimiter='\t')
    csv_writer.writerow(['comment_id', 'sdqc_parent', 'sdqc_submission', 'feature_vector'])
    
    for (id, sdqc_p, sdqc_s, vec) in feature_list:
        csv_writer.writerow([id, sdqc_p, sdqc_s, vec])
        
