import word_embeddings, preprocess
import os, json, csv, sys
from classes.Annotation import CommentAnnotation
from classes.Annotation import Annotations
from classes.Features import FeatureExtractor

datafolder = '../data/'
preprocessed_folder = os.path.join(datafolder, 'preprocessed/')
annotated_folder = os.path.join(datafolder, 'annotated/')
fasttext_data = os.path.join(datafolder, 'fasttext/cc.da.300.bin')
word2vec_data = os.path.join(datafolder, 'word2vec/dsl_sentences_200_cbow_softmax.kv')

# Dynamic tester of the text and lexicon features
# enter exit to quit the tester
def main(argv):

    wembs = word_embeddings.load_saved_word2vec_wv(word2vec_data)
    swear_words = preprocess.read_lexicon('../data/lexicon/swear_words.txt')
    negation_words = preprocess.read_lexicon('../data/lexicon/negation_words.txt')
    positive_smileys = preprocess.read_lexicon('../data/lexicon/positive_smileys.txt')
    negative_smileys = preprocess.read_lexicon('../data/lexicon/negative_smileys.txt')
    wembs_dim = 200

    vector_names =  {0: "period",
     1: "e_mark",
     2: "q_mark",
     3: "hasTripDot",
     4: "url count",
     5: "quote count",
     6: "capital ratio",
     7: "text length",
     8: "tokens length",
     9: "avg. word length",
     10: "Cap letter max sequence length"
     11: "swear word count",
     12: "negation count",
     13: "positive smiley count",
     14: "negative smiley count" }

    extractor = FeatureExtractor([], swear_words, negation_words, negative_smileys, positive_smileys, wembs, wembs_dim, test=True)
    text_input = ""
    
    while text_input != "exit":
        text_input = input("Insert test text:\n")
        if text_input != "" and text_input != "exit":
            annotation = CommentAnnotation(text_input, test=True)
            feature_vector = extractor.create_feature_vector(annotation, include_reddit_features=False)[3]
            for x in range(len(feature_vector)-wembs_dim):
                print("%s : %0.02f\n" % (vector_names[x], feature_vector[x]))
            print("\n")

if __name__ == "__main__":
    main(sys.argv[1:])