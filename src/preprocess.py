import os, json, csv, sys
import word_embeddings
from classes.Annotation import RedditDataset
from classes.Features import FeatureExtractor
import argparse

datafolder = '../data/'
preprocessed_folder = os.path.join(datafolder, 'preprocessed/')
annotated_folder = os.path.join(datafolder, 'annotated/')
fasttext_data = os.path.join(datafolder, 'fasttext/cc.da.300.bin')
word2vec_data = lambda dim: os.path.join(datafolder, 'word2vec/dsl_sentences_{0}_cbow_softmax.kv'.format(dim))

# Loads lexicon file given path
# Assumes file has one word per line
def read_lexicon(file_path):
    with open(file_path, "r") as lexicon_file:
        return [line.strip().lower() for line in lexicon_file.readlines()]

swear_words = []
negation_words = []
positive_smileys = read_lexicon('../data/lexicon/positive_smileys.txt')
negative_smileys = read_lexicon('../data/lexicon/negative_smileys.txt')

with open('../data/lexicon/swear_words.txt', "r") as swear_word_file:
    for line in swear_word_file.readlines():
        swear_words.append(line.strip().lower())

with open('../data/lexicon/negation_words.txt', "r") as negation_word_file:
    for line in negation_word_file.readlines():
        negation_words.append(line.strip().lower())


def loadAnnotations(filename, sub_sample, super_sample):
    if not filename:
        return
    dataset = RedditDataset()
    for rumour_folder in os.listdir(filename):
        rumour_folder_path = os.path.join(filename, rumour_folder)
        if not os.path.isdir(rumour_folder_path):
            continue
        print("Preprocessing event: ", rumour_folder)
        for submission_json in os.listdir(rumour_folder_path):
            submission_json_path = os.path.join(rumour_folder_path, submission_json)
            with open(submission_json_path, "r", encoding='utf-8') as file:
                print("Preprocessing submission: ", submission_json)
                json_obj = json.load(file)
                sub = json_obj['redditSubmission']
                dataset.add_reddit_submission(sub)
                branches = json_obj['branches']
                for i, branch in enumerate(branches):
                    print("Adding branch", i)
                    dataset.add_submission_branch(branch, sub_sample=sub_sample, super_sample=super_sample)
    print(dataset.size())
    dataset.print_status_report()
    return dataset


def preprocess(dataset, emb_dim, text, lexicon, sentiment, reddit, most_freq, bow, pos):
    if not dataset:
        return
    feature_extractor = \
        FeatureExtractor(dataset, swear_words, negation_words,
                         negative_smileys, positive_smileys, emb_dim, wv_model=True)
    data = feature_extractor.create_feature_vectors(text, lexicon, sentiment, reddit, most_freq, bow, pos)
    return data


def write_preprocessed(preprocessed_data, filename):
    if not preprocessed_data:
        return
    with open(preprocessed_folder + filename, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        csv_writer.writerow(['comment_id', 'sdqc_parent', 'sdqc_submission', 'feature_vector'])
        
        for (id, sdqc_p, sdqc_s, vec) in preprocessed_data:
            csv_writer.writerow([id, sdqc_p, sdqc_s, vec])

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
    # parser.add_argument('-o', '--output', type=str, nargs='*', default='preprocessed.csv',
    #                     help='Output file of the feature vectors')
    parser.add_argument('-v', '--vector_size', dest='dim', type=int, default=300, help='the size of a word vector')
    parser.add_argument('-sub', '--sub_sample', dest='sub', default=False, action='store_true',
                        help='Sub sample by removing pure comment branches')
    parser.add_argument('-sup', '--super_sample', dest='sup', default=False, action='store_true',
                        help='Super sample by duplicating modified SDQ comments')
    parser.add_argument('-t', '--text', dest='text', default=False, action='store_true', help='Enable text features')
    parser.add_argument('-l', '--lexicon', dest='lexicon', default=False, action='store_true',
                        help='Enable lexicon features')
    parser.add_argument('-s', '--sentiment', dest='sentiment', default=False, action='store_true',
                        help='Enable sentiment features')
    parser.add_argument('-r', '--reddit', dest='reddit', default=False, action='store_true',
                        help='Enable Reddit features')
    parser.add_argument('-freq', '--most_frequent', dest='freq', default=False, action='store_true',
                        help='Enable most frequent words per class features')
    parser.add_argument('-b', '--bow', default=False, dest='bow', action='store_true', help='Enable BOW features')
    parser.add_argument('-p', '--pos', default=False, dest='pos', action='store_true', help='Enable POS features')
    args = parser.parse_args(argv)

    outputfile = 'preprocessed_dim%d' % args.dim
    for arg in vars(args):
        if arg == 'dim':
            continue
        if getattr(args, arg):
            outputfile += '_%s' % arg
    outputfile += '.csv'

    word_embeddings.load_saved_word2vec_wv(word2vec_data(args.dim))
    dataset = loadAnnotations(annotated_folder, args.sub, args.sup)
    # lexicon, sentiment, reddit, most_freq, bow, pos

    data = preprocess(dataset,
                      args.dim, args.text, args.lexicon, args.sentiment, args.reddit, args.freq, args.bow, args.pos)
    write_preprocessed(data, outputfile)

if __name__ == "__main__":
    main(sys.argv[1:])