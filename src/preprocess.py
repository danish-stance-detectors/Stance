import os, json, csv, sys
import word_embeddings
from classes.Annotation import RedditDataset
from classes.Features import FeatureExtractor
import argparse

datafolder = '../data/'
preprocessed_folder = os.path.join(datafolder, 'preprocessed/')
annotated_folder = os.path.join(datafolder, 'annotated/')


def preprocess(filename, sub_sample, super_sample):
    if not filename:
        return
    dataset = RedditDataset()
    s = 'Loading and preprocessing data '
    if sub_sample:
        s += 'with sub sampling'
    elif super_sample:
        s += 'with super sampling'
    print(s)
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
                    dataset.add_submission_branch(branch, sub_sample=sub_sample, super_sample=super_sample)
    print('Done')
    print('Number of data points:', dataset.size())
    dataset.print_status_report()
    return dataset


def create_features(dataset, wembs, text, lexicon, sentiment, reddit, most_freq, bow, pos):
    if not dataset:
        return
    print('Extracting and creating feature vectors')
    feature_extractor = FeatureExtractor(dataset)
    data = feature_extractor.create_feature_vectors(wembs, text, lexicon, sentiment, reddit, most_freq, bow, pos)
    print('Done')
    return data


def write_preprocessed(preprocessed_data, filename):
    if not preprocessed_data:
        return
    out_path = os.path.join(preprocessed_folder, filename)
    print('Writing feature vectors to', out_path)
    with open(out_path, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        csv_writer.writerow(['comment_id', 'sdqc_parent', 'sdqc_submission', 'feature_vector'])
        
        for (id, sdqc_p, sdqc_s, vec) in preprocessed_data:
            csv_writer.writerow([id, sdqc_p, sdqc_s, vec])
    print('Done')


def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
    parser.add_argument('-w2v', '--word2vec', dest='w2v', nargs='?', type=int, const=300,
                        help='Enable word2vec word embeddings and specify vector size')
    parser.add_argument('-ft', '--fasttext', dest='fasttext', default=False, action='store_true')
    parser.add_argument('-sub', '--sub_sample', dest='sub', default=False, action='store_true',
                        help='Sub sample by removing pure comment branches')
    parser.add_argument('-sup', '--super_sample', dest='sup', nargs='?', type=int, const=5,
                        help='Super sample by duplicating modified SDQ comments')
    parser.add_argument('-t', '--text', dest='text', default=False, action='store_true', help='Enable text features')
    parser.add_argument('-l', '--lexicon', dest='lexicon', default=False, action='store_true',
                        help='Enable lexicon features')
    parser.add_argument('-s', '--sentiment', dest='sentiment', default=False, action='store_true',
                        help='Enable sentiment features')
    parser.add_argument('-r', '--reddit', dest='reddit', default=False, action='store_true',
                        help='Enable Reddit features')
    parser.add_argument('-mf', '--most_frequent', dest='freq', nargs='?', type=int, const=50,
                        help='Enable most frequent words per class features')
    parser.add_argument('-b', '--bow', default=False, dest='bow', action='store_true', help='Enable BOW features')
    parser.add_argument('-p', '--pos', default=False, dest='pos', action='store_true', help='Enable POS features')
    args = parser.parse_args(argv)

    outputfile = 'preprocessed'
    for arg in vars(args):
        attr = getattr(args, arg)
        if attr:
            outputfile += '_%s' % arg
            if type(attr) is int:
                outputfile += '%d' % attr
    outputfile += '.csv'

    word_embeddings.load_saved_word_embeddings(args.w2v, args.fasttext)
    dataset = preprocess(annotated_folder, args.sub, args.sup)

    data = create_features(dataset, (args.w2v or args.fasttext),
                           args.text, args.lexicon, args.sentiment, args.reddit, args.freq, args.bow, args.pos)
    write_preprocessed(data, outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
