import os, json, csv, sys, re
import word_embeddings
from classes.Annotation import RedditDataset
from classes.Features import FeatureExtractor
from nltk import word_tokenize, sent_tokenize
import argparse
import datetime
import time
from data_loader import load_bow_list

datafolder = '../data/'
hmm_folder = os.path.join(datafolder, 'hmm/')
preprocessed_folder = os.path.join(datafolder, 'preprocessed/')
annotated_folder = os.path.join(datafolder, 'twitter_test_data/')

punctuation = re.compile('[^a-zA-ZæøåÆØÅ0-9]')

def preprocess(filename, sub_sample, super_sample):
    if not filename:
        return
    dataset = RedditDataset()
    s = 'Loading and preprocessing data '
    if sub_sample:
        s += 'with sub sampling'
    if super_sample:
        if sub_sample:
            s += ' and '
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
                    dataset.add_submission_branch(branch, sub_sample=sub_sample)
    print('Done\n')
    dataset.print_status_report()
    print()
    print('Making train test split')
    train, test = dataset.train_test_split()
    print('Done\n')
    if super_sample:
        print('Super sampling...')
        train = dataset.super_sample(train, pct_words_to_replace=super_sample)
        print('Done\n')
        print('Dataset after super sampling:')
        print('Total:')
        dataset.print_status_report()
        print('Train:')
        dataset.print_status_report(train)
        print('Test:')
        dataset.print_status_report(test)
        print()
    return dataset, train, test


def create_features(feature_extractor, data,  text, lexicon, sentiment, reddit,
                                        most_freq, bow, pos, wembs):
    if not feature_extractor or not data:
        return
    print('Extracting and creating feature vectors')
    data = feature_extractor.create_feature_vectors(data, text, lexicon, sentiment, reddit, most_freq, bow, pos, wembs)
    print('Done')
    return data


def write_preprocessed(header_features, preprocessed_data, filename):
    if not preprocessed_data:
        return
    out_path = os.path.join(preprocessed_folder, filename)
    print('Writing feature vectors to', out_path)
    with open(out_path, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        header = ['comment_id', 'sdqc_parent', 'sdqc_submission']
        header.extend(header_features)
        csv_writer.writerow(header)
        
        for (id, sdqc_p, sdqc_s, vec) in preprocessed_data:
            csv_writer.writerow([id, sdqc_p, sdqc_s, *vec])
    print('Done')

def write_reddit_corupus(annotations, filename='../data/corpus/reddit_sentences.txt'):
    with open(filename, 'w+', encoding='utf-8') as outfile:
        for i, annotation in enumerate(annotations):
            sentences = sent_tokenize(annotation.text.lower(), language='danish')
            sentence_tokens = [word_tokenize(t, language='danish') for t in sentences]

            for tokens in sentence_tokens:
                tokens_clean = []
                for token in tokens:
                    if token and not punctuation.match(token):
                        token = re.sub(r'\.\.$', '', token)
                        tokens_clean.append(token)
                if not tokens_clean:
                    continue
                # only write out tokens after checking list, otherwise whitespace appear
                for t in tokens_clean:
                    outfile.write(t + ' ')
                outfile.write('\n')

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
    parser.add_argument('-sub', '--sub_sample', dest='sub', default=False, action='store_true',
                        help='Sub sample by removing pure comment branches')
    parser.add_argument('-sup', '--super_sample', dest='sup', nargs='?', type=float, const=0.5,
                        help='Super sample by duplicating modified SDQ comments')
    parser.add_argument('-t', '--text', dest='text', default=False, action='store_true', help='Enable text features')
    parser.add_argument('-l', '--lexicon', dest='lexicon', default=False, action='store_true',
                        help='Enable lexicon features')
    parser.add_argument('-s', '--sentiment', dest='sentiment', default=False, action='store_true',
                        help='Enable sentiment features')
    parser.add_argument('-r', '--reddit', dest='reddit', default=False, action='store_true',
                        help='Enable Reddit features')
    parser.add_argument('-mf', '--most_frequent', nargs='?', type=int, const=100,
                        help='Enable most frequent words per class features')
    parser.add_argument('-b', '--bow', default=False, dest='bow', action='store_true', help='Enable BOW features')
    parser.add_argument('-p', '--pos', default=False, dest='pos', action='store_true', help='Enable POS features')
    parser.add_argument('-w2v', '--word2vec', nargs='?', type=int, const=300,
                        help='Enable word2vec word embeddings and specify vector size (default: 300)')
    parser.add_argument('-ft', '--fasttext', default=False, action='store_true',
                        help='Enable fastText word embeddings with default vector size 300')
    parser.add_argument('-c', '--corpus', default=False, dest='corpus', action='store_true',
                        help='Write a corpus file for Reddit data')
    parser.add_argument('-concat', '--concat', default=False, action='store_true', help='Concatenate train and test to a single output file')
    parser.add_argument('-bow_dict', '--bow_dict', default=False, action='store_true', help='Store file with bow words.')
    parser.add_argument('-o', '--outname', help='output name prefix for preprocessed file')
    parser.add_argument('-path', '--path', help='path to data to preprocess', default='../data/annotated/')
    args = parser.parse_args(argv)

    if args.bow_dict:
        bow_set = load_bow_list(args.path)
        with open(args.outname, 'w+', encoding='utf-8') as file:
            for word in bow_set:
                file.write(word + '\n')
    else:

        outputfile = 'PP'
        if args.outname:
            outputfile = args.outname
        
        features = []
        for arg in vars(args):
            attr = getattr(args, arg)
            if attr:
                if not (arg == 'sup' or arg == 'sub'):
                    features.append(arg)
                outputfile += '_%s' % arg
                if type(attr) is int:
                    outputfile += '%d' % attr
                if type(attr) is float:
                    outputfile += '%d' % int(attr*100)

        word_embeddings.load_saved_word_embeddings(args.word2vec, args.fasttext)

        dataset, train, test = preprocess(args.path, args.sub, args.sup)
        if args.corpus:
            train.extend(test)
            write_reddit_corupus(train)
            return

        feature_extractor = FeatureExtractor(dataset)
        train_features = create_features(feature_extractor, train, args.text, args.lexicon, args.sentiment, args.reddit,
                                        args.most_frequent, args.bow, args.pos, (args.word2vec or args.fasttext))
        test_features = create_features(feature_extractor, test, args.text, args.lexicon, args.sentiment, args.reddit,
                                        args.most_frequent, args.bow, args.pos, (args.word2vec or args.fasttext))
        
        if args.concat:
            train_features.extend(test_features)
            write_preprocessed(features, train_features, outputfile + '_concat.csv')
        else:
            write_preprocessed(features, train_features, outputfile + '_train.csv')
            write_preprocessed(features, test_features, outputfile + '_test.csv')


if __name__ == "__main__":
    main(sys.argv[1:])
