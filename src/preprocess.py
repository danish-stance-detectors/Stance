import os, json, csv, sys
import word_embeddings
from classes.Annotation import RedditDataset
from classes.Features import FeatureExtractor
import argparse
import datetime
import time

datafolder = '../data/'
hmm_folder = os.path.join(datafolder, 'hmm/')
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

def read_hmm_data(filename):
    if not filename:
        return
    
    label_data = []
    sdqc_to_int = {'Supporting':0, 'Denying':1, 'Querying':2, 'Commenting':3}
    label_distribution = {'Supporting':0, 'Denying':0, 'Querying':0, 'Commenting':0}
    rumour_count = 0
    truth_count = 0
    for rumour_folder in os.listdir(filename):
        rumour_folder_path = os.path.join(filename, rumour_folder)
        if not os.path.isdir(rumour_folder_path):
            continue
        for submission_json in os.listdir(rumour_folder_path):
            submission_json_path = os.path.join(rumour_folder_path, submission_json)
            with open(submission_json_path, "r", encoding='utf-8') as file:
                json_obj = json.load(file)
                sub = json_obj['redditSubmission']
                if sub['IsRumour'] and not sub['IsIrrelevant']:
                    print("Adding {} as rumour".format(submission_json))
                    rumour_truth = int(sub['TruthStatus'] == 'True')
                    print(rumour_truth)
                    rumour_count += 1
                    truth_count += rumour_truth
                    for branch in json_obj['branches']:
                        branch_labels = []
                        for comment in branch:
                            label = comment['comment']['SDQC_Submission']
                            label_distribution[label] += 1
                            branch_labels.append(sdqc_to_int[label])
                        label_data.append((rumour_truth, branch_labels))
    
    print("Preprocessed {} rumours of which {} were true".format(rumour_count, truth_count))
    print("With sdqc overall distribution: ")
    print(label_distribution)
    return label_data

def read_hmm_data_no_branches(filename):
    if not filename:
        print("Cannot run method read_hmm_data_no_branches without filename parameter")
        return
    
    label_data = []

    sdqc_to_int = {'Supporting':0, 'Denying':1, 'Querying':2, 'Commenting':3}
    label_distribution = {'Supporting':0, 'Denying':0, 'Querying':0, 'Commenting':0}
    rumour_count = 0
    truth_count = 0
    for rumour_folder in os.listdir(filename):
        rumour_folder_path = os.path.join(filename, rumour_folder)
        if not os.path.isdir(rumour_folder_path):
            continue
        for submission_json in os.listdir(rumour_folder_path):
            submission_json_path = os.path.join(rumour_folder_path, submission_json)
            with open(submission_json_path, "r", encoding='utf-8') as file:
                json_obj = json.load(file)
                sub = json_obj['redditSubmission']
                if sub['IsRumour'] and not sub['IsIrrelevant']:
                    print("Adding {} as rumour".format(submission_json))
                    
                    distinct_comments = dict()
                    rumour_truth = int(sub['TruthStatus'] == 'True')
                    rumour_count += 1
                    truth_count += rumour_truth
                    for branch in json_obj['branches']:
                        branch_labels = []
                        for comment in branch:

                            label = comment['comment']['SDQC_Submission']
                            created = comment['comment']['created']
                            comment_id = comment['comment']['comment_id']
                            
                            time_stamp = time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%S"))
                            distinct_comments[comment_id] = (sdqc_to_int[label], time_stamp)

                            label_distribution[label] += 1
                            branch_labels.append(sdqc_to_int[label])
                    
                    # sort them by time
                    comments_by_time = sorted(distinct_comments.values(), key=lambda x: x[1])

                    # discard time stamps for now
                    label_data.append((rumour_truth, [x[0] for x in comments_by_time]))
    
    print("Preprocessed {} rumours of which {} were true".format(rumour_count, truth_count))
    print("With sdqc overall distribution: ")
    print(label_distribution)
    return label_data

def write_hmm_data(filename, data):
    if not data:
        return
    out_path = os.path.join(hmm_folder, filename)
    print('Writing hmm vectors to', out_path)
    with open(out_path, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        csv_writer.writerow(['TruthStatus', 'SDQC_Labels'])
        
        for (truth_status, labels) in data:
            csv_writer.writerow([truth_status, labels])
    print('Done')

def write_reddit_corupus(annotations, filename='../data/corpus/reddit_sentences.txt'):
    with open(filename, 'w+', encoding='utf-8') as outfile:
        for annotation in annotations:
            for token in annotation.tokens:
                outfile.write(token + ' ')
            outfile.write('\n')

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
    parser.add_argument('-sub', '--sub_sample', default=False, action='store_true',
                        help='Sub sample by removing pure comment branches')
    parser.add_argument('-sup', '--super_sample', nargs='?', type=int, const=0.25,
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
    parser.add_argument('-hmm', '--hiddenMarkovModel', default=False, dest='hmm', action='store_true', help='Get HMM features instead of stance preprocessing features')
    parser.add_argument('-br', '--branch', default=False, dest='branch', action='store_true', help='Get hmm features in branches')
    parser.add_argument('-c', '--corpus', default=False, dest='corpus', action='store_true',
                        help='Write a corpus file for Reddit data')
    args = parser.parse_args(argv)

    outputfile = 'preprocessed'
    features = []
    for arg in vars(args):
        attr = getattr(args, arg)
        if attr:
            features.append(arg)
            outputfile += '_%s' % arg
            if type(attr) is int:
                outputfile += '%d' % attr

    if args.hmm:
        labels = []
        if args.branch:
            labels = read_hmm_data(annotated_folder)
        else:
            labels = read_hmm_data_no_branches(annotated_folder)
            outputfile += '_no_branch'

        write_hmm_data(outputfile + '.csv', labels)
    else:
        word_embeddings.load_saved_word_embeddings(args.word2vec, args.fasttext)

        dataset, train, test = preprocess(annotated_folder, args.sub_sample, args.super_sample)
        if args.corpus:
            train.extend(test)
            write_reddit_corupus(train)
            return

        feature_extractor = FeatureExtractor(dataset)
        train_features = create_features(feature_extractor, train, args.text, args.lexicon, args.sentiment, args.reddit,
                                        args.most_frequent, args.bow, args.pos, (args.word2vec or args.fasttext))
        test_features = create_features(feature_extractor, test, args.text, args.lexicon, args.sentiment, args.reddit,
                                        args.most_frequent, args.bow, args.pos, (args.word2vec or args.fasttext))
        write_preprocessed(features, train_features, outputfile + '_train.csv')
        write_preprocessed(features, test_features, outputfile + '_test.csv')


if __name__ == "__main__":
    main(sys.argv[1:])
