import os, json, csv, sys, re
from src import word_embeddings
from src.classes.Annotation import RedditDataset
from src.classes.Features import FeatureExtractor
from src.data_loader import load_bow_list
from nltk import word_tokenize, sent_tokenize
import argparse

datafolder = '../data/'
hmm_folder = os.path.join(datafolder, 'hmm/')
preprocessed_folder = os.path.join(datafolder, 'preprocessed/')
annotated_folder = os.path.join(datafolder, 'twitter_test_data/')

punctuation = re.compile('[^a-zA-ZæøåÆØÅ0-9]')

truth_to_id = {
    'False': 0,
    'True': 1,
    'Unverified': 2
}
sub_to_truth = {}


def preprocess(filename, sub_sample, write_rumours=False):
    if not filename:
        return
    dataset = RedditDataset()
    s = 'Loading and preprocessing data '
    if sub_sample:
        s += 'with sub sampling'
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
                if sub['IsRumour'] and (not sub['IsIrrelevant']):
                    truth = sub["TruthStatus"]
                    sub_id = submission_json.replace('.json', '')
                    sub_to_truth[sub_id] = truth_to_id[truth]
                    if write_rumours:
                        with open('rumour_branches.txt', 'a+', encoding='utf8') as outfile:
                            outfile.write(sub_id + '\n\n')
                            for branch in branches:
                                for anno in branch:
                                    c_id = anno['comment']['comment_id']
                                    outfile.write(c_id + '\n')
                                outfile.write('\n')
                            outfile.write('\n')
                for i, branch in enumerate(branches):
                    dataset.add_submission_branch(branch, sub_sample=sub_sample)
    print('Done\n')
    dataset.print_status_report()
    print()
    return dataset


def super_sample(dataset, sup):
    print('Super sampling...')
    print('Making train test split')
    train, test = dataset.train_test_split()
    print('Done\n')
    print('Super sampling train...')
    train = dataset.super_sample(train, pct_words_to_replace=sup)
    print('Done\n')
    print('Dataset after super sampling:')
    print('Total:')
    dataset.print_status_report()
    print('Train:')
    dataset.print_status_report(train)
    print('Test:')
    dataset.print_status_report(test)
    print()
    return train, test


def create_features(feature_extractor, data,  text, lexicon, sentiment, reddit,
                                        most_freq, bow, pos, wembs):
    if not feature_extractor or not data:
        return
    print('Extracting and creating feature vectors')
    data = feature_extractor.create_feature_vectors(data, text, lexicon, sentiment, reddit, most_freq, bow, pos, wembs)
    print('Done')
    return data


def write_preprocessed(header_features, preprocessed_data, filename, write_rumours_sep=False):
    if not preprocessed_data:
        return
    out_path = os.path.join(preprocessed_folder, filename)
    print('Writing feature vectors to', out_path)
    with open(out_path + '.csv', "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        header = ['sub_id', 'sdqc_parent', 'sdqc_submission']
        header.extend(header_features)
        csv_writer.writerow(header)

        if write_rumours_sep:
            with open(out_path + '_rumours.csv', 'w+', newline='') as rumour_file:
                csv_writer_r = csv.writer(rumour_file, delimiter='\t')
                header_r = ['sub_id', 'comment_id', 'time', 'truth', 'sdqc_submission']
                header_r.extend(header_features)
                csv_writer_r.writerow(header_r)
        
        for (sub_id, id, t, sdqc_p, sdqc_s, vec) in preprocessed_data:
            if write_rumours_sep and sub_id in sub_to_truth:
                truth = sub_to_truth[sub_id]
                with open(out_path + '_rumours.csv', 'a', newline='') as rumour_file:
                    csv_writer_r = csv.writer(rumour_file, delimiter='\t')
                    csv_writer_r.writerow([sub_id, id, t, truth, sdqc_s, *vec])
            else:
                csv_writer.writerow([id, sdqc_p, sdqc_s, *vec])
    print('Done')


def write_reddit_corpus(annotations, filename='../data/corpus/reddit_sentences.txt'):
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

def clear_usernames(path, outpath):
    print("Cleaning user names and annotator names from annotated data.. \n")
    for rumour_folder in os.listdir(filename):
        rumour_folder_path = os.path.join(filename, rumour_folder)
        if not os.path.isdir(rumour_folder_path):
            continue
        print("Cleaning event: ", rumour_folder)
        for submission_json in os.listdir(rumour_folder_path):
            submission_json_path = os.path.join(rumour_folder_path, submission_json)
            with open(submission_json_path, "r", encoding='utf-8') as file:
                print("Cleaning submission: ", submission_json)
                json_obj = json.load(file)
                sub = json_obj['redditSubmission']
                del sub['user']['username']
                
                for branch in sub['branches']:
                    for comment in branch:
                        del comment['annotator']
                        del comment['comment']['user']['username']
                    
                with open(outpath + '/' + rumour_folder + '/' + submission_json, "w+") as clean_file:
                    json.dump(json_obj)

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
    parser.add_argument('-bow_dict', '--bow_dict', default=False, action='store_true', help='Store file with bow words.')
    parser.add_argument('-o', '--outname', help='output name prefix for preprocessed file')
    parser.add_argument('-path', '--path', help='path to data to preprocess', default='../data/annotated/')
    parser.add_argument('-clean', '--clean', default=False, dest='clean', action='store_true', help='Clean usernames and annotator names from annotated data')
    args = parser.parse_args(argv)

    if args.clean:
        if not args.outname:
            print("Please provide an outpath pointing to an empty directory.")
        else:
            clear_usernames(args.path, args.outname)

    if args.bow_dict:
        bow_set = load_bow_list(args.path)
        with open(args.outname, 'w+', encoding='utf-8') as file:
            for word in bow_set:
                file.write(word + '\n')
        sys.exit(0)

    outputfile = 'PP'
    if args.outname:
        outputfile = args.outname

    features = []
    for arg in vars(args):
        attr = getattr(args, arg)
        if attr and not arg == 'path':
            if not (arg == 'sup' or arg == 'sub'):
                features.append(arg)
            outputfile += '_%s' % arg
            if type(attr) is int:
                outputfile += '%d' % attr
            if type(attr) is float:
                outputfile += '%d' % int(attr*100)

    word_embeddings.load_saved_word_embeddings(args.word2vec, args.fasttext)

    dataset = preprocess(args.path, args.sub)
    if args.corpus:
        write_reddit_corpus(dataset.iterate_annotations())
        return

    feature_extractor = FeatureExtractor(dataset)

    if args.sup:
        train, test = super_sample(dataset, args.sup)
        train_features = create_features(feature_extractor, train, args.text, args.lexicon, args.sentiment, args.reddit,
                                        args.most_frequent, args.bow, args.pos, (args.word2vec or args.fasttext))
        test_features = create_features(feature_extractor, test, args.text, args.lexicon, args.sentiment, args.reddit,
                                        args.most_frequent, args.bow, args.pos, (args.word2vec or args.fasttext))
        write_preprocessed(features, train_features, outputfile + '_train')
        write_preprocessed(features, test_features, outputfile + '_test')
    else:
        feature_vectors = create_features(feature_extractor, dataset.iterate_annotations(), args.text, args.lexicon,
                                          args.sentiment, args.reddit, args.most_frequent, args.bow, args.pos,
                                          (args.word2vec or args.fasttext))
        write_preprocessed(features, feature_vectors, outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
