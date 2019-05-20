# external imports
import argparse
import numpy as np
import sys
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from joblib import dump

# our imports
import data_loader

rand = np.random.RandomState(42)

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
    parser.add_argument('-x', '--train_file', dest='train_file',
                        default='../data/preprocessed/PP_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_fasttext_train.csv',
                        help='Input file holding train data')
    parser.add_argument('-o', '--out_path', help='Output path for storing the joblib model file')
    parser.add_argument('-m', '--model', help='Model to train. Can choose \'svm\' and \'logit\'', default='svm')
    args = parser.parse_args(argv)

    if args.train_file and args.out_path:
        clf = None
        if args.model == "svm":
            clf = LinearSVC(penalty='l2', C=10, class_weight=None, dual=True, max_iter=50000, random_state=rand)
        elif args.model == "logit":
            clf = LogisticRegression(solver='liblinear', multi_class='auto', dual=True,
                                     penalty='l2', C=1, class_weight='balanced', max_iter=50000)
        
        X, y, n_features, feature_mapping = data_loader.load_train_test_data(train_file=args.train_file, test_file=args.test_file, split=False)

        clf = clf.fit(X, y)
        dump(clf, args.out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
