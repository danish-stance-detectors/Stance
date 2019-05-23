# external imports
import argparse
import os
import sys
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from joblib import dump
import numpy as np

# our imports
import data_loader

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
    parser.add_argument('-x', '--train_file', dest='train_file', help='Input file holding train data')
    parser.add_argument('-y', '--test_file', help='Input file containing more data. Is concatenated with training data for model.')
    parser.add_argument('-o', '--out_path', help='Output path for storing the joblib model file')
    parser.add_argument('-m', '--model', help='Model to train. Can choose \'svm\' and \'logit\'', default='svm')
    args = parser.parse_args(argv)

    if args.train_file and args.test_file and args.out_path:
        clf = None
        if args.model == "svm":
            clf = LinearSVC(C=10, class_weight=None, dual=True)
        elif args.model == "logit":
            clf = LogisticRegression(C=1, class_weight="balanced", dual=True)
        
        X, y, n_features, feature_mapping = data_loader.load_train_test_data(train_file=args.train_file, test_file=args.test_file, split=False)
        
        X = [flatten(x) for x in X]

        clf = clf.fit(X, y)
        dump(clf, args.out_path)

def flatten(list_of_lists):
    flat_list = []
    for sub_item in list_of_lists:
        if type(sub_item) == list:
            flat_list.extend(sub_item)
        else:
            flat_list.append(sub_item)
    return flat_list

if __name__ == "__main__":
    main(sys.argv[1:])