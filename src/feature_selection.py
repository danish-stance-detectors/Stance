from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, GenericUnivariateSelect
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import argparse
import data_loader
import matplotlib.pyplot as plt
import sys

rand = random.Random(42)

classifiers = {
    'logistic-regression': LogisticRegression(solver='liblinear'),
    'tree': DecisionTreeClassifier(),
    'rbf-svm': SVC(kernel='rbf'),
    'linear-svm': LinearSVC(),
    'random-forest': RandomForestClassifier(),
    'majority-vote': DummyClassifier(strategy='most_frequent'),
    'stratified-random': DummyClassifier(strategy='stratified')
}

parser = argparse.ArgumentParser(description='Feature selection for stance classification')
parser.add_argument('-x', '--train_file', dest='train_file',
                    default='../data/preprocessed/PP_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_train.csv',
                    help='Input file holding train data for X')
parser.add_argument('-y', '--test_file', dest='test_file',
                    default='../data/preprocessed/PP_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_test.csv',
                    help='Input file holding test data for y')
args = parser.parse_args()

X, y, n_features, feature_mapping = data_loader.load_train_test_data(
    train_file=args.train_file, test_file=args.test_file, split=False
)

config_map = data_loader.get_features()


def eleminate_low_variance_features(threshold, xx):
    selector = VarianceThreshold(threshold)
    X_ = selector.fit_transform(xx)
    n = len(X_[0])
    return n
# ll = eleminate_low_variance_features(0.01, data_loader.select_features(X, feature_mapping, config_map, merge=True))
# print(ll)

def test_variance_threshold(X, plot=True):
    xx = data_loader.select_features(X, feature_mapping, config_map, merge=True)
    print(len(xx[0]))
    thresholds = np.linspace(0, 0.01, 21)
    feature_lengths = []
    for t in thresholds:
        f = eleminate_low_variance_features(t, xx)
        print('t: %.4f\tn:%d' % (t, f))
        feature_lengths.append(f)

    if plot:
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Variance threshold")
        plt.ylabel("Number of features selected")
        plt.plot(thresholds, feature_lengths)
        plt.show()
# test_variance_threshold(X, plot=False)

def check_feature_importance():
    for feature_name in config_map.keys():
        config_map[feature_name] = False
    for feature_name in config_map.keys():
        if feature_name == 'all':
            continue
        config_map[feature_name] = True
        X__ = data_loader.select_features(X, feature_mapping, config_map)
        l = len(X__[0])
        # print(l)
        f = eleminate_low_variance_features(0.001, X__)
        # try:
        #     f = eleminate_low_variance_features(0.0, X__)
        # except ValueError:
        #     f = l
        print('{:20}All:{}\tAfter:{}'.format(feature_name, l, f))
        config_map[feature_name] = False
# check_feature_importance()


def check_feature_selection():
    for feature_name in config_map.keys():
        config_map[feature_name] = False
        X__, shape = data_loader.select_features(X, feature_mapping, config_map, merge=False)
        print('LOO:{:20}{}'.format(feature_name, shape))
        config_map[feature_name] = True


def check_rfecv_feature_selection(x):
    x_ = data_loader.select_features(x, feature_mapping, config_map, merge=True)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for name, clf in classifiers.items():
        rfecv = RFECV(estimator=clf, cv=skf, scoring='f1_macro')
        rfecv.fit(x_, y)
        print("Optimal number of features : %d" % rfecv.n_features_)
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()


def check_univariate_feature_selection(x):
    trans = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=50)
    x = data_loader.select_features(x, feature_mapping, config_map, merge=True)
    len(x[0])
    X_trans = trans.fit_transform(x, y)
    len(X_trans[0])

# X = np.array(X)
# y = np.array(y)
# print(X.shape)
#
# clf = SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=50000))
# clf.fit(X, y)
# X = clf.transform(X)
# print(X.shape)


# clf = Pipeline([
#   ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=50000))),
#   ('classification', RandomForestClassifier(random_state=42))
# ])
# clf.fit(X, y)
#
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')
# print("%-20s%s %0.2f (+/- %0.2f)" % ('RandomForestClassifier', 'f1_macro', scores.mean(), scores.std() * 2))
#
# clf = RandomForestClassifier(random_state=42)
# clf.fit(X, y)
# scores = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')
# print("%-20s%s %0.2f (+/- %0.2f)" % ('RandomForestClassifier', 'f1_macro', scores.mean(), scores.std() * 2))




