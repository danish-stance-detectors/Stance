from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import expon as sp_expon
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import argparse, os, csv, sys
import time
import data_loader
from model_stats import plot_confusion_matrix, cm_acc_f1

output_folder = '../output/'

rand = np.random.RandomState(42)

parser = argparse.ArgumentParser(description='Hyper parameter search for stance classification models')
parser.add_argument('-x', '--train_file', dest='train_file', default='../data/preprocessed/preprocessed_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_train.csv',
                        help='Input file holding train data')
parser.add_argument('-y', '--test_file', dest='test_file', default='../data/preprocessed/preprocessed_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_test.csv',
                    help='Input file holding test data')
parser.add_argument('-k', '--k_folds', dest='k_folds', default=3, type=int, nargs='?',
                    help='Number of folds for cross validation (default=5)')
parser.add_argument('-g', '--grid', default=False, action='store_true',
                    help='Enable GridSearchCV, otherwise use RandomizedSearchCV')
parser.add_argument('-n', '--rand_samples', default=10, type=int, nargs='?',
                    help='Number of random samples if using RandomizedSearchCV')
parser.add_argument('-r', '--reduce_features', action='store_true', default=False,
                    help='Reduce features by Variance Threshold')
args = parser.parse_args()

X_train, X_test, y_train, y_test, _, feature_mapping = data_loader.load_train_test_data(
    train_file=args.train_file, test_file=args.test_file
)

def union_reduce_then_split(x1, x2):
    len1 = len(x1)
    len2 = len(x2)
    x_union = x1
    x_union.extend(x2)
    x_transformed = VarianceThreshold(0.001).fit_transform(x_union)
    # support = x_transformed.get_support(indices=True)
    # x1_transformed = []
    # x2_transformed = []
    # for i in support:
    #     if 0 <= i < len1:
    #         x1_transformed.append(x_transformed[i])
    #     elif len1 <= i < max_len:
    #         x2_transformed.append(x1_transformed[i])
    #     else:
    #         print('Transformation error')
    return x_transformed[:len1], x_transformed[len1:]

settings = [
    ('rbf-svm', SVC(), {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}),
    ('linear-svm', SVC(), {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}),
    ('tree', DecisionTreeClassifier(), {'criterion': ['entropy', 'gini'], 'splitter':['best', 'random'],
                                        'max_depth': range(2, 7, 2)}),
    ('logistic-regression', LogisticRegression(), {'solver': ['liblinear'], 'penalty':['l1', 'l2'],
                                                   'C': [1, 10, 100, 1000], 'multi_class': ['auto']}),
    ('random-forest', RandomForestClassifier(), {'n_estimators': [10, 100, 1000], 'criterion': ['entropy', 'gini'],
                                                 'max_depth': range(2, 7, 2)})
]

settings_rand = [
    ('linear-svm', LinearSVC(random_state=rand), {
        'C': sp_randint(1, 1000), 'class_weight': ['balanced', None],
        'max_iter': [50000], 'dual': [True, False]}),
    ('tree', DecisionTreeClassifier(presort=True, random_state=rand), {
        'criterion': ['entropy', 'gini'], 'splitter':['best', 'random'],
        'max_depth': [3, 10, 50, None], "min_samples_split": sp_randint(2, 11),
        'max_features': ['auto', 'log2', None], 'class_weight': ['balanced', None]}),
    ('logitl1', LogisticRegression(solver='liblinear', multi_class='auto', penalty='l1'), {
        'class_weight': ['balanced', None], 'C': sp_randint(1, 1000)}),
    ('logitl2', LogisticRegression(solver='liblinear', multi_class='auto', penalty='l2'), {
        'dual': [True, False], 'class_weight': ['balanced', None], 'C': sp_randint(1, 1000)}),
    ('random-forest', RandomForestClassifier(n_jobs=-1, random_state=rand), {
        'n_estimators': sp_randint(10, 1000), 'criterion': ['entropy', 'gini'],
        'max_depth': [3, 10, 50, None], 'max_features': ['auto', 'log2', None],
        "min_samples_split": sp_randint(2, 11), "bootstrap": [True, False],
        'class_weight': ['balanced_subsample', None]})
]

scorer = 'f1_macro'
folds = args.k_folds
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand)
features = data_loader.get_features()
feature_names = features.keys()

grid_search = args.grid  # whether to use GridSearchCV or RandomizedSearchCV
rand_iter = args.rand_samples  # number of random samples to use

def parameter_search_rand_VT(X_train, X_test, y_train, y_test):
    print("# Tuning hyper-parameters with random search and variance threshold on F1 macro")
    print('# - %d random samples' % rand_iter)
    print('# - %d folds in RandomSearch Cross Validation' % folds)
    for name, estimator, tuned_parameters in (settings_rand if not grid_search else settings):
        filepath = os.path.join(output_folder, name)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        print("## Running %s" % name)
        stats_filename = '%s/parameter_stats_rand_vt' % filepath
        if not os.path.exists(stats_filename):
            with open('%s.csv' % stats_filename, 'w+', newline='') as statsfile:
                csv_writer = csv.writer(statsfile)
                csv_writer.writerow(['estimator', 'f1_macro', 'acc', 'folds', 'rand_iter', 'parameters', 'features'])
        results_filename = '%s/params_rand_vt_iter%d_k%d' % (filepath, rand_iter, folds)
        if os.path.exists(results_filename):
            print('Skipping since %s exists' % results_filename)
            continue
        start = time.time()
        with open('%s.txt' % results_filename, 'a+') as outfile, \
                open('%s.csv' % stats_filename, 'a', newline='') as statsfile:
            csv_writer = csv.writer(statsfile)
            clf = RandomizedSearchCV(
                estimator, tuned_parameters, scoring=scorer, n_jobs=-1, error_score=0, n_iter=rand_iter,
                cv=skf, iid=False, return_train_score=False, pre_dispatch='2*n_jobs', random_state=rand
            )
            clf.fit(X_train, y_train)

            s = "Best parameters set found on development set for F1 macro:"
            print(s)
            outfile.write(s + '\n')
            print()
            s = "%0.3f for %r" % (clf.best_score_, clf.best_params_)
            print(s)
            outfile.write(s + '\n')
            print()
            s = "Randomized scores on development set:"
            print(s)
            outfile.write(s + '\n')
            print()
            results = clf.cv_results_
            means = results['mean_test_score']
            stds = results['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                s = "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)
                print(s)
                outfile.write(s + '\n')
            print()

            outfile.write('Classification report for results on evaluation set:' + '\n')
            print("Classification report for results on evaluation set:")
            y_true, y_pred = y_test, clf.predict(X_test)
            outfile.write(classification_report(y_true, y_pred))
            outfile.write('\n')
            cm, acc, f1 = cm_acc_f1(y_true, y_pred)
            outfile.write(np.array2string(cm))
            outfile.write('\n')
            print('acc: %.4f' % acc)
            outfile.write('acc: %.4f\n' % acc)
            print('f1 macro: %.4f' % f1)
            outfile.write('f1 macro: %.4f\n\n' % f1)
            print()
            csv_writer.writerow([name, '%.4f' % f1, '%.4f' % acc, folds, rand_iter, clf.best_params_, features])
        end = time.time()
        print('Took %.1f seconds' % (end - start))
        print('Done with', name)


X_train_ = data_loader.select_features(X_train, feature_mapping, features)
X_test_ = data_loader.select_features(X_test, feature_mapping, features)
old_len = len(X_train_[0])
X_train_, X_test_ = union_reduce_then_split(X_train_, X_test_)
new_len = len(X_train_[0])
print('Reduced features from %d to %d' % (old_len, new_len))
parameter_search_rand_VT(X_train_, X_test_, y_train, y_test)


def parameter_search_LOO_features():
    for name, estimator, tuned_parameters in (settings_rand if not grid_search else settings):
        filepath = os.path.join(output_folder, name)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        print("# Tuning hyper-parameters on F1 macro for %s" % name)
        stats_filename = '%s/parameter_stats_iter%d_k%d' % (filepath, rand_iter, folds)
        if args.reduce_features:
            stats_filename += '_vt'
        if not os.path.exists(stats_filename):
            with open('%s.csv' % stats_filename, 'w+', newline='') as statsfile:
                csv_writer = csv.writer(statsfile)
                csv_writer.writerow(['estimator', 'f1_macro', 'acc', 'LOO feature', 'parameters', 'features'])
        for feature_name in feature_names:
            results_filename = '%s/params_%s_iter%d_k%d' % (filepath, feature_name, rand_iter, folds)
            if not features[feature_name] or os.path.exists(results_filename):
                print('Skipping %s since %s exists' % (feature_name, results_filename))
                continue
            if feature_name == 'all':
                print('Running with all features enabled')
            else:
                print('Leaving %s features out' % feature_name)
            features[feature_name] = False
            X_train_ = data_loader.select_features(X_train, feature_mapping, features)
            X_test_ = data_loader.select_features(X_test, feature_mapping, features)
            if args.reduce_features:
                old_len = len(X_train_[0])
                X_train_, X_test_ = union_reduce_then_split(X_train_, X_test_)
                new_len = len(X_train_[0])
                print('Reduced features from %d to %d' % (old_len, new_len))
                results_filename += '_vt%d' % old_len
            start = time.time()
            with open('%s.txt' % results_filename, 'a+') as outfile, \
                    open('%s.csv' % stats_filename, 'a', newline='') as statsfile:
                csv_writer = csv.writer(statsfile)
                if not grid_search:
                    clf = RandomizedSearchCV(
                        estimator, tuned_parameters, scoring=scorer, n_jobs=-1, error_score=0, n_iter=rand_iter,
                        cv=skf, iid=False, return_train_score=False, pre_dispatch=None
                    )
                else:
                    clf = GridSearchCV(
                        estimator, tuned_parameters, scoring=scorer, n_jobs=-1, error_score=0,
                        cv=skf, iid=False, return_train_score=False, pre_dispatch=None
                    )
                clf.fit(X_train_, y_train)

                s = "Best parameters set found on development set for F1 macro:"
                print(s)
                outfile.write(s + '\n')
                print()
                s = "%0.3f for %r" % (clf.best_score_, clf.best_params_)
                print(s)
                outfile.write(s + '\n')
                print()
                s = "Grid scores on development set:"
                print(s)
                outfile.write(s + '\n')
                print()
                results = clf.cv_results_
                means = results['mean_test_score']
                stds = results['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    s = "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)
                    print(s)
                    outfile.write(s + '\n')
                print()

                outfile.write('Classification report for results on evaluation set:' + '\n')
                print("Classification report for results on evaluation set:")
                y_true, y_pred = y_test, clf.predict(X_test_)
                outfile.write(classification_report(y_true, y_pred))
                outfile.write('\n')
                cm, acc, f1 = cm_acc_f1(y_true, y_pred)
                outfile.write(np.array2string(cm))
                outfile.write('\n')
                print('acc: %.4f' % acc)
                outfile.write('acc: %.4f\n' % acc)
                print('f1 macro: %.4f' % f1)
                outfile.write('f1 macro: %.4f\n\n' % f1)
                print()
                csv_writer.writerow([name, '%.4f' % f1, '%.4f' % acc, feature_name, clf.best_params_, features])
            if not feature_name == 'all':
                features[feature_name] = True
            end = time.time()
            print('Done with %s features' % feature_name)
            print('Took %.1f seconds' % (end - start))
        print('Done with', name)

