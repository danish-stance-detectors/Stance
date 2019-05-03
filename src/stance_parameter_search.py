from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import expon as sp_expon
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import argparse, os, csv
import time
import data_loader
from model_stats import plot_confusion_matrix, cm_acc_f1

output_folder = '../output/'

parser = argparse.ArgumentParser(description='Hyper parameter search for stance classification models')
parser.add_argument('-x', '--train_file', dest='train_file', default='../data/preprocessed/preprocessed_train.csv',
                        help='Input file holding train data')
parser.add_argument('-y', '--test_file', dest='test_file', default='../data/preprocessed/preprocessed_test.csv',
                    help='Input file holding test data')
parser.add_argument('-k', '--k_folds', dest='k_folds', default=3, type=int, nargs='?',
                    help='Number of folds for cross validation (default=5)')
parser.add_argument('-g', '--grid', default=False, action='store_true',
                    help='Enable GridSearchCV, otherwise use RandomizedSearchCV')
parser.add_argument('-n', '--rand_samples', default=10, type=int, nargs='?',
                    help='Number of random samples if using RandomizedSearchCV')
args = parser.parse_args()

X_train, X_test, y_train, y_test, _, feature_mapping = data_loader.load_train_test_data(
    train_file=args.train_file, test_file=args.test_file
)

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
    # ('rbf-svm', SVC(), {'kernel': ['rbf'], 'gamma': sp_expon(scale=.1), 'C': sp_randint(1, 1000),
    #                     'class_weight': ['balanced', None]}),
    # ('linear-svm', LinearSVC(), {'C': sp_randint(1, 1000), 'multi_class': ['crammer_singer', 'ovr'],
    #                              'class_weight': ['balanced', None], 'max_iter': [100000],
    #                              'tol': sp_expon(scale=1e-4)}),
    # ('tree', DecisionTreeClassifier(), {'criterion': ['entropy', 'gini'], 'splitter':['best', 'random'],
    #                                     'max_depth': sp_randint(2, 50), "min_samples_split": sp_randint(2, 11),
    #                                     'max_features': ['auto', 'log2', None], 'class_weight': ['balanced', None],
    #                                     'presort': [True]}),
    ('logistic-regression', LogisticRegression(), {'solver': ['liblinear'], 'penalty':['l1', 'l2'],
                                                   'class_weight': ['balanced', None],
                                                   'C': sp_randint(1, 1000), 'multi_class': ['auto']}),
    # ('random-forest', RandomForestClassifier(), {'n_estimators': sp_randint(10, 2000), 'criterion': ['entropy', 'gini'],
    #                                              'max_depth': sp_randint(2, 50), 'max_features': ['auto', 'log2', None],
    #                                              "min_samples_split": sp_randint(2, 11), "bootstrap": [True, False],
    #                                              'class_weight': ['balanced_subsample', None]})
]

scorer = 'f1_macro'
folds = args.k_folds
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
features = data_loader.get_features()
feature_names = features.keys()

grid_search = args.grid  # whether to use GridSearchCV or RandomizedSearchCV
rand_iter = args.rand_samples  # number of random samples to use

for name, estimator, tuned_parameters in (settings_rand if not grid_search else settings):
    filepath = os.path.join(output_folder, name)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    print("# Tuning hyper-parameters on F1 macro for %s" % name)
    stats_filename = '%s/parameter_stats_iter%d_k%d.txt' % (filepath, rand_iter, folds)
    if not os.path.exists(stats_filename):
        with open(stats_filename, 'w+', newline='') as statsfile:
            csv_writer = csv.writer(statsfile)
            csv_writer.writerow(['estimator', 'f1_macro', 'acc', 'LOO feature', 'parameters', 'features'])
    for feature_name in feature_names:
        results_filename = '%s/params_%s_iter%d_k%d.txt' % (filepath, feature_name, rand_iter, folds)
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
        start = time.time()
        with open(results_filename, 'a+') as outfile, open(stats_filename, 'a', newline='') as statsfile:
            csv_writer = csv.writer(statsfile)
            if not grid_search:
                clf = RandomizedSearchCV(
                    estimator, tuned_parameters, scoring=scorer, n_jobs=--1, error_score=0, n_iter=rand_iter,
                    cv=skf, iid=False, return_train_score=False, pre_dispatch=None
                )
            else:
                clf = GridSearchCV(
                    estimator, tuned_parameters, scoring=scorer, n_jobs=--1, error_score=0,
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

