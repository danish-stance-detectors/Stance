from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import argparse, os
import data_loader
from model_stats import plot_confusion_matrix, cm_acc_f1

output_folder = '../output/'

parser = argparse.ArgumentParser(description='Hyper parameter search for stance classification models')
parser.add_argument('-x', '--train_file', dest='train_file', default='../data/preprocessed/preprocessed_train.csv',
                        help='Input file holding train data')
parser.add_argument('-y', '--test_file', dest='test_file', default='../data/preprocessed/preprocessed_test.csv',
                    help='Input file holding test data')
parser.add_argument('-k', '--k_folds', dest='k_folds', default=5, type=int, nargs='?',
                    help='Number of folds for cross validation (default=5)')
args = parser.parse_args()

X_train, X_test, y_train, y_test, _, feature_mapping = data_loader.load_train_test_data(
    train_file=args.train_file, test_file=args.test_file
)
X_train = data_loader.select_features(X_train, feature_mapping, text=True, lexicon=True, pos=True)
X_test = data_loader.select_features(X_test, feature_mapping, text=True, lexicon=True, pos=True)

settings = [
    ('rbf-svm', SVC(), {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}),
    ('linear-svm', SVC(), {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}),
    ('tree', DecisionTreeClassifier(), {'criterion': ['entropy', 'gini'], 'splitter':['best', 'random'],
                                        'max_depth': range(2, 7, 2)}),
    ('logistic-regression', LogisticRegression(), {'solver': ['liblinear'], 'penalty':['l1', 'l2'],
                                                   'C': [1, 10, 100, 1000], 'multi_class': ['auto']}),
    ('random-forest', RandomForestClassifier(), {'n_estimators': [10, 50, 100, 200], 'criterion': ['entropy', 'gini'],
                                                 'max_depth': range(2, 7, 2)})
]

scorer = 'f1_macro'

# scores = [
#     'accuracy',
#     'f1_macro'
# ]
skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

for name, estimator, tuned_parameters in settings:
    for
    filepath = os.path.join(output_folder, name)
    with open('%s_parameters.txt' % filepath, 'w+') as outfile:
        s = "# Tuning hyper-parameters on F1 macro for %s" % name
        print(s)
        outfile.write(s + '\n')
        print()
        clf = GridSearchCV(
            estimator, tuned_parameters, scoring=scorer, n_jobs=-1, error_score=0,
            cv=skf
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
        s = "Grid scores on development set:"
        print(s)
        outfile.write(s + '\n')
        print()
        results = clf.cv_results_
        print(scorer)
        outfile.write(scorer + '\n')
        means = results['mean_test_%s' % scorer]
        stds = results['std_test_%s' % scorer]
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            s = "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)
            print(s)
            outfile.write(s + '\n')
        print()

        outfile.write('Classification report for results on evaluation set:' + '\n')
        print("Classification report for results on evaluation set:")
        y_true, y_pred = y_test, clf.predict(X_test)
        # s = classification_report(y_true, y_pred)
        cm, acc, f1 = cm_acc_f1(y_true, y_pred, ['S', 'D', 'Q', 'C'])
        outfile.write(cm + '\n')
        print('acc: %.5f' % acc)
        outfile.write('acc: %.5f\n' % acc)
        print('f1 macro: %f' % f1)
        outfile.write('f1 macro: %f\n\n' % f1)
        print()
        # print(s)
        # outfile.write(s + '\n')
        # print()
        # np.set_printoptions(precision=2)
        # plot_confusion_matrix(y_test, y_pred, title='Confusion matrix, without normalization',
        #                       save_to_filename='%s_param_cm.png' % filepath)



