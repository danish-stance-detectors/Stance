import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, cross_val_predict
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import data_loader

output_folder = '../output/'

parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
parser.add_argument('-i', '--input_file', dest='file', default='../data/preprocessed/preprocessed.csv',
                    help='Input file holding train data. If a folder, iterates files within.')
parser.add_argument('-k', '--k_folds', dest='k_folds', const=5, type=int, nargs='?',
                    help='Number of folds for cross validation (default=5)')
parser.add_argument('-a', '--accuracy', dest='acc', action='store_true', default=False,
                    help='Enable accuracy metric')
parser.add_argument('-f', '--f1_macro', dest='f1_macro', action='store_true', default=False,
                    help='Enable F1 macro metric')
parser.add_argument('-p', '--predict', dest='predict', action='store_true', default=False,
                    help='Visualize prediction errors')
parser.add_argument('-l', '--learning_curve', dest='learning_curve', action='store_true', default=False,
                    help='Enable plotting of learning curve')
parser.add_argument('-s', '--score', dest='score', action='store_true', default=False,
                    help='Cross-validate scoring')
args = parser.parse_args()


def plot_learning_curve(estimator, title, X, y, scoring='f1_macro', ylim=None, cv=5, n_jobs=-1):
    plt.figure()
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross-validation score")
    plt.legend(loc='best')
    return plt



classifiers = {
    'Logistic Regression': LogisticRegression(solver='liblinear', C=10, penalty='l1', multi_class='auto'),
    'Decision Tree': DecisionTreeClassifier(criterion="entropy", splitter='random'),
    'Linear SVM': SVC(kernel='linear', C=10),
    'Random Forest': RandomForestClassifier(criterion='entropy', n_estimators=10),
    'Majority vote': DummyClassifier(strategy='most_frequent'),
    'Stratified Random': DummyClassifier(strategy='stratified')
}

X, y, _ = data_loader.get_features_and_labels(filename=args.file)
skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)


def cross_val(score, plot=False, predict=False):
    if plot:
        cross_val_plot(score)
    if predict:
        for name, clf in classifiers.items():
            predicted = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)
            fig, ax = plt.subplots()
            ax.scatter(y, predicted, edgecolors=(0, 0, 0))
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            plt.show()
    else:
        filepath = os.path.join(output_folder, 'cross_val_scoring')
        with open('%s_%s.txt' % (filepath, score), 'w+') as outfile:
            for name, clf in classifiers.items():
                scores = cross_val_score(clf, X, y, cv=skf, scoring=score)
                s = "%-20s%s %0.2f (+/- %0.2f)" % (name, score, scores.mean(), scores.std() * 2)
                print(s)
                outfile.write(s + '\n')

def cross_val_plot(score):
    filepath = os.path.join(output_folder, 'cross_val_plot')
    for name, clf in classifiers.items():
        print('Plotting learning curve for', name)
        plot_learning_curve(clf, name, X, y, scoring=score, cv=skf)
        s = '%s - %s.png' % (filepath, name)
        plt.savefig(s, bbox_inches='tight')
        print('Saved plot to', s)
        # plt.show()


if args.acc:
    cross_val('accuracy', args.score, args.learning_curve, args.predict)
if args.f1_macro:
    cross_val('f1_macro', args.score, args.learning_curve, args.predict)


