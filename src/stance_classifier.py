import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, cross_val_predict
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import argparse
import os
import sys
import data_loader
import model_stats

output_folder = '../output/'


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

def plot_cv_indices(cv, X, y, ax, n_splits, cmap_data, cmap_cv, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class']
    ax.set(yticks=np.arange(n_splits+1) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+1.2, -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

classifiers = {
    'Logistic Regression': LogisticRegression(solver='liblinear', C=10, penalty='l1', multi_class='auto'),
    'Decision Tree': DecisionTreeClassifier(criterion="entropy", splitter='random'),
    'RBF SVM': SVC(kernel='rbf', C=1000, gamma=0.001),
    'Linear SVM': SVC(kernel='linear', C=10),
    'Random Forest': RandomForestClassifier(criterion='entropy', n_estimators=10),
    'Majority vote': DummyClassifier(strategy='most_frequent'),
    'Stratified Random': DummyClassifier(strategy='stratified')
}


def visualize_cv(cv, n_splits, X, y):
    fig, ax = plt.subplots()
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm
    plot_cv_indices(cv, X, y, ax, n_splits, cmap_data, cmap_cv)
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
    # Make the legend fit
    plt.tight_layout()
    plt.show()

    
def cross_val_plot(score):
    filepath = os.path.join(output_folder, 'cross_val_plot')
    for name, clf in classifiers.items():
        print('Plotting learning curve for', name)
        plot_learning_curve(clf, name, X, y, scoring=score, cv=skf)
        s = '%s - %s.png' % (filepath, name)
        plt.savefig(s, bbox_inches='tight')
        print('Saved plot to', s)
        # plt.show()


def cross_val(score_metric, X, y, skf, score=False,  plot=False):
    if plot:
        cross_val_plot(score_metric)
    if score:
        filepath = os.path.join(output_folder, 'cross_val_scoring')
        with open('%s_%s.txt' % (filepath, score_metric), 'w+') as outfile:
            for name, clf in classifiers.items():
                scores = cross_val_score(clf, X, y, cv=skf, scoring=score_metric)
                s = "%-20s%s %0.2f (+/- %0.2f)" % (name, score_metric, scores.mean(), scores.std() * 2)
                print(s)
                outfile.write(s + '\n')

def cross_predict(X, y, skf):
    for name, clf in classifiers.items():
        predicted = cross_val_predict(clf, X, y, cv=skf)
        model_stats.plot_confusion_matrix(y, predicted, title='%s Confusion matrix - no normalization' % name,
                                          save_to_filename='%s_cm.png' % (output_folder + name))

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
    parser.add_argument('-x', '--train_file', dest='train_file', default='../data/preprocessed/preprocessed_train.csv',
                        help='Input file holding train data')
    parser.add_argument('-y', '--test_file', dest='test_file', default='../data/preprocessed/preprocessed_test.csv',
                        help='Input file holding test data')
    parser.add_argument('-k', '--k_folds', dest='k_folds', const=5, type=int, nargs='?',
                        help='Number of folds for cross validation (default=5)')
    parser.add_argument('-a', '--accuracy', dest='acc', action='store_true', default=False,
                        help='Enable accuracy metric')
    parser.add_argument('-f', '--f1_macro', dest='f1_macro', action='store_true', default=False,
                        help='Enable F1 macro metric')
    parser.add_argument('-l', '--learning_curve', dest='learning_curve', action='store_true', default=False,
                        help='Enable plotting of learning curve')
    parser.add_argument('-s', '--score', dest='score', action='store_true', default=True,
                        help='Cross-validate scoring')
    parser.add_argument('-p', '--predict', action='store_true', default=False,
                        help='Cross-validate prediction')

    args = parser.parse_args(argv)
    X, y, _, feature_mapping = data_loader.load_train_test_data(train_file=args.train_file,
                                                                test_file=args.test_file, split=False)
    X = data_loader.select_features(X, feature_mapping, text=True, lexicon=False, pos=False)
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    # visualize_cv(skf, args.k_folds, X, y)
    cross_val('accuracy' if args.acc else 'f1_macro', X, y, skf, args.score, args.learning_curve)
    if args.predict:
        cross_predict(X, y, skf)

if __name__ == "__main__":
    main(sys.argv[1:])