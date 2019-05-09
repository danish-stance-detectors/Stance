import argparse
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, cross_val_predict, cross_validate
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import argparse
import os
import sys
import data_loader
import model_stats

output_folder = '../output/cross_validation/'
rand = np.random.RandomState(42)

def plot_learning_curve(estimator, title, X, y, scoring='f1_macro', ylim=None, cv=3, n_jobs=-1):
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
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross-validation test score")
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

baselines = {
    'majority-vote': DummyClassifier(strategy='most_frequent'),
    'stratified-random': DummyClassifier(strategy='stratified')
}

classifiers = {
    'logit': (LogisticRegression(solver='liblinear', multi_class='auto', penalty='l2', C=50, class_weight='balanced'),
              None),
    'tree': (DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
                                   max_depth=50, max_features=None, presort=True, random_state=rand,
                                   min_samples_split=3, splitter='best'), data_loader.get_features()),
    'svm': (LinearSVC(C=160, class_weight=None, max_iter=50000, multi_class='crammer_singer',
                      tol=3.1, random_state=rand), 'reddit'),
    'rf': (RandomForestClassifier(bootstrap=False, class_weight='balanced_subsample',
                                 criterion='gini', max_depth=10,
                                 max_features='auto', min_samples_split=6,
                                 n_estimators=700, n_jobs=-1, random_state=rand), 'wembs'),
    'mv': DummyClassifier(strategy='most_frequent'),
    'stratify': DummyClassifier(strategy='stratified', random_state=rand),
    'random': DummyClassifier(strategy='uniform', random_state=rand)
}

classifiers_all = {
    'logit': LogisticRegression(solver='liblinear', multi_class='auto', penalty='l2', C=50, class_weight='balanced'),
    # 'tree': DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
    #                                max_depth=50, max_features=None, presort=True, random_state=rand,
    #                                min_samples_split=3, splitter='best'),
    'svm': LinearSVC(C=500, class_weight='balanced', max_iter=50000, multi_class='crammer_singer', random_state=rand),
    'rf': RandomForestClassifier(bootstrap=False, class_weight='balanced_subsample',
                                 criterion='entropy', max_depth=10,
                                 max_features=None, min_samples_split=2,
                                 n_estimators=230, n_jobs=-1, random_state=rand),
    'mv': DummyClassifier(strategy='most_frequent'),
    'stratify': DummyClassifier(strategy='stratified', random_state=rand),
    'random': DummyClassifier(strategy='uniform', random_state=rand)
}

classifiers_vt = {
    # 'logit': LogisticRegression(solver='liblinear', multi_class='auto',
    #                             penalty='l1', C=60, class_weight=None),
    # 'tree': DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
    #                                max_depth=50, max_features=None, presort=True, random_state=rand,
    #                                min_samples_split=3, splitter='best'),
    # 'svm': LinearSVC(C=50, class_weight=None, dual=False, max_iter=50000, random_state=rand),
    # 'rf': RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample',
    #                              criterion='entropy', max_depth=3,
    #                              max_features='auto', min_samples_split=9,
    #                              n_estimators=280, n_jobs=-1, random_state=rand),
    # 'mv': DummyClassifier(strategy='most_frequent'),
    # 'stratify': DummyClassifier(strategy='stratified', random_state=rand),
    'random': DummyClassifier(strategy='uniform', random_state=rand)
}

    
def cross_val_plot(score, X, y, skf):
    filepath = os.path.join(output_folder, 'cross_val_plot')
    for name, clf in classifiers.items():
        print('Plotting learning curve for', name)
        plot_learning_curve(clf, name, X, y, scoring=score, cv=skf)
        s = '%s - %s.png' % (filepath, name)
        plt.savefig(s, bbox_inches='tight')
        print('Saved plot to', s)
        # plt.show()


def cross_val(X, y, clfs, skf, plot_lc, reduced_feature):
    scoring = [
        'f1_macro',
        'accuracy'
    ]
    for name, clf in clfs.items():
        scores = cross_validate(clf, X, y, cv=skf, scoring=scoring, n_jobs=-1, verbose=1,
                                pre_dispatch='2*n_jobs', return_train_score=False)
        s = "%-20s%s %0.2f (+/- %0.2f) %s %0.2f (+/- %0.2f)" \
            % (name, 'f1', scores['test_f1_macro'].mean(), scores['test_f1_macro'].std() * 2,
               'acc', scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2)
        print(s)
        filepath = os.path.join(output_folder, 'cross_validate_k%d' % skf.n_splits)
        if reduced_feature:
            filepath += '_vt'
        with open('%s.txt' % filepath, 'a+') as outfile:
            outfile.write(s + '\n')
        if plot_lc:
            print('Plotting learning curve')
            folder = os.path.join(output_folder, name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            for score in scoring:
                plot_learning_curve(clf, '%s %s' % (name, score), X, y, scoring=score, cv=skf)
                s = os.path.join(folder, 'learning_curve_%s_k%d' % (score, skf.n_splits))
                if reduced_feature:
                    s += '_vt'
                s += '.png'
                plt.savefig(s, bbox_inches='tight')
                print('Saved plot to', s)


def cross_predict(X, y, clfs, skf, reduced_feature):
    for name, clf in clfs.items():
        folder = os.path.join(output_folder, name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        predicted = cross_val_predict(clf, X, y, cv=skf)
        filename = os.path.join(folder, 'prediction_cm_k%d' % skf.n_splits)
        if reduced_feature:
            filename += '_vt'
        _, acc, f1 = model_stats.plot_confusion_matrix(y, predicted,
                                                       title='%s Confusion matrix - no normalization' % name,
                                                       save_to_filename='%s.png' % filename)
        print('Acc: %.5f', acc)
        print('f1: %.5f', f1)


def fit_predict(X_train, X_test, y_train, y_test, clf, name):
    folder = os.path.join(output_folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, '%s_fit_predict_cm' % name)
    clf.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    _, acc, f1 = model_stats.plot_confusion_matrix(y_true, y_pred,
                                                   title='%s Confusion matrix - no normalization' % name,
                                                   save_to_filename='%s.png' % filename)
    print('Acc: %.5f' % acc)
    print('f1: %.5f' % f1)


def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
    parser.add_argument('-x', '--train_file', dest='train_file',
                        default='../data/preprocessed/PP_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_train.csv',
                        help='Input file holding train data')
    parser.add_argument('-y', '--test_file', dest='test_file',
                        default='../data/preprocessed/PP_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_test.csv',
                        help='Input file holding test data')
    parser.add_argument('-k', '--k_folds', dest='k_folds', default=5, type=int, nargs='?',
                        help='Number of folds for cross validation (default=5)')
    parser.add_argument('-l', '--learning_curve', dest='learning_curve', action='store_true', default=False,
                        help='Enable plotting of learning curve')
    parser.add_argument('-cv', '--cross_validate', action='store_true', default=False,
                        help='Cross-validate scoring F1 macro')
    parser.add_argument('-cp', '--cv_predict', action='store_true', default=False,
                        help='Cross-validate prediction')
    parser.add_argument('-p', '--predict', nargs=1, help='Single classifier prediction')
    parser.add_argument('-vskf', '--visualize_skf', action='store_true', default=False,
                        help='Cross-validate prediction')
    parser.add_argument('-r', '--reduce_features', action='store_true', default=False,
                        help='Reduce features by Variance Threshold')
    args = parser.parse_args(argv)
    X_train, X_test, y_train, y_test, n_features, feature_mapping = data_loader.load_train_test_data(
        train_file=args.train_file, test_file=args.test_file
    )
    config = data_loader.get_features()
    # Split data
    X_train = data_loader.select_features(X_train, feature_mapping, config)
    X_test = data_loader.select_features(X_test, feature_mapping, config)
    # Merged data
    X_all = []
    X_all.extend(X_train)
    X_all.extend(X_test)
    y_all = []
    y_all.extend(y_train)
    y_all.extend(y_test)

    clfs = classifiers_all

    if args.reduce_features:
        clfs = classifiers_vt
        print(len(X_train[0]))
        print(len(X_test[0]))
        X_train, X_test = data_loader.union_reduce_then_split(X_train, X_test)
        print(len(X_train[0]))
        print(len(X_test[0]))
        X_all = np.append(X_train, X_test, axis=0)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=rand)

    if args.visualize_skf:
        visualize_cv(skf, args.k_folds, X_all, y_all)
    if args.cross_validate:
        cross_val(X_all, y_all, clfs, skf, args.learning_curve, args.reduce_features)
    if args.cv_predict:
        cross_predict(X_all, y_all, clfs, skf, args.reduce_features)
    if args.predict:
        clf = clfs[args.predict] if args.reduce_features else clfs[args.predict][0]
        fit_predict(X_train, X_test, y_train, y_test, clf, 'logit')


if __name__ == "__main__":
    main(sys.argv[1:])
