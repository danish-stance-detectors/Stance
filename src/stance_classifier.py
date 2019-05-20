import argparse
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, cross_val_predict, cross_validate, StratifiedShuffleSplit
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import argparse
import os
import sys
import csv
import data_loader
import model_stats

output_folder = '../output/cross_validation/'
rand = np.random.RandomState(42)

def plot_learning_curve(estimator, title, X, y, scoring='f1_macro', ylim=None, cv=5, n_jobs=-1):
    plt.figure()
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("%s score" % scoring)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring, verbose=1
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

def feature_LOO_cross_val(X_train, X_test, y_train, y_test, vt, features, feature_mapping, skf, clfs):
    feature_names = features.keys()
    scoring = [
        'f1_macro',
        'accuracy'
    ]
    y_all = []
    y_all.extend(y_train)
    y_all.extend(y_test)
    folder = os.path.join(output_folder, 'LOO/')
    if not os.path.exists(folder):
        os.mkdir(folder)
    for name, clf in clfs.items():
        filepath = os.path.join(folder, name)
        if vt:
            filepath += '_vt'
        if not os.path.exists(filepath):
            with open('%s.csv' % filepath, 'w+', newline='') as statsfile:
                csv_writer = csv.writer(statsfile)
                csv_writer.writerow(['estimator', 'f1', 'f1_std', 'acc', 'acc_std', 'LOO feature'])
        else:
            continue
        for feature_name in feature_names:
            if not features[feature_name] :
                print('Skipping %s' % feature_name)
                continue
            if feature_name == 'all':
                print('Running with all features enabled')
            else:
                print('Leaving %s features out' % feature_name)
            features[feature_name] = False
            X_train_ = data_loader.select_features(X_train, feature_mapping, features)
            X_test_ = data_loader.select_features(X_test, feature_mapping, features)
            if vt:
                old_len = len(X_train_[0])
                X_train_, X_test_ = data_loader.union_reduce_then_split(X_train_, X_test_)
                new_len = len(X_train_[0])
                print('Reduced features from %d to %d' % (old_len, new_len))
            X_all = []
            X_all.extend(X_train_)
            X_all.extend(X_test_)
            with open('%s.csv' % filepath, 'a', newline='') as statsfile:
                csv_writer = csv.writer(statsfile)
                scores = cross_validate(clf, X_all, y_all, cv=skf, scoring=scoring, n_jobs=-1, verbose=1,
                                        pre_dispatch='2*n_jobs', return_train_score=False, error_score='raise')
                f1 = scores['test_f1_macro'].mean()
                f1_std = scores['test_f1_macro'].std() * 2
                acc = scores['test_accuracy'].mean()
                acc_std = scores['test_accuracy'].std() * 2
                print("%-20s%s %0.2f (+/- %0.2f) %s %0.2f (+/- %0.2f)" % (name, 'f1', f1, f1_std, 'acc', acc, acc_std))
                csv_writer.writerow([name, '%.4f' % f1, '%0.2f' % f1_std, '%.4f' % acc, '%0.2f' % acc_std, feature_name])
            features[feature_name] = True

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
    'mv': DummyClassifier(strategy='most_frequent'),
    'sc': DummyClassifier(strategy='stratified', random_state=rand),
    'random': DummyClassifier(strategy='uniform', random_state=rand)
}

classifiers_LOO = {
    'logit': (LogisticRegression(solver='liblinear', multi_class='auto', penalty='l2', C=50, class_weight='balanced'),
              None),
    'tree': (DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
                                   max_depth=50, max_features=None, presort=True, random_state=rand,
                                   min_samples_split=3, splitter='best'), 'lexicon'),
    'svm': (LinearSVC(C=160, class_weight=None, max_iter=50000, multi_class='crammer_singer',
                      tol=3.1, random_state=rand), 'reddit'),
    'rf': (RandomForestClassifier(bootstrap=False, class_weight='balanced_subsample',
                                  criterion='gini', max_depth=10,
                                  max_features='auto', min_samples_split=6,
                                  n_estimators=700, n_jobs=-1, random_state=rand), 'wembs')
}

classifiers_simple = {
    'logit': LogisticRegression(solver='liblinear', multi_class='auto', random_state=rand, max_iter=50000),
    # 'tree': DecisionTreeClassifier(presort=True, random_state=rand),
    'svm': LinearSVC(random_state=rand, max_iter=50000),
    # 'rf': RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=rand)
}

classifiers_all = {
    'logit': LogisticRegression(solver='liblinear', multi_class='auto', penalty='l2', C=50, class_weight='balanced'),
    'tree': DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
                                   max_depth=10, max_features=None, presort=True, random_state=rand,
                                   min_samples_split=8, splitter='best'),
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
    'logit': LogisticRegression(solver='liblinear', multi_class='auto', penalty='l2', C=50, class_weight='balanced'),
    'tree': DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
                                   max_depth=50, max_features=None, presort=True, random_state=rand,
                                   min_samples_split=3, splitter='best'),
    'svm': LinearSVC(penalty='l2', C=50, class_weight=None, dual=False, max_iter=50000, random_state=rand),
    'rf': RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample',
                                 criterion='entropy', max_depth=3,
                                 max_features='auto', min_samples_split=9,
                                 n_estimators=280, n_jobs=-1, random_state=rand)
}

classifiers_best = {
    'logit': LogisticRegression(solver='liblinear', multi_class='auto', dual=True,
                                penalty='l2', C=1, class_weight='balanced', max_iter=50000),
    'svm': LinearSVC(penalty='l2', C=10, class_weight=None, dual=True, max_iter=50000, random_state=rand),
    'mv': DummyClassifier(strategy='most_frequent'),
    'stratify': DummyClassifier(strategy='stratified', random_state=rand),
    'random': DummyClassifier(strategy='uniform', random_state=rand)
}

    
def cross_val_plot(X, y, skf, clfs, score='f1_macro'):
    for name, clf in clfs.items():
        folder = os.path.join(output_folder, name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        print('Plotting learning curve for', name)
        plot_learning_curve(clf, name, X, y, scoring=score, cv=skf)
        s = os.path.join(folder, 'learning_curve_%s_k%d' % (score, skf.n_splits))
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
                                pre_dispatch='2*n_jobs', return_train_score=False, error_score='raise')
        s = "%-20s%s %0.4f (+/- %0.2f) %s %0.4f (+/- %0.2f)" \
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
                plot_learning_curve(clf, name, X, y, scoring=score, cv=skf)
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
        cm, acc, f1, sdqc_acc = model_stats.plot_confusion_matrix(y, predicted,
                                                       title='%s Confusion matrix - no normalization' % name,
                                                       save_to_filename='%s.png' % filename)
        target_names = ['S', 'D', 'Q', 'C']
        cr = classification_report(y, predicted, labels=[0, 1, 2, 3], target_names=target_names, output_dict=True)
        print('Acc: %.5f' % acc)
        print('f1: %.5f' % f1)
        print("SDQC acc:", sdqc_acc)
        sdqc_f1 = [cr['S']['f1-score'], cr['D']['f1-score'], cr['Q']['f1-score'], cr['C']['f1-score']]
        print('SDQC f1:', sdqc_f1)


def fit_predict(X_train, X_test, y_train, y_test, clfs, name):
    clf = clfs[name]
    folder = os.path.join(output_folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, '%s_fit_predict_cm' % name)
    clf.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    cm, acc, f1, sdqc_acc = model_stats.plot_confusion_matrix(y_true, y_pred,
                                                              title='%s confusion matrix' % name,
                                                              save_to_filename='%s.png' % filename)
    target_names = ['S', 'D', 'Q', 'C']
    cr = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=target_names, output_dict=True)
    print('Acc: %.4f' % acc)
    print('f1: %.4f' % f1)
    print("SDQC acc:", sdqc_acc)
    sdqc_f1 = [cr['S']['f1-score'], cr['D']['f1-score'], cr['Q']['f1-score'], cr['C']['f1-score']]
    print('SDQC f1:', sdqc_f1)
    with open('%s.txt' % filename, 'w+') as stats:
        stats.write(np.array2string(cm) + '\n')
        stats.write('Acc: %.4f\n' % acc)
        stats.write('F1: %.4f\n' % f1)
        stats.write('SDQC acc: {}\n'.format(sdqc_acc))
        stats.write('SDQC f1 : {}\n'.format(sdqc_f1))


def BOW_VT(X_train, X_test, y_train, y_test, feature_mapping):
    config_LRMFW = data_loader.get_features(reddit=False, lexicon=False, most_freq=False, bow=False)
    config_BOW = data_loader.get_features(all_true=False)
    config_BOW['bow'] = True
    # Split data
    X_train_LRMFW = data_loader.select_features(X_train, feature_mapping, config_LRMFW)
    X_test_LRMFW = data_loader.select_features(X_test, feature_mapping, config_LRMFW)
    X_train_BOW = data_loader.select_features(X_train, feature_mapping, config_BOW)
    X_test_BOW = data_loader.select_features(X_test, feature_mapping, config_BOW)
    # Merged data
    X_all_LRMFW = []
    X_all_LRMFW.extend(X_train_LRMFW)
    X_all_LRMFW.extend(X_test_LRMFW)
    y_all = []
    y_all.extend(y_train)
    y_all.extend(y_test)

    print(len(X_train_BOW[0]))
    print(len(X_test_BOW[0]))
    X_train_BOW_, X_test_BOW_ = data_loader.union_reduce_then_split(X_train_BOW, X_test_BOW)
    print(len(X_train_BOW_[0]))
    print(len(X_test_BOW_[0]))
    X_all_BOW = np.append(X_train_BOW_, X_test_BOW_, axis=0)
    X_all = []
    for x1, x2 in zip(X_all_LRMFW, X_all_BOW):
        row = []
        row.extend(x1)
        row.extend(x2.tolist())
        X_all.append(row)
    X_all = np.array(X_all, dtype=np.float64, order='C')
    print(len(X_all[0]))
    return X_all, y_all


def CV_sup(clfs, name, X, y, X_sup, y_sup, k_folds=5):
    clf = clfs[name]
    rs = StratifiedShuffleSplit(n_splits=k_folds, test_size=.25, random_state=rand)
    f1s = []
    accs = []
    for train_i, test_i in rs.split(X, y):
        X_train = [X[i] for i in train_i]
        X_train = np.append(X_train, X_sup, axis=0)
        X_test = [X[i] for i in test_i]
        y_train = [y[i] for i in train_i]
        y_train.extend(y_sup)
        y_test = [y[i] for i in test_i]
        clf.fit(X_train, y_train)
        y_true, y_pred = y_test, clf.predict(X_test)
        _, acc, f1 = model_stats.cm_acc_f1(y_true, y_pred)
        f1s.append(f1)
        accs.append(acc)
    f1s = np.asarray(f1s)
    accs = np.asarray(accs)
    s = "%-20s%s %0.4f (+/- %0.2f) %s %0.4f (+/- %0.2f)" \
        % (name, 'f1', f1s.mean(), f1s.std() * 2,
           'acc', accs.mean(), accs.std() * 2)
    print(s)


def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')
    parser.add_argument('-x', '--train_file', dest='train_file',
                        default='../data/preprocessed/PP_sup50_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_train.csv',
                        help='Input file holding train data')
    parser.add_argument('-y', '--test_file', dest='test_file',
                        default='../data/preprocessed/PP_sup50_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_test.csv',
                        help='Input file holding test data')
    parser.add_argument('-sup', '--sup_file',
                        default='../data/preprocessed/PP_sup50_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_sup.csv',
                        help='Input file holding test data')
    parser.add_argument('-k', '--k_folds', dest='k_folds', default=5, type=int, nargs='?',
                        help='Number of folds for cross validation (default=5)')
    parser.add_argument('-l', '--learning_curve', dest='learning_curve', action='store_true', default=False,
                        help='Enable plotting of learning curve')
    parser.add_argument('-cv', '--cross_validate', action='store_true', default=False,
                        help='Cross-validate scoring F1 macro')
    parser.add_argument('-cvsup', '--cross_validate_sup', action='store_true', default=False,
                        help='Cross-validate scoring F1 macro with super sampling')
    parser.add_argument('-cvp', '--cross_val_plot', action='store_true', default=False,
                        help='Plot learning curve through CV')
    parser.add_argument('-cp', '--cv_predict', action='store_true', default=False,
                        help='Cross-validate prediction')
    parser.add_argument('-loo', '--loo_features', action='store_true', default=False,
                        help='LOO features with CV')
    parser.add_argument('-p', '--predict', type=str, help='Single classifier prediction')
    parser.add_argument('-vskf', '--visualize_skf', action='store_true', default=False,
                        help='Cross-validate prediction')
    parser.add_argument('-r', '--reduce_features', action='store_true', default=False,
                        help='Reduce features by Variance Threshold')
    args = parser.parse_args(argv)
    X_train, y_train, n_features, feature_mapping, train_ids = data_loader.get_features_and_labels(args.train_file,
                                                                                                   with_ids=True)
    X_test, y_test, _, _, test_ids = data_loader.get_features_and_labels(args.test_file, with_ids=True)
    # X_train, X_test, y_train, y_test, n_features, feature_mapping = data_loader.load_train_test_data(
    #     train_file=args.train_file, test_file=args.test_file
    # )
    config = data_loader.get_features(most_freq=False, reddit=False, lexicon=False)
    # Split data
    X_train_ = np.asarray(data_loader.select_features(X_train, feature_mapping, config), dtype=np.float64, order='C')
    X_test_ = np.asarray(data_loader.select_features(X_test, feature_mapping, config), dtype=np.float64, order='C')
    X_sup_, y_sup = [], []
    if args.sup_file:
        X_sup, y_sup, _, _, sup_ids = data_loader.get_features_and_labels(args.sup_file, with_ids=True)
        X_train_new, y_train_new = [], []
        for x_id, x_tr, y_tr in zip(train_ids, X_train, y_train):
            if '%s_' % x_id in sup_ids:
                X_sup.append(x_tr)
                y_sup.append(y_tr)
            else:
                X_train_new.append(x_tr)
                y_train_new.append(y_tr)
        X_train_ = np.asarray(data_loader.select_features(X_train_new, feature_mapping, config),
                              dtype=np.float64, order='C')
        y_train = y_train_new
        X_sup_ = np.asarray(data_loader.select_features(X_sup, feature_mapping, config), dtype=np.float64, order='C')

    # Merged data
    X_all = np.append(X_train_, X_test_, axis=0)
    y_all = []
    y_all.extend(y_train)
    y_all.extend(y_test)

    clfs = classifiers_best

    # X_all, y_all = BOW_VT(X_train, X_test, y_train, y_test, feature_mapping)

    if args.reduce_features:
        # clfs = classifiers_vt
        print(len(X_train_[0]))
        print(len(X_test_[0]))
        X_train_, X_test_ = data_loader.union_reduce_then_split(X_train_, X_test_)
        print(len(X_train_[0]))
        print(len(X_test_[0]))
        X_all = np.append(X_train_, X_test_, axis=0)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=rand)

    if args.cross_validate_sup:
        CV_sup(clfs, 'svm', X_all, y_all, X_sup_, y_sup, k_folds=3)
    if args.cross_val_plot:
        cross_val_plot(X_all, y_all, skf, clfs)
    if args.loo_features:
        feature_LOO_cross_val(X_train, X_test, y_train, y_test, args.reduce_features, config, feature_mapping, skf, clfs)
    if args.visualize_skf:
        visualize_cv(skf, args.k_folds, X_all, y_all)
    if args.cross_validate:
        cross_val(X_all, y_all, clfs, skf, args.learning_curve, args.reduce_features)
    if args.cv_predict:
        cross_predict(X_all, y_all, clfs, skf, args.reduce_features)
    if args.predict:
        fit_predict(X_train_, X_test_, y_train, y_test, clfs, args.predict)


if __name__ == "__main__":
    main(sys.argv[1:])
