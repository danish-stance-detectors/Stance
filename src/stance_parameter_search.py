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

output_folder = '../output/'

parser = argparse.ArgumentParser(description='Hyper parameter search for stance classification models')
parser.add_argument('-i', '--input_file', dest='file', default='../data/preprocessed/preprocessed.csv',
                    help='Relative input file path holding train data')
parser.add_argument('-k', '--k_folds', dest='k_folds', default=5, type=int, nargs='?',
                    help='Number of folds for cross validation (default=5)')
args = parser.parse_args()


X_train, X_test, y_train, y_test, _ = data_loader.get_train_test_split(filename=args.file, test_size=0.25)

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


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [0, 1, 2, 3]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


scores = [
    'accuracy',
    'f1_macro'
]
skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

for name, estimator, tuned_parameters in settings:
    filepath = os.path.join(output_folder, name)
    with open('%s_parameters.txt' % filepath, 'w+') as outfile:
        s = "# Tuning hyper-parameters on F1 macro for %s" % name
        print(s)
        outfile.write(s + '\n')
        print()
        clf = GridSearchCV(
            estimator, tuned_parameters, scoring=scores, n_jobs=-1, error_score=0, refit='f1_macro',
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
        for scorer in scores:
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
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        s = classification_report(y_true, y_pred)
        print(s)
        outfile.write(s + '\n')
        print()
        np.set_printoptions(precision=2)
        plot_confusion_matrix(y_test, y_pred, title='Confusion matrix, without normalization')
        plt.savefig('%s_cm.png' % filepath, bbox_inches='tight')



