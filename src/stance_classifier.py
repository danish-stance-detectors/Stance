import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
import argparse, os
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
args = parser.parse_args()

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


def cross_val(score):
    filepath = os.path.join(output_folder, 'cross_val_scoring')
    with open('%s_%s.txt' % (filepath, score), 'w+') as outfile:
        for name, clf in classifiers.items():
            scores = cross_val_score(clf, X, y, cv=skf, scoring=score)
            s = "%-20s%s %0.2f (+/- %0.2f)" % (name, score, scores.mean(), scores.std() * 2)
            print(s)
            outfile.write(s + '\n')


if args.acc:
    cross_val('accuracy')
if args.f1_macro:
    cross_val('f1_macro')


