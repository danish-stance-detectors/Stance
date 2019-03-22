import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

import data_loader, model_stats

classifiers = {
    'Logistic Regression': LogisticRegression(solver='liblinear', multi_class='auto'),
    'Decision Tree': DecisionTreeClassifier(criterion="entropy"),
    'Linear SVM': SVC(kernel='linear', C=1),
    'Random Forest': RandomForestClassifier(n_estimators=50)
}

scoring = [
    'accuracy',
    'f1_micro',
    'f1_macro',
    'f1_weighted',
]

# parser = argparse.ArgumentParser(description='Stance classification for different models')
# parser.add_argument('-logit', help='Logistic Regression')
# parser.add_argument('-lstm', help='Long-Short Term Memory RNN')
# parser.add_argument('-random_forest', '-forest', dest='random_forest', help='Random Forest')
# parser.add_argument('-svm', help='Linear Support Vector Machine')
# parser.add_argument('-tree', '-decision_tree', dest='tree', help='Decision Tree')
# args = parser.parse_args()

training_data = '../data/preprocessed/preprocessed.csv'
instances, _ = data_loader.get_instances(training_data, '\t')

X = [x[2] for x in instances]
y = [x[1] for x in instances]

for score in scoring:
    print()
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=5, scoring=score)
        print("%-20s%s %0.2f (+/- %0.2f)" % (name, score, scores.mean(), scores.std() * 2))

