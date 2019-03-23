import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import data_loader

classifiers = {
    'Logistic Regression': LogisticRegression(solver='liblinear', C=10, penalty='l1', multi_class='auto'),
    'Decision Tree': DecisionTreeClassifier(criterion="entropy", splitter='random'),
    'Linear SVM': SVC(kernel='linear', C=10),
    'Random Forest': RandomForestClassifier(criterion='entropy', n_estimators=10)
}

scoring = [
    'accuracy',
    'f1_micro',
    'f1_macro',
    'f1_weighted',
]

training_data = '../data/preprocessed/preprocessed.csv'
instances, _ = data_loader.get_instances(training_data, '\t')

X = [x[2] for x in instances]
y = [x[1] for x in instances]

for score in scoring:
    print()
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=5, scoring=score)
        print("%-20s%s %0.2f (+/- %0.2f)" % (name, score, scores.mean(), scores.std() * 2))

