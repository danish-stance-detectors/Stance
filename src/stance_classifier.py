import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier

import data_loader

classifiers = {
    'Logistic Regression': LogisticRegression(solver='liblinear', C=10, penalty='l1', multi_class='auto'),
    'Decision Tree': DecisionTreeClassifier(criterion="entropy", splitter='random'),
    'Linear SVM': SVC(kernel='linear', C=10),
    'Random Forest': RandomForestClassifier(criterion='entropy', n_estimators=10),
    'Majority vote': DummyClassifier(strategy='most_frequent'),
    'Stratified Random': DummyClassifier(strategy='stratified')
}

scoring = [
    'accuracy',
    'f1_macro'
]

X, y, _ = data_loader.get_features_and_labels()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for score in scoring:
    print()
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=skf, scoring=score)
        print("%-20s%s %0.2f (+/- %0.2f)" % (name, score, scores.mean(), scores.std() * 2))

