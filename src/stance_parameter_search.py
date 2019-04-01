from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import data_loader


X_train, X_test, y_train, y_test, _ = data_loader.get_train_test_split(test_size=0.25)

settings = [
    ('rbf-svm', SVC(), {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}),
    ('linear-svm', SVC(), {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}),
    ('tree', DecisionTreeClassifier(), {'criterion': ['entropy', 'gini'], 'splitter':['best', 'random']}),
    ('logistic-regression', LogisticRegression(), {'solver': ['liblinear'], 'penalty':['l1', 'l2'],
                                                   'C': [1, 10, 100, 1000], 'multi_class': ['auto']}),
    ('random-forest', RandomForestClassifier(), {'n_estimators': [10, 20, 50, 100], 'criterion': ['entropy', 'gini']})

]

scores = [
    'accuracy',
    'f1_macro'
]
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for score in scores:
    for name, estimator, tuned_parameters in settings:
        print("# Tuning hyper-parameters for %s with %s" % (name, score))
        print()
        clf = GridSearchCV(
            estimator, tuned_parameters, cv=skf, scoring=score, n_jobs=-1, error_score=0
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

