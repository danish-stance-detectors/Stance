import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import data_loader, model_stats

classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(criterion="entropy"),
    'Linear Support Vector Machine': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(n_estimators=50)
}

# parser = argparse.ArgumentParser(description='Stance classification for different models')
# parser.add_argument('-logit', help='Logistic Regression')
# parser.add_argument('-lstm', help='Long-Short Term Memory RNN')
# parser.add_argument('-random_forest', '-forest', dest='random_forest', help='Random Forest')
# parser.add_argument('-svm', help='Linear Support Vector Machine')
# parser.add_argument('-tree', '-decision_tree', dest='tree', help='Decision Tree')
# args = parser.parse_args()

training_data = '../data/preprocessed/preprocessed.csv'
instances, emb_size = data_loader.get_instances(training_data, '\t')

train, test = train_test_split(instances, test_size=0.25)

# train vectors
train_vec = [x[2] for x in train]
# train labels
train_label = [x[1] for x in train]

# test vectors
test_vec = [x[2] for x in test]
# test labels
labels_true = [x[1] for x in test]

for name, clf in classifiers.items():
    clf.fit(train_vec, train_label)
    labels_pred = clf.predict(test_vec)
    print(name)
    model_stats.print_confusion_matrix(labels_true, labels_pred, [0,1,2,3])

