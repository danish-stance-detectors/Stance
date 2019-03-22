from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import data_loader
import model_stats

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

clf = svm.LogisticRegression()
clf.fit(train_vec, train_label)

labels_pred = clf.predict(test_vec)

model_stats.print_confusion_matrix(labels_true, labels_pred, [0,1,2,3])