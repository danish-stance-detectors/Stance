#external imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#internal imports (our py classes)
import data_loader
import model_stats

training_data = '../data/preprocessed/preprocessed.csv'
instances, emb_size = data_loader.get_instances(training_data, '\t')

train, test = train_test_split(instances, test_size=0.25, random_state=42)

# train vectors
train_vec = [x[2] for x in train]
# train labels
train_labels = [x[1] for x in train]

# test vector
test_vec = [x[2] for x in test]
# true labels
labels_true = [x[1] for x in test]

# make classifier, fit and predict
clf = RandomForestClassifier(n_estimators=10) 
clf = clf.fit(train_vec, train_labels)
labels_pred = clf.predict(test_vec)

model_stats.print_confusion_matrix(labels_true, labels_pred, [0, 1, 2, 3])