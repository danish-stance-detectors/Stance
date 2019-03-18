#external imports
import graphviz
import os
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#internal imports (our py classes)
import data_loader
import model_stats

training_data = '../data/preprocessed/preprocessed_nouser.csv'
instances, emb_size = data_loader.get_instances(training_data, '\t')

train, test = train_test_split(instances, test_size=0.25, random_state=42)

# train vectors
train_vec = [x[2] for x in train]
# train labels
train_label = [x[1] for x in train]

train_comment_label = [1 if x[1] == 3 else 0 for x in train]

train_vec_no_comments = [x[2] for x in train if x[1] != 3]
train_label_no_comments = [x[1] for x in train if x[1] != 3]

# test vectors
test_vec = [x[2] for x in test]
test_vec_no_comments = [x[2] for x in test if x[1] != 3]

# test labels
labels_true = [x[1] for x in test]
labels_true_no_comments = [x[1] for x in test if x[1] != 3]
# labels for comment or not
test_comment_labels = [1 if x[1] == 3 else 0 for x in test]

# train tree to recognize comments
comment_tree = gnb = GaussianNB()#svm.LinearSVC()#tree.DecisionTreeClassifier(criterion="entropy")
comment_tree = comment_tree.fit(train_vec, train_comment_label)
comment_pred = comment_tree.predict(test_vec)

model_stats.print_confusion_matrix(test_comment_labels, comment_pred, [0,1])

non_comment_vec = []
for (i, pred) in enumerate(comment_pred, start=0):
    if pred == 0:
        non_comment_vec.append((i, test_vec[i]))
    else:
        comment_pred[i] = 3 # set as comment

# train tree to recognize all
clf = tree.DecisionTreeClassifier(criterion="entropy") 
clf = clf.fit(train_vec_no_comments, train_label_no_comments)
labels_pred = clf.predict([x[1] for x in non_comment_vec])

for (idx, act_pred) in enumerate(labels_pred):
    pred_idx = non_comment_vec[idx][0]
    comment_pred[pred_idx] = act_pred

#model_stats.print_confusion_matrix(labels_true_no_comments, labels_pred, [0,1,2])

# two_step_pred = [3 if comment_pred[x] == 1 else labels_pred[x] for x in range(len(labels_pred))]

model_stats.print_confusion_matrix(labels_true, comment_pred, [0,1,2,3])
