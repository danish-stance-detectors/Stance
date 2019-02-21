#external imports
import graphviz
import os
from sklearn import tree
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
train_label = [x[1] for x in train]

# test vectors
test_vec = [x[2] for x in test]
# test labels
labels_true = [x[1] for x in test]

clf = tree.DecisionTreeClassifier(criterion="entropy") 
clf = clf.fit(train_vec, train_label)
labels_pred = clf.predict(test_vec)

model_stats.print_confusion_matrix(labels_true, labels_pred, [0,1,2,3])

# Below lines of code will make visual representation of tree as pdf
# if run multiple times be sure to delete the existing one or rename the output
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("tree_pdfs/tree")