import sklearn.metrics as sk
import numpy as np

def print_confusion_matrix(labels_true, labels_pred, act_labels):
    cm = sk.confusion_matrix(labels_true, labels_pred, labels=act_labels)
    print("Confusion matrix:")
    print("  S  D  Q  C")
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sdqc_acc = cm.diagonal()
    acc = sk.accuracy_score(labels_true, labels_pred)
    f1 = sk.f1_score(labels_true, labels_pred, average='macro')
    print("SDQC acc:", sdqc_acc)
    print("Accuracy: %.5f" % acc )
    print("F1-macro:", f1)