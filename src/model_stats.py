import sklearn.metrics as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def cm_acc_f1(labels_true, labels_pred):
    cm = sk.confusion_matrix(labels_true, labels_pred)
    acc = sk.accuracy_score(labels_true, labels_pred)
    f1 = sk.f1_score(labels_true, labels_pred, average='macro')
    return cm, acc, f1

def cm_acc_f1_sdqc(labels_true, labels_pred):
    cm = sk.confusion_matrix(labels_true, labels_pred)
    acc = sk.accuracy_score(labels_true, labels_pred)
    f1 = sk.f1_score(labels_true, labels_pred, average='macro')
    sdqc_acc = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).diagonal()
    return cm, acc, f1, sdqc_acc


def print_confusion_matrix(labels_true, labels_pred):
    cm, acc, f1 = cm_acc_f1(labels_true, labels_pred)
    print("Confusion matrix:")
    print("  S  D  Q  C")
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sdqc_acc = cm.diagonal()
    acc = sk.accuracy_score(labels_true, labels_pred)
    f1 = sk.f1_score(labels_true, labels_pred, average='macro')
    precision = sk.precision_score(labels_true, labels_pred)
    recall = sk.recall_score(labels_true, labels_pred)

    print("SDQC acc:", sdqc_acc)
    print("Accuracy: %.5f" % acc )
    print("F1-macro:", f1)
    print("precision:", precision)
    print("recall:", recall)

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          save_to_filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm, acc, f1, sdqc_acc = cm_acc_f1_sdqc(y_true, y_pred)
    # cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['S', 'D', 'Q', 'C']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_to_filename:
        plt.savefig(save_to_filename, bbox_inches='tight')
    return cm, acc, f1, sdqc_acc