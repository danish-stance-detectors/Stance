import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as sk
from sklearn.model_selection import train_test_split
import numpy as np

import data_loader

class StanceLSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_labels, vocab_size=0, pre_trained=True):
        super(StanceLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pre_trained = pre_trained
        if not pre_trained:
            self.word_embeddings = nn.Embedding(vocab_size, emb_dim)

        # LSTM layer
        self.lstm = nn.LSTM(emb_dim, hidden_dim, 2)

        # Linear layer
        self.hidden2label = nn.Linear(hidden_dim, num_labels)

        #initialize state
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (hidden_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))

    def forward(self, data):
        x = data
        if not self.pre_trained:
            x = self.word_embeddings(data)
        lstm_out, self.hidden = self.lstm(
            x.view(len(x), 1, -1), self.hidden)
        label_space = self.hidden2label(lstm_out[-1])
        label_scores = F.log_softmax(label_space, dim=1)
        return label_scores
    

# EMB = 100
# HIDDEN_DIM = 6
# EPOCHS = 100

# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq.split()]
#     return torch.tensor(idxs, dtype=torch.long)

# training_data = [
#     ("The dog ate the apple", 'C'),
#     ("Everybody read that book", 'D')
# ]

# w2i = {}
# for sent, _ in training_data:
#     for word in sent.split():
#         if word not in w2i:
#             w2i[word] = len(w2i)
# w2i['cat'] = len(w2i) #For testing
# w2i['page'] = len(w2i)#For testing
l2i = {'S': 0, 'D': 1, 'Q': 2, 'C': 3}

training_data = '../data/preprocessed/preprocessed.csv'
instances, emb_size = data_loader.get_instances(training_data, '\t')
train, test = train_test_split(instances, test_size=0.25, random_state=42)
EMB = emb_size
HIDDEN_DIM = 100
EPOCHS = 30

model = StanceLSTM(EMB, HIDDEN_DIM, len(l2i), pre_trained=True)
loss_func = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("#Training")
for epoch in range(EPOCHS):
    avg_loss = 0.0
    for _, label, vec in train:
        # Clear stored gradient
        model.zero_grad()

        # Initialize hidden state
        model.hidden = model.init_hidden()

        # Prepare data
        # sentence_in = prepare_sequence(sentence, w2i)
        data_in = torch.tensor([vec])
        target = torch.tensor([label])

        # Make prediction
        pred = model(data_in)

        # Calculate loss
        loss = loss_func(pred, target)
        avg_loss += loss.item()

        loss.backward() # Back propagate
        optimizer.step() # Update parameters
    avg_loss /= len(training_data)
    print("Epoch: {0}\tavg_loss: {1}".format(epoch, avg_loss))
        

# test_data = [
#     ("The cat ate the apple", 'C'),
#     ("Everybody read that page", 'D')
# ]

print("#Testing")
with torch.no_grad():
    labels_true = []
    labels_pred = []
    for _, label, vec in test:
        data_in = torch.tensor([vec])
        pred = model(data_in)
        predicted = torch.argmax(pred.data, dim=1)
        pred_val = predicted[0].item()
        # print("Predicted: {0}\tActual: {1}".format(pred_val,label))
        labels_pred.append(pred_val)
        labels_true.append(label)
    cm = sk.confusion_matrix(labels_true, labels_pred, labels=[0, 1, 2, 3])
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