import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as sk


class StanceLSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab_size, num_labels):
        super(StanceLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, emb_dim)

        # LSTM layer
        self.lstm = nn.LSTM(emb_dim, hidden_dim)

        # Linear layer
        self.hidden2label = nn.Linear(hidden_dim, num_labels)

        #initialize state
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        label_space = self.hidden2label(lstm_out[-1])
        label_scores = F.log_softmax(label_space, dim=1)
        return label_scores

EMB = 100
HIDDEN_DIM = 6
EPOCHS = 100

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq.split()]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("The dog ate the apple", 'C'),
    ("Everybody read that book", 'D')
]

w2i = {}
for sent, _ in training_data:
    for word in sent.split():
        if word not in w2i:
            w2i[word] = len(w2i)
w2i['cat'] = len(w2i) #For testing
w2i['page'] = len(w2i)#For testing
l2i = {'S': 0, 'D': 1, 'Q': 2, 'C': 3}

model = StanceLSTM(EMB, HIDDEN_DIM, len(w2i), len(l2i))
loss_func = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("#Training")
for epoch in range(EPOCHS):
    avg_loss = 0.0
    for sentence, label in training_data:
        # Clear stored gradient
        model.zero_grad()

        # Initialize hidden state
        model.hidden = model.init_hidden()

        # Prepare data
        sentence_in = prepare_sequence(sentence, w2i)
        target = torch.tensor([l2i[label]], dtype=torch.long)

        # Make prediction
        pred = model(sentence_in)

        # Calculate loss
        loss = loss_func(pred, target)
        avg_loss += loss.item()

        loss.backward() # Back propagate
        optimizer.step() # Update parameters
    avg_loss /= len(training_data)
    print("Epoch: {0}\tavg_loss: {1}".format(epoch, avg_loss))
        

test_data = [
    ("The cat ate the apple", 'C'),
    ("Everybody read that page", 'D')
]

print("#Testing")
with torch.no_grad():
    labels_true = []
    labels_pred = []
    for sent, label in test_data:
        sentence_in = prepare_sequence(sent, w2i)
        pred = model(sentence_in)
        predicted = torch.argmax(pred.data, dim=1)
        pred_val = predicted[0].item()
        label_val = l2i[label]
        print("Predicted: {0}\tActual: {1}".format(pred_val,label_val))
        labels_pred.append(pred_val)
        labels_true.append(label_val)
    acc = sk.accuracy_score(labels_true, labels_pred)
    print("Accuracy: ", acc)