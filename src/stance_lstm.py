import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as sk
from sklearn.metrics import classification_report
import collections
import argparse
import data_loader

parser = argparse.ArgumentParser(description='Train and test LSTM model')
parser.add_argument('-c', '--cuda', dest='cuda', action='store_true', help='Enable CUDA')
args = parser.parse_args()

class StanceLSTM(nn.Module):
    def __init__(self, lstm_layers, lstm_dim, hidden_layers, hidden_dim,
                 num_labels, emb_dim, vocab_size=0, dropout=False, pre_trained=True):
        super(StanceLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.pre_trained = pre_trained
        if not pre_trained:
            self.word_embeddings = nn.Embedding(vocab_size, emb_dim)

        # LSTM layer
        self.lstm = nn.LSTM(emb_dim, lstm_dim, lstm_layers)

        # Linear layer
        dense_layers = collections.OrderedDict()
        dense_layers["lin0"] = torch.nn.Linear(lstm_dim, hidden_dim)
        dense_layers["rec0"] = torch.nn.ReLU()
        for i in range(hidden_layers - 1):
            dense_layers["lin%d" % (i + 1)] = torch.nn.Linear(hidden_dim, hidden_dim)
            dense_layers["rec%d" % (i + 1)] = torch.nn.ReLU()
        if dropout:
            dense_layers["drop"] = torch.nn.Dropout(p=0.5)
        dense_layers["lin%d" % hidden_layers] = torch.nn.Linear(hidden_dim, num_labels)
        self.hidden2label = torch.nn.Sequential(dense_layers)

        #initialize state
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (hidden_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers, 1, self.hidden_dim),
                torch.zeros(self.lstm_layers, 1, self.hidden_dim))

    def forward(self, data):
        x = data
        if not self.pre_trained:
            x = self.word_embeddings(data)
        lstm_out, self.hidden = self.lstm(
            x.view(len(x), 1, -1), self.hidden)
        label_space = self.hidden2label(lstm_out[-1])
        label_scores = F.log_softmax(label_space, dim=1)
        return label_scores


l2i = {'S': 0, 'D': 1, 'Q': 2, 'C': 3}

def train(X_train, y_train, lstm_layers, lstm_units, linear_layers, linear_units,
          learning_rate, L2_reg, epochs, emb_size):
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    model = StanceLSTM(lstm_layers, lstm_units, linear_layers, linear_units, len(l2i), emb_size).to(args.device)
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_reg)
    print("#Training")
    for epoch in range(epochs):
        avg_loss = 0.0
        for feature_vec, label in zip(X_train, y_train):
            # Clear stored gradient
            model.zero_grad()

            # Initialize hidden state
            model.hidden = model.init_hidden()

            # Prepare data
            data_in = torch.tensor([feature_vec], device=args.device)
            target = torch.tensor([label], device=args.device)

            # Make prediction
            pred = model(data_in)

            # Calculate loss
            loss = loss_func(pred, target)
            avg_loss += loss.item()

            loss.backward()  # Back propagate
            optimizer.step()  # Update parameters
        avg_loss /= len(X_train)
        print("Epoch: {0}\tavg_loss: {1}".format(epoch, avg_loss))
    return model


def test(model, X_test, y_test):
    print("#Testing")
    with torch.no_grad():
        labels_true = y_test
        labels_pred = []
        for vec in X_test:
            data_in = torch.tensor([vec], device=args.device)
            pred = model(data_in)
            predicted = torch.argmax(pred.data, dim=1)
            pred_val = predicted[0].item()
            labels_pred.append(pred_val)
        print(classification_report(labels_true, labels_pred))
        cm = sk.confusion_matrix(labels_true, labels_pred, labels=[0, 1, 2, 3])
        print("Confusion matrix:")
        print("  S  D  Q  C")
        print(cm)
        acc = sk.accuracy_score(labels_true, labels_pred)
        print("Accuracy: %.5f" % acc)
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # sdqc_acc = cm.diagonal()
        # f1 = sk.f1_score(labels_true, labels_pred, average='macro')
        # print("SDQC acc:", sdqc_acc)
        # print("F1-macro:", f1)


def run():
    X_train, X_test, y_train, y_test, EMB = data_loader.get_train_test_split()

    # Hyper parameters
    EPOCHS = [10, 30, 50, 100, 200]
    LSTM_LAYERS = [1, 2]
    LSTM_UNITS = [100, 200, 300]
    LINEAR_LAYERS = [1, 2, 3]
    LINEAR_UNITS = [100, 200, 300]
    LEARNING_RATE = [0.1, 0.01, 0.001]
    L2_REG = [0, 0.1, 0.01, 0.001]

    model = train(X_train, y_train, LSTM_LAYERS[0], LSTM_UNITS[0], LINEAR_LAYERS[0],
                  LINEAR_UNITS[0], LEARNING_RATE[0], L2_REG[0], EPOCHS[0], EMB)
    test(model, X_test, y_test)


run()
