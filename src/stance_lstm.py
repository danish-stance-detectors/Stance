import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import sklearn.metrics as sk
import numpy as np
from sklearn.metrics import classification_report
import collections
import argparse
import data_loader
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from sklearn.model_selection import cross_val_score, StratifiedKFold
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='Train and test LSTM model')
parser.add_argument('-c', '--cuda', dest='cuda', action='store_true', help='Enable CUDA')
args = parser.parse_args()
if args.cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

class StanceDataset(Dataset):
    def __init__(self, X, y):
        self.data = [(
            torch.tensor([x], device=args.device),
            torch.tensor([y], device=args.device)
        ) for x, y in zip(X, y)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class StanceLSTM(nn.Module):
    def __init__(self, lstm_layers, lstm_dim, hidden_layers, hidden_dim,
                 num_labels, emb_dim, device, batch_size, vocab_size=0, dropout=True, pre_trained=True):
        super(StanceLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.pre_trained = pre_trained
        self.emb_dim = emb_dim
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.device = device
        if not pre_trained:
            self.word_embeddings = nn.Embedding(vocab_size, emb_dim)

        # LSTM layer
        self.lstm = nn.LSTM(emb_dim, lstm_dim, lstm_layers, batch_first=True)

        # Linear layer
        dense_layers = collections.OrderedDict()
        dense_layers["lin0"] = torch.nn.Linear(lstm_dim, hidden_dim).to(device)
        dense_layers["rec0"] = torch.nn.ReLU().to(device)
        for i in range(hidden_layers - 1):
            dense_layers["lin%d" % (i + 1)] = torch.nn.Linear(hidden_dim, hidden_dim).to(device)
            dense_layers["rec%d" % (i + 1)] = torch.nn.ReLU().to(device)
        if dropout:
            dense_layers["drop"] = torch.nn.Dropout(p=0.5).to(device)
        dense_layers["lin%d" % hidden_layers] = torch.nn.Linear(hidden_dim, num_labels).to(device)
        self.hidden2label = torch.nn.Sequential(dense_layers).to(device)

        #initialize state
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (hidden_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers, self.batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.lstm_layers, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, data):
        x = pack_sequence(data)
        # if not self.pre_trained:
        #     x = self.word_embeddings(data)
        # lstm_out, self.hidden = self.lstm(
        #     x.view(len(x), 1, -1), self.hidden)
        x, _ = self.lstm(x, self.hidden)
        x = pad_packed_sequence(x, batch_first=True)
        # x = x.contiguous()
        x = x.view(-1, x.shape[2])
        # label_space = self.hidden2label(lstm_out[-1])
        x = self.hidden2label(x)
        x = F.log_softmax(x, dim=1)
        x = x.view(self.batch_size, self.emb_dim, self.num_labels)
        y_hat = x
        # label_scores = F.log_softmax(label_space, dim=1)
        return y_hat


l2i = {'S': 0, 'D': 1, 'Q': 2, 'C': 3}

def train(X_train, y_train, lstm_layers, lstm_units, linear_layers, linear_units,
          learning_rate, L2_reg, epochs, emb_size, dropout=False, validation_size=0.2):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size
    )
    params = {'batch_size': 32, 'shuffle': False}
    training_set = StanceDataset(X_train, y_train)
    training_generator = DataLoader(training_set, **params)
    validation_set = StanceDataset(X_val, y_val)
    validation_generator = DataLoader(validation_set, **params)
    dataloader = {
        'train': training_generator,
        'val': validation_generator
    }
    dataset_sizes = {
        'train': training_set.__len__(),
        'val': validation_set.__len__()
    }
    model = StanceLSTM(lstm_layers, lstm_units, linear_layers, linear_units,
                       len(l2i), emb_size, args.device, params['batch_size'], dropout=dropout).to(args.device)
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_reg)
    print("#Training")
    for epoch in range(epochs):
        print("*****Epoch {}*****".format(epoch))
        for phase, data_generator in dataloader.items():
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for X_batch, y_batch in data_generator:
                X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)

                # Clear stored gradient
                model.zero_grad()

                # Initialize hidden state
                model.hidden = model.init_hidden()

                # Prepare data
                # data_in = torch.tensor([feature_vec], device=args.device)
                # target = torch.tensor([label], device=args.device)

                with torch.set_grad_enabled(phase == 'train'):
                    # Make prediction
                    outputs = model(X_batch)
                    _, preds = torch.max(outputs, 1)
                    # Calculate loss
                    loss = loss_func(outputs, y_batch)

                    if phase == 'train':
                        loss.backward()  # Back propagate
                        optimizer.step()  # Update parameters

                running_loss += loss.item()
                running_corrects += torch.sum(preds == target.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            print('{:10} loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

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
    LSTM_UNITS = [100, 150, 200, 300]
    LINEAR_LAYERS = [1, 2, 3]
    LINEAR_UNITS = [100, 150, 200, 300]
    LEARNING_RATE = [0.1, 0.01, 0.001]
    L2_REG = [0, 0.1, 0.01, 0.001]
    
    model = train(X_train, y_train, LSTM_LAYERS[0], LSTM_UNITS[0], LINEAR_LAYERS[0],
                  LINEAR_UNITS[0], LEARNING_RATE[0], L2_REG[0], EPOCHS[0], EMB, dropout=True)
    test(model, X_test, y_test)

def grid_search():
    # X_train, X_test, y_train, y_test, EMB = data_loader.get_train_test_split()
    X, y, EMB = data_loader.get_features_and_labels()
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int64)
    print(X.shape)
    print(y.shape)
    print(y.mean())
    # Hyper parameters
    EPOCHS = [10, 30, 50, 100, 200]
    LSTM_LAYERS = [1, 2]
    LSTM_UNITS = [100, 150, 200, 300]
    LINEAR_LAYERS = [1, 2, 3]
    LINEAR_UNITS = [100, 150, 200, 300]
    LEARNING_RATE = [0.1, 0.01, 0.001]
    L2_REG = [0, 0.1, 0.01, 0.001]

    # model = train(X_train, y_train, LSTM_LAYERS[0], LSTM_UNITS[0], LINEAR_LAYERS[0],
    #               LINEAR_UNITS[0], LEARNING_RATE[0], L2_REG[0], EPOCHS[0], EMB, args.device, dropout=True)
    # lstm_layers, lstm_dim, hidden_layers, hidden_dim,
    # num_labels, emb_dim
    # model = StanceLSTM(1, 100, 1, 100,
    #                    4, EMB, args.device, dropout=True).to(args.device)

    auc = EpochScoring(scoring='accuracy', lower_is_better=False)

    net = NeuralNetClassifier(
        module=StanceLSTM,
        module__lstm_layers=1,
        module__lstm_dim=100,
        module__hidden_layers=1,
        module__hidden_dim=100,
        module__num_labels=4,
        module__emb_dim=EMB,
        module__device=args.device,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.SGD,
        optimizer__lr=0.01,
        optimizer__weight_decay=0.001,
        max_epochs=10,
        batch_size=1,
        device=args.device,
        callbacks=[auc],
    )

    net.fit(X, y)


run()

