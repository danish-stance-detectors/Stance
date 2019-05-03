import data_loader
import hmm_data_loader
import model_stats
import sys
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from hmmlearn import hmm


def main(argv):
    
    parser = argparse.ArgumentParser(description='Rumour veracity classification via hmm')
    parser.add_argument('-loo', '--LeaveOneOut', default=False, dest='loo', action='store_true', help='Do leave one out testing on pheme dataset')
    parser.add_argument('-da', '--danish', default=False, action='store_true', help='Train and test solely on danish rumour data')
    parser.add_argument('-cmt', '--comment', default=False, action='store_true', help='Use hmm features from comment trees. Only has effect for danish data.')
    parser.add_argument('-f', '--data file path', dest='file', default='../data/hmm/hmm_data_comment_trees.csv', help='Input folder holding annotated data')
    parser.add_argument('-o', '--out file path', dest='outfile', default='../data/hmm/hmm_data.csv', help='Output filer holding preprocessed data')
    args = parser.parse_args(argv)
    
    if args.loo:
        Loo_event_test(args.file)
    elif args.danish:
        train_test_danish(file_name=args.file)

# partition data into events
def loadEvents(data, print_dist=False):
    events = dict()
    for event, truth, vec in data:
        if event not in events:
            events[event] = []
        
        events[event].append((truth, vec))
    
    if print_dist:
        print("\n")
        print("Conversations per event:")
        for k, v in events.items():
            print("Event: {}\t conversations: {}".format(k, len(v)))

        print("\n")
    
    return events

# partition data dependin on label
def get_x_y(data, min_len):
    data = [x for x in data if len(x[1]) >= min_len]
    data_true = [x for x in data if x[0] == 1]
    data_false = [x for x in data if x[0] == 0]
    y_t = [x[0] for x in data_true]
    X_t = [x[1] for x in data_true]
    y_f = [x[0] for x in data_false]
    X_f = [x[1] for x in data_false]

    return X_t, y_t, X_f, y_f

# flatten lambda to pass to hmm
flatten = lambda l: [item for sublist in l for item in sublist]

def apply_random_prob(clf, components):
    start_prob = np.random.rand(components)
    start_prob /= start_prob.sum()

    trans_prob = np.random.rand(components, components)
    trans_prob /= trans_prob.sum()

    emission_prob = np.random.rand(components, 4)
    emission_prob /= emission_prob.sum()

    # Check that probability values sum to (nearly) 1
    assert start_prob.sum() - 1 < 1e-4, 'start probability does not sum to 1, but {}'.format(start_prob.sum())
    assert trans_prob.sum() - 1 < 1e-4, 'transition probability does not sum to 1, but {}'.format(trans_prob.sum())
    assert emission_prob.sum() - 1 < 1e-4, 'emission probability does not sum to 1, but {}'.format(emission_prob.sum())

    clf.startprob_ = start_prob
    clf.transmat_ = trans_prob
    clf.emissionprob_ = emission_prob

def train_models(data, min_len=10, iter=10, components=1, init_random=False):
    # True and false training data
    X_t, y_t, X_f, y_f = get_x_y(data, min_len)
    
    # lengths of each branch
    X_t_len = [len(x) for x in X_t]
    X_f_len = [len(x) for x in X_f]
    
    # reshape to flat arrays
    X_t = np.array(flatten(X_t)).reshape(-1, 1)
    X_f = np.array(flatten(X_f)).reshape(-1, 1)

    # Init models
    clf_true = hmm.GaussianHMM(n_components=components)
    clf_false = hmm.GaussianHMM(n_components=components)

    # If init random, initialize and apply random start, emission and transition probabilities
    if init_random:
        apply_random_prob(clf_true, components)
        apply_random_prob(clf_false, components)
    
    # fit models
    clf_true = clf_true.fit(X_t, lengths=X_t_len)
    clf_false = clf_false.fit(X_f, lengths=X_f_len)

    return clf_true, clf_false

def predict(X, clf_true, clf_false):
    predicts = []
    for branch in X:
        # get len of branch and reshape it
        b_len = len(branch)
        branch = np.array(branch).reshape(-1, 1)

        # score branch on each model
        prop_t = clf_true.score(branch, lengths=[b_len])
        prop_f = clf_false.score(branch, lengths=[b_len])

        # if true model has higher probability, append 1 otherwise 0
        predicts.append(int(prop_t > prop_f))
    return predicts

# Leaves one event out for testing and trains on the others
# does so for each event
def Loo_event_test():
    
    # load data
    danish_data, emb_size_da = hmm_data_loader.get_hmm_data(filename='../data/hmm/hmm_data_comment_trees.csv')
    danish_data_X = [x[1] for x in danish_data]
    danish_data_y = [x[0] for x in danish_data]
    
    data_train, _ = hmm_data_loader.get_semeval_hmm_data(filename='../data/hmm/semeval_rumours_train_pheme.csv')
    data_train_y_X = [(x[1], x[2]) for x in data_train]

    events = loadEvents(data_train)
    event_list = [(k,v) for k,v in events.items()]
    
    print("%-20s%10s%10s%10s" % ('event', 'components', 'accuracy', 'f1'))
    for i in range(len(event_list)):
        
        for s in range(1, 16):
            best_acc = 0.0
            best_f1 = 0.0

            # try out different random configurations
            for c in range(1):
                test_event, test_vec = event_list[i]

                train = [vec for e, vec in event_list if e != test_event]
                train = flatten(train)
                
                clf_true, clf_false = train_models(train, components=s, init_random=True)

                # partition test data and y
                y_test = [x[0] for x in test_vec]
                X_test = [x[1] for x in test_vec]
                X_test_len = [len(x) for x in X_test]
                
                predicts = predict(X_test, clf_true, clf_false)

                # print result
                _, acc_t, f1_t = model_stats.cm_acc_f1(y_test, predicts)
                
                # save results from model with best f1 score
                if f1_t > best_f1:
                    best_acc = acc_t
                    best_f1 = f1_t
            print("%-20s%-10s%10.2f%10.2f" % (test_event, s, best_acc, best_f1))
    
    print("Testing on danish data")
    for s in range(1,16):
        best_acc = 0.0
        best_f1 = 0.0

        # try out different random configurations
        for c in range(1):
            # test on danish data
            clf_true, clf_false = train_models(data_train_y_X, components=s, init_random=True)
            da_predicts = predict(danish_data_X, clf_true, clf_false)

            _, acc_t_da, f1_t_da = model_stats.cm_acc_f1(danish_data_y, da_predicts)
            
            if f1_t_da > best_f1:
                best_acc = acc_t_da
                best_f1 = f1_t_da
        print("%-20s%-10s%10.2f%10.2f" % ('danish', s, best_acc, best_f1))

def train_test_danish(file_name='../data/hmm/hmm_data_comment_trees.csv'):

    print("Testing on danish data")

    danish_data, emb_size_da = hmm_data_loader.get_hmm_data(filename=file_name)

    danish_data_X = [x[1] for x in danish_data]
    danish_data_y = [x[0] for x in danish_data]
    
    X_train, X_test, y_train, y_test = train_test_split(
        danish_data_X, danish_data_y, test_size=0.2, random_state=42, stratify=danish_data_y)
    
    train_y_x = list(zip(y_train, X_train))

    for s in range(1,16):
        best_acc = 0.0
        best_f1 = 0.0

        # try out different random configurations
        for c in range(1):
            # test on danish data
            clf_true, clf_false = train_models(train_y_x, components=s, init_random=True, min_len=1)
            da_predicts = predict(X_test, clf_true, clf_false)

            _, acc_t_da, f1_t_da = model_stats.cm_acc_f1(y_test, da_predicts)
            
            if f1_t_da > best_f1:
                best_acc = acc_t_da
                best_f1 = f1_t_da
        print("%-20s%-10s%10.2f%10.2f" % ('danish', s, best_acc, best_f1))      
    
    
if __name__ == "__main__":
    main(sys.argv[1:])