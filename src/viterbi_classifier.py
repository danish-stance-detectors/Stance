import data_loader
import hmm_data_loader
import model_stats
import sys
import argparse
from sklearn.base import BaseEstimator
import collections

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, cross_val_predict, cross_validate, train_test_split
from hmmlearn import hmm
from joblib import dump, load

import data_utils


def main(argv):
    
    parser = argparse.ArgumentParser(description='Rumour veracity classification via hmm')
    parser.add_argument('-loo', '--LeaveOneOut', default=False, dest='loo', action='store_true', help='Do leave one out testing on pheme dataset')
    parser.add_argument('-sv', '--save', default=False, action='store_true', help='Save model')
    parser.add_argument('-ph', '--pheme', default=False, dest='pheme', action='store_true', help='Train and test on pheme dataset')
    parser.add_argument('-pd', '--print_dist', default=False, dest='print_dist', action='store_true', help='print event distribution')
    parser.add_argument('-da', '--danish', default=False, action='store_true', help='Train and test solely on danish rumour data')
    parser.add_argument('-cmt', '--comment', default=False, action='store_true', help='Use hmm features from comment trees. Only has effect for danish data.')
    parser.add_argument('-fdk', '--data file path danish', dest='file_dk', default='../data/hmm/preprocessed_hmm.csv', help='Danish rumour data')
    parser.add_argument('-fen', '--data file path english', dest='file_en', default='../data/hmm/semeval_rumours_train_pheme.csv', help='English rumour data')
    parser.add_argument('-o', '--out file path', dest='outfile', default='../data/hmm/hmm_data.csv', help='Output filer holding preprocessed data')
    parser.add_argument('-ml', '--minumum branch length', dest='min_len', default=1, help='Minimum branch lengths included in training')
    parser.add_argument('-en_da', '--eng_train_da_test', default=False, dest='en_da', action='store_true', help='Train on english data and test on danish data')
    parser.add_argument('-mix', '--mix_train_test', default=False, dest='mix', action='store_true', help='Train and test on mix of pheme and danish data')
    parser.add_argument('-sub', '--sub_sample', default=False, action='store_true', help='Sub sample when training on single dataset')    
    parser.add_argument('-dis', '--distribution_voter', default=False, action='store_true', help='Use a dummy distribution voter as model')


    args = parser.parse_args(argv)
    
    if args.loo:
        Loo_event_test(args.file_en, int(args.min_len), args.print_dist)
    elif args.danish:
        train_test_danish(args.distribution_voter, file_name=args.file_dk, min_length=int(args.min_len))
    elif args.en_da:
        train_eng_test_danish(args.file_en, args.file_dk, args.distribution_voter, int(args.min_len))
    elif args.pheme:
        train_test_pheme(args.file_en, int(args.min_len))
    elif args.mix:
        train_test_mix(args.file_en, args.file_dk, int(args.min_len))
    elif args.save:
        save_model(args.file_dk, args.outfile)

# partition data into events
def loadEvents(data, print_dist=False, min_len=1):
    events = dict()
    for event, truth, vec in data:
        if event not in events:
            events[event] = []
        
        events[event].append((truth, vec))
    
    if print_dist:
        print("\n")
        print("Conversations per event:")
        for k, v in events.items():
            true_cnt = len([x for x in v if x[0] == 1])
            false_cnt = len(v) - true_cnt
            print("Event: {}\t conversations: {} with True {} / False {}".format(k, len(v), true_cnt, false_cnt))

        print("\n")
        
        print("Conversations per event with atleast {} in len:".format(min_len))
        for k, v in events.items():
            tmp_v = [x for x in v if len(x[1]) >= min_len]
            true_cnt = len([x for x in tmp_v if x[0] == 1])
            false_cnt = len(tmp_v) - true_cnt
            print("Event: {}\t conversations: {} with True {} / False {}".format(k, len(tmp_v), true_cnt, false_cnt))
    
        print("\n")

    return events

flatten = lambda l: [item for sublist in l for item in sublist]

# partition data dependin on label
def get_x_y(data, min_len):
    data = [x for x in data if len(x[1]) >= min_len]
    data_true = [x for x in data if x[0] == 1]
    data_false = [x for x in data if x[0] == 0]

    assert (len(data_true) + len(data_false)) == len(data), "data lost in get_x_y true false partitioning, {} while org was {}".format(len(data_true) + len(data_false), len(data))

    y_t = [x[0] for x in data_true]
    X_t = [x[1] for x in data_true]
    y_f = [x[0] for x in data_false]
    X_f = [x[1] for x in data_false]

    return X_t, y_t, X_f, y_f

class HMM(BaseEstimator):

    def __init__(self, components):
        self.components = components
        self.flatten = lambda l: [item for sublist in l for item in sublist]

    def fit(self, X, y):
        dict_y = dict()

        # Partition data in labels
        for i in range(len(X)):
            if y[i] not in dict_y:
                dict_y[y[i]] = []
            
            dict_y[y[i]].append(X[i])
        
        self.models = dict()

        # Make and fit model for each label
        for y_k, X_list in dict_y.items():
            X_len = [len(x) for x in X_list]
            X_tmp = np.array(flatten(X_list)).reshape(-1, 1)
            
            self.models[y_k] = hmm.GaussianHMM(n_components=self.components).fit(X_tmp, lengths=X_len)
        
        return self
        
    def predict(self, X):
        predicts = []
        for branch in X:
            # get len of branch and reshape it
            b_len = len(branch)
            branch = np.array(branch).reshape(-1, 1)
            best_y = -1
            best_score = None
            for y, model in self.models.items():
                score = model.score(branch, lengths=[b_len])
                if best_score is None or score > best_score:
                    best_y = y
                    best_score = score
            
            predicts.append(best_y)
        
        return predicts

class DistributionVoter(BaseEstimator):
    """
    Stores distributions of stance labels for each rumour truth in training data.
    Chooses the y whose overall distributions is closest to the branch.
    """
    def fit(self, X, y):
        dist_dict = dict()
        for i in range(len(X)):
            dist = self.get_distribution(X[i])
            
            if y[i] not in dist_dict:
                dist_dict[y[i]] = np.zeros(4)
            dist_dict[y[i]] += dist

        for i in range(len(dist_dict)):
            dist_dict[i] /= dist_dict[i].sum()

        self.dist_dict = dist_dict

        return self
    
    def predict(self, X):
        distributions = [self.get_distribution(x) for x in X]
        predicts = []
        for distribution in distributions:
            min_dist = None
            best_y = None
            for k, v in self.dist_dict.items():
                distance = self.get_distance(distribution, v)
                if min_dist is None or distance < min_dist:
                    min_dist = distance
                    best_y = k

            predicts.append(best_y)

        return predicts
    
    def get_distribution(self, x):
        dist = np.zeros(4)
        for stance in x:
            dist[int(stance)] += 1

        return dist

    def get_distance(self, dist_a, dist_b):
        return np.linalg.norm(dist_a-dist_b)
# Leaves one event out for testing and trains on the others
# does so for each event
def Loo_event_test(file_en, min_length, print_distribution):
    # load data
    data_train, _ = hmm_data_loader.get_semeval_hmm_data(filename=file_en)
    data_train_y_X = [(x[1], x[2]) for x in data_train]

    events = loadEvents(data_train, print_dist=print_distribution, min_len=min_length)
    event_list = [(k,v) for k,v in events.items()]
    
    print("%-20s%10s%10s%10s" % ('event', 'components', 'accuracy', 'f1'))
    for i in range(len(event_list)):
        
        test_event, test_vec = event_list[i]

        train = [vec for e, vec in event_list if e != test_event]
        train = flatten(train)
        y_train = [x[0] for x in train]
        X_train = [x[1] for x in train]
       
        test_vec = [x for x in test_vec if len(x[1]) >= min_length]
        # partition test data and y
        y_test = [x[0] for x in test_vec] 
        X_test = [x[1] for x in test_vec]
        X_test_len = [len(x) for x in X_test]
        
        for s in range(1, 16):
            best_acc = 0.0
            best_f1 = 0.0

            # try out different random configurations
            for c in range(1):
                clf = HMM(s).fit(X_train, y_train)
                predicts = clf.predict(X_test)

                assert len(y_test) == len(predicts), "The length of y_test does not match number of predictions"

                # print result
                _, acc_t, f1_t = model_stats.cm_acc_f1(y_test, predicts)
                
                # save results from model with best f1 score
                if f1_t > best_f1:
                    best_acc = acc_t
                    best_f1 = f1_t
            print("%-20s%-10s%10.2f%10.2f" % (test_event, s, best_acc, best_f1))

def train_test_mix(file_en, file_da, min_length=1):    
    # load data
    danish_data, emb_size_da = hmm_data_loader.get_hmm_data(filename=file_da)
    X_da = [x[1] for x in danish_data]
    y_da = [x[0] for x in danish_data]
    
    data_train, _ = hmm_data_loader.get_semeval_hmm_data(filename=file_en)
    y_en = [x[1] for x in data_train]
    X_en = [x[2] for x in data_train]

    X_en.extend(X_da)
    y_en.extend(y_da)

    for s in range(1,16):
        
        cross_val(HMM(s), X_en, y_en, 3, 42, s)

def train_eng_test_danish(file_en, file_da, distribution_voter, min_length=1):    
    # load data
    danish_data, emb_size_da = hmm_data_loader.get_hmm_data(filename=file_da)
    danish_data_X = [x[1] for x in danish_data]
    danish_data_y = [x[0] for x in danish_data]
    
    data_train, _ = hmm_data_loader.get_semeval_hmm_data(filename=file_en)
    y_train = [x[1] for x in data_train]
    X_train = [x[2] for x in data_train]

    print("%-20s%10s%10s%10s" % ('event', 'components', 'accuracy', 'f1'))

    if distribution_voter:
        clf = DistributionVoter().fit(X_train, y_train)
        predicts = clf.predict(danish_data_X)
        _, acc, f1 = model_stats.cm_acc_f1(danish_data_y, predicts)
        print("%-20s%-10s%10.2f%10.2f" % ('danish', '-', acc, f1))
    else:
        best_acc = 0.0
        best_f1 = 0.0
        best_s = None
        for s in range(1,16):
            # try out different random configurations
            for c in range(1):
                # test on danish data
                clf = HMM(s).fit(X_train, y_train)
                da_predicts = clf.predict(danish_data_X)

            cm, acc_t_da, f1_t_da = model_stats.cm_acc_f1(danish_data_y, da_predicts)
            # print(cm)
            
            if f1_t_da > best_f1:
                best_acc = acc_t_da
                best_f1 = f1_t_da
                best_s = s
        print("%-20s%-10s%10.2f%10.2f" % ('danish', best_s, best_acc, best_f1))

def train_test_pheme(file_name, min_length=1):
    
    print("Testing on english data")

    data_train, _ = hmm_data_loader.get_semeval_hmm_data(filename=file_name)
    data_train = [x for x in data_train if len(x[2]) >= min_length]
    data_train_y = [x[1] for x in data_train]
    data_train_X = [x[2] for x in data_train]
    
    for s in range(1,16):
        
        cross_val(HMM(s), data_train_X, data_train_y, 3, 42, s)    

def train_test_danish(distribution_voter, file_name='../data/hmm/hmm_data_comment_trees.csv', min_length=1):


    print("Testing on danish data")

    danish_data, emb_size_da = hmm_data_loader.get_hmm_data(filename=file_name)

    X = [x[1] for x in danish_data]
    y = [x[0] for x in danish_data]
    
    if distribution_voter:
        cross_val(DistributionVoter(), X, y, 3, 42, 0)
    else:
        for s in range(1,16):
            best_acc = 0.0
            best_f1 = 0.0

            cross_val(HMM(s), X, y, 3, 42, s)    
    
def save_model(file_name, out_file):
    danish_data, emb_size_da = hmm_data_loader.get_hmm_data(filename=file_name)

    danish_data_X = [x[1] for x in danish_data]
    danish_data_y = [x[0] for x in danish_data]

    X_train, X_test, y_train, y_test = train_test_split(
        danish_data_X, danish_data_y, test_size=0.20, random_state=42, stratify=danish_data_y)
    
    best_model = None
    train_y_x = list(zip(y_train, X_train))
    for s in range(1, 16):

        best_acc = 0.0
        best_f1 = 0.0
        clf = HMM(s).fit(X_train, y_train)
        da_predicts = clf.predict(X_test)
    
        _, acc_t_da, f1_t_da = model_stats.cm_acc_f1(y_test, da_predicts)
        if f1_t_da > best_f1:
            best_acc = acc_t_da
            best_f1 = f1_t_da
            best_model = clf

    dump(best_model, out_file)
    
def cross_val(clf, X, y, folds, rand, i):
    scoring = [
        'f1_macro',
        'accuracy'
    ]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand)
    scores = cross_validate(clf, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)
    s = "%-20s %-4s %-5s %s %0.2f (+/- %0.2f) %s %0.2f (+/- %0.2f)" \
        % (i, 'rand', rand, 'f1', scores['test_f1_macro'].mean(), scores['test_f1_macro'].std() * 2,
        'acc', scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2)
    print(s)
    
if __name__ == "__main__":
    main(sys.argv[1:])