# External imports
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, cross_val_predict, cross_validate, train_test_split
from hmmlearn import hmm
import sys
import argparse
import random

# Our imports
import data_loader
import hmm_data_loader
import model_stats

def main(argv):

    parser = argparse.ArgumentParser(description='Rumour veracity classification via multinomial hmm')
    parser.add_argument('-ftrain', '--file_train', default='../data/hmm/hmm_data_branch_time.csv', help='Train file. Tests on this if no test file is given.')
    parser.add_argument('-ftest', '--file_test', help='Test file')
    parser.add_argument('-sem_tr', '--sem_data_train', action='store_true', default=False, help='Read train in semeval data format')
    parser.add_argument('-sem_te', '--sem_data_test', action='store_true', default=False, help='Read test in semeval data format')
    parser.add_argument('-kf', '--k_folds', help='Number of folds to do in cross validation', type=int)
    parser.add_argument('-rs', '--restarts', help='Number of times to restart with new random state', type=int)
    parser.add_argument('-rand', '--rand_state', default=357, help='Specific random state', type=int)
    
    
    args = parser.parse_args(argv)

    if args.file_train:
        data = get_data(args.file_train, args.sem_data_train)
        y = [x[0] for x in data]
        X = [x[1] for x in data]
        
        if args.file_test:
            data_test = get_data(args.file_test, args.sem_data_test)
            y_test = [x[0] for x in data_test]
            X_test = [x[1] for x in data_test]

            print("%-10s%10s%10s" % ('state space', 'acc', 'f1'))
            for i in range(1, 16):
                clf = MSHMM(2, i).fit(X, y)
                predicts = clf.predict(X_test)
                _, acc, f1 = model_stats.cm_acc_f1(y_test, predicts)
                print("%-10s%10.2f%10.2f" % (i, acc, f1))


        else:
            if not args.k_folds:
                print("%-10s%10s%10s" % ('state space', 'acc', 'f1'))

            for i in range(1,16):
                if args.k_folds:
                    if args.restarts:
                        for r in random.sample(range(1, 1000), args.restarts):
                            cross_val(MSHMM(2, i), X, y, args.k_folds, args.rand_state, i)
                        print("\n")
                    else:
                        cross_val(MSHMM(2, i), X, y, args.k_folds, args.rand_state, i)
                else:
                    best_acc = 0.0
                    best_f1 = 0.0
                    best_rand_state = None
                    best_i = None

                    if args.restarts:
                        for r in random.sample(range(1, 1000), args.restarts):
                            acc, f1 = test(X, y, args.rand_state, i)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_acc = acc
                                best_rand_state = args.rand_state
                                best_i = i
                    else:
                        acc, f1 = test(X, y, args.rand_state, i)
                        
                        best_acc = acc
                        best_f1 = f1
                        best_rand_state = args.rand_state
                        best_i = i
                    
                    print("%-10s%-10s%10.2f%10.2f" % (best_i, best_rand_state, best_acc, best_f1))

def test(X, y, rand, i):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rand)
    clf = MSHMM(2, i).fit(X_train, y_train)
    predicts = clf.predict(X_test)
    
    _, acc, f1 = model_stats.cm_acc_f1(y_test, predicts)
    return acc, f1

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

# Returns data
def get_data(file_name, sem_data):
    if sem_data:
        data, _ = hmm_data_loader.get_semeval_hmm_data(filename=file_name)
        data = [(x[1], x[2]) for x in data]
    else:
        data, _ = hmm_data_loader.get_hmm_data(filename=file_name)
    
    return data

class MSHMM(BaseEstimator):

    def __init__(self, space_size, components):
        self.space_size = space_size
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
            X_len = [int(len(x) / self.space_size) for x in X_list]
            X_tmp = np.array(self.flatten(X_list))
            X_tmp = X_tmp.reshape(int(len(X_tmp) / self.space_size), self.space_size)
            
            self.models[y_k] = hmm.GaussianHMM(n_components=self.components).fit(X_tmp, lengths=X_len)

        return self
    
    def predict(self, X):
        predicts = []
        for branch in X:
            # get len of branch and reshape it
            b_len = int(len(branch) / self.space_size)
            branch = np.array(branch)
            branch = branch.reshape(int(len(branch) / self.space_size), self.space_size)
            best_y = -1
            best_score = None
            for y, model in self.models.items():
                score = model.score(branch, lengths=[b_len])
                if best_score is None or score > best_score:
                    best_y = y
                    best_score = score
            
            predicts.append(best_y)
        
        return predicts

if __name__ == "__main__":
    main(sys.argv[1:])