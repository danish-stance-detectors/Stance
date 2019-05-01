import data_loader
import model_stats
import numpy as np
from sklearn.model_selection import train_test_split
from hmmlearn import hmm

# data, emb_size_max = data_loader.get_hmm_data(filename='../data/hmm/semeval_rumours.csv')
# semeval_dev_data, _ = data_loader.get_hmm_data(filename='../data/hmm/semeval_rumours_dev.csv')
# semeval_dungs, _ = data_loader.get_hmm_data(filename='../data/hmm/semeval_rumours_dungs_unv.csv')
danish_data, emb_size_da = data_loader.get_hmm_data(filename='../data/hmm/preprocessed_hmm_no_branch.csv')

data_train, _ = data_loader.get_semeval_hmm_data()
#data, emb_size_max = data_loader.get_hmm_data()
#data.extend(danish_data)

def calculate_probabilities(X, y, print_p=False):
    start_probs = { 
        0 : y.count(0) / len(y), 
        1 : y.count(1) / len(y)
    }

    transition_probs = {
        0 : { 0: 0, 1: 0, 2: 0, 3 : 0},
        1 : { 0: 0, 1: 0, 2: 0, 3 : 0},
        2 : { 0: 0, 1: 0, 2: 0, 3 : 0},
        3 : { 0: 0, 1: 0, 2: 0, 3 : 0}
    }

    emmision_probs = {
        0 : { 0: 0, 1: 0, 2: 0, 3 : 0},
        1 : { 0: 0, 1: 0, 2: 0, 3 : 0}
    }

    emmission_prob_count = { 0: 0, 1: 1}

    for (b_i, branch) in enumerate(X):
        for (i, label) in enumerate(branch):
            if i > 0:
                transition_probs[branch[i-1]][label] += 1
            emmision_probs[y[b_i]][label] += 1
            emmission_prob_count[y[b_i]] += 1    

    for (label, t_prob) in transition_probs.items():
        total = sum([count for (label, count) in t_prob.items()])
        for i in range(4):
            t_prob[i] /= total

    for (label, em_prob) in emmision_probs.items():
        for i in range(4):
            em_prob[i] /= emmission_prob_count[label]
    if print_p:
        print(start_probs)

        print("transition probabilities\n")
        for (label, t_prob) in transition_probs.items():
            print("{}\t{}".format(label, t_prob))

        print("Emission probabilities\n")
        for (label, em_prob) in emmision_probs.items():
            print("{}\t{}".format(label, em_prob))
    
    return start_probs, transition_probs, emmision_probs

# Code heavily inspired by wikipedia impl for viterbi
def viterbi(hid_states, obs, start_probs, trans_probs, em_probs):
    V = [{}]
    # v_ = V[0]
    # o_ = em_probs[0][3]
    for st in hid_states:
        V[0][st] = { 'prob' : start_probs[st] * em_probs[st][obs[0]],
                     'prev' : None }
        
    for t in range(1, len(obs)):
        V.append({})

        for st in hid_states:
            max_tr_prob = V[t-1][hid_states[0]]['prob']*trans_probs[hid_states[0]][st]
            prev_st_selected = hid_states[0]
            for prev_st in hid_states[1:]:
                tr_prob = V[t-1][prev_st]['prob']*trans_probs[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                    
            max_prob = max_tr_prob * em_probs[st][obs[t]]
            V[t][st] = {'prob': max_prob, 'prev': prev_st_selected}

    opt = []
    max_prob = max([value['prob'] for value in V[-1].values()])
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data['prob'] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]['prev'])
        previous = V[t + 1][previous]['prev']
    
    #print("The steps of states are {} with highest probability of {}".format(opt, max_prob))
    return max_prob

events = dict()
for event, truth, vec in data_train:
    if event not in events:
        events[event] = []
    
    events[event].append((truth, vec))

print("\n")
print("Conversations per event:")
for k, v in events.items():
    print("Event: {}\t conversations: {}".format(k, len(v)))

print("\n")
def get_x_y(data, min_len):
    data = [x for x in data if len(x[1]) >= min_len]
    data_true = [x for x in data if x[0] == 1]
    data_false = [x for x in data if x[0] == 0]
    y_t = [x[0] for x in data_true]
    X_t = [x[1] for x in data_true]
    y_f = [x[0] for x in data_false]
    X_f = [x[1] for x in data_false]

    return X_t, y_t, X_f, y_f

flatten = lambda l: [item for sublist in l for item in sublist]

def Loo_event_test(events):
    event_list = [(k,v) for k,v in events.items()]
    
    print("%-20s%5s%5s" % ('event', 'accuracy', 'f1'))
    for i in range(len(event_list)):
        
        test_event, test_vec = event_list[i]

        train = [vec for e, vec in event_list if e != test_event]
        train = [item for sublist in train for item in sublist]
        
        # True and false training data
        X_t, y_t, X_f, y_f = get_x_y(train, 1)


        # print(X_t)

        # lengths
        X_t_len = [len(x) for x in X_t]
        X_f_len = [len(x) for x in X_f]
        
        # X_train = np.reshape(X_train, (-1, 1))
        X_t = np.array(flatten(X_t)).reshape(-1, 1)#np.reshape(X_t, (-1, 1))
        X_f = np.array(flatten(X_f)).reshape(-1, 1)

        clf_true = hmm.GaussianHMM(n_components=1, n_iter=100, random_state=42).fit(X_t, lengths=X_t_len)
        clf_false = hmm.GaussianHMM(n_components=1, n_iter=100, random_state=42).fit(X_f, lengths=X_f_len)

        y_test = [x[0] for x in test_vec]
        X_test = [x[1] for x in test_vec]
        X_test_len = [len(x) for x in X_test]
        
        predicts = []
        # X_test = np.array(flatten(X_test)).reshape(-1, 1)
        for branch in X_test:
            b_len = len(branch)
            branch = np.array(branch).reshape(-1, 1)

            prop_t = clf_true.score(branch, lengths=[b_len])
            prop_f = clf_false.score(branch, lengths=[b_len])

            # print(prop_t)
            
            # print(prop_f)
            # prop_t = prop_t_a[np.argmax(prop_t_a[len(prop_t_a)-1, :])]
            # prop_f = prop_f_a[np.argmax(prop_f_a[len(prop_f_a)-1, :])]
            predicts.append(int(prop_t > prop_f))
            # print(np.argmax(prop_t_a[b_len-1]))
            # print(np.argmax(prop_f_a[b_len-1]))

        # prop_t, predicts_t = clf_true.decode(np.array(X_test[0]).reshape(-1, 1), lengths=X_test_len[0])#, lengths=X_test_len)
        # prop_f, predicts_f = clf_false.decode(X_test, lengths=X_test_len)
        
        # print(prop_t)

        # print(len(predicts_t))
        # print(len(y_test))
        # print(predicts_t)
        # print(predicts_f)
        
        # predicts = []
        # for branch in X_test:
        #     pred_f_prob = viterbi([0], branch, start_probs, transition_probs, emmision_probs)
        #     pred_t_prob = viterbi([1], branch, start_probs, transition_probs, emmision_probs)
        #     predicts.append(int(pred_t_prob > pred_f_prob))
        _, acc_t, f1_t = model_stats.cm_acc_f1(y_test, predicts, [0,1])
        # _, acc_f, f1_f = model_stats.cm_acc_f1(y_test, predicts_f, [0,1])
        print("%-20s%-10.2f%-10.2f" % (test_event, acc_t, f1_t))
        # print("%-20s%-10s%-10.2f%-10.2f" % (test_event, "false", acc_f, f1_f))
        # print(train)

Loo_event_test(events)

# X_train, y_train = get_x_y(data_train, 1)
# X_dev, y_dev = get_x_y(data_dev, 1)
# print(len(X_train))

# start_probs, transition_probs, emmision_probs = calculate_probabilities(X_train, y_train)

# predicts = []
# for branch in X_dev:
#     pred_f_prob = viterbi([0], branch, start_probs, transition_probs, emmision_probs)
#     pred_t_prob = viterbi([1], branch, start_probs, transition_probs, emmision_probs)
#     #print("Probability for false: {}\t true: {}".format(pred_f_prob, pred_t_prob))
#     predicts.append(int(pred_t_prob > pred_f_prob))

# model_stats.print_confusion_matrix(y_dev, predicts, [0,1])

# print("semeval training, danish test:\n")
# y_danish = [x[0] for x in danish_data]
# X_danish = [x[1] for x in danish_data]

# danish_predicts = []

# for branch in X_danish:
#     pred_f_prob = viterbi([0], branch, start_probs, transition_probs, emmision_probs)
#     pred_t_prob = viterbi([1], branch, start_probs, transition_probs, emmision_probs)

#     danish_predicts.append(int(pred_t_prob > pred_f_prob))

# model_stats.print_confusion_matrix(y_danish, danish_predicts, [0,1])