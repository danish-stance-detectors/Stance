import data_loader
import model_stats
import numpy as np
from sklearn.model_selection import train_test_split

#data, emb_size_max = data_loader.get_hmm_data(filename='../data/hmm/preprocessed_hmm_no_branch.csv')
data, emb_size_max = data_loader.get_hmm_data()

y = [x[0] for x in data]
X = [x[1] for x in data]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)


start_probs = { 
    0 : y_train.count(0) / len(y_train), 
    1 : y_train.count(1) / len(y_train)
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

for (b_i, branch) in enumerate(X_train):
    for (i, label) in enumerate(branch):
        if i > 0:
            transition_probs[branch[i-1]][label] += 1
        emmision_probs[y_train[b_i]][label] += 1
        emmission_prob_count[y_train[b_i]] += 1    

for (label, t_prob) in transition_probs.items():
    total = sum([count for (label, count) in t_prob.items()])
    for i in range(4):
        t_prob[i] /= total

for (label, em_prob) in emmision_probs.items():
    for i in range(4):
        em_prob[i] /= emmission_prob_count[label]

print(start_probs)

print("transition probabilities\n")
for (label, t_prob) in transition_probs.items():
    print("{}\t{}".format(label, t_prob))

print("Emission probabilities\n")
for (label, em_prob) in emmision_probs.items():
    print("{}\t{}".format(label, em_prob))


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

single_start_p = { 0 : 1, 1 : 1}

predicts = []
for branch in X_test:
    pred_f_prob = viterbi([0], branch, start_probs, transition_probs, emmision_probs)
    pred_t_prob = viterbi([1], branch, start_probs, transition_probs, emmision_probs)

    predicts.append(int(pred_t_prob > pred_f_prob))

model_stats.print_confusion_matrix(y_test, predicts, [0,1])