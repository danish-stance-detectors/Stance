import data_loader
import model_stats
import numpy as np
from sklearn.model_selection import train_test_split
from hmmlearn import hmm

data, emb_size_max = data_loader.get_hmm_data()

y = [x[0] for x in data]
X = [x[1] for x in data]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

X_train_lengts = [len(x) for x in X_train]

train_sample_count = len(X_train_lengts)

# print(X_train)
# X_train = np.concatenate(X_train)
# print(X_train)
# X_train = X_train.reshape(-1, 1)
# print(X_train)

hmm_model = hmm.GaussianHMM(n_components=2)

for branch in X_train:
    samples = len(branch)
    branch_arr = np.asarray(branch).reshape(1,-1)
    print(branch_arr)
    hmm_model = hmm_model.fit(branch_arr, [samples,1])

# hmm_model.fit(X_train, lengths=X_train_lengts)
pred = []
for test_branch in X_test:
    pred.append(hmm_model.predict(test_branch, len(test_branch)))


X_test_lengths = [len(x) for x in X_test]

test_sample_count = len(X_test_lengths)

X_test = np.concatenate(X_test)
# print(len(X_test))
X_test = X_test.reshape(-1, 1)
# print(len(X_test))

# print(X_test_lengths)
# predicts = []
# for xtest in X_test:
#     test = np.concatenate(xtest).reshape(-1, 1)
#     predicts.append(hmm_model.predict(test))

predicts = hmm_model.predict(X_test, lengths=X_test_lengths)
# print(len(predicts.tolist()))
# print(len(y_test))
model_stats.print_confusion_matrix(y_test, pred, [0,1])
#print(predicts)