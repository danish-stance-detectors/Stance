
from sklearn.base import BaseEstimator

class two_step(BaseEstimator):
    """ Draft of two step classifier """
    import numpy as np
    def __init__(self, comment_model, other_model):
        self.m1 = comment_model
        self.m2 = other_model
    
    def fit(self, X, y, **kwargs):
        comment_labels = []
        other_vec = []
        other_label = []
        # Build comment binary label list and other lists
        for i in range(len(y)):
            if y[i] == 3:
                comments.append(i, X[i])
                comment_labels.append(1)
            else:
                comment_labels.append(0)
                other_vec.append(X[i])
                other_label.append(y[i])

        #Train model 1 one to guess comments
        self.m1.fit(X, comment_labels)

        #Train model 2 to guess others (SDQ)
        self.m2.fit(other_vec, other_label)

    def predict(self, X):
        #Get all comment predictions
        comment_pred = self.m1.predict(X)

        # for each which is predicted as not comment (0), store it and its original idx
        non_comments = []
        for i in range(comment_pred):
            if comment_pred[i] == 0:
                non_comments.append((i, comment_pred[i]))
        
        # predict others
        other_pred = self.m2.predict([x[1] for x in non_comments])

        # insert predictions from model two into 0 preds from m1
        for (idx, pred) in enumerate(other_pred):
            org_idx = non_comments[idx][0] # original idx in list
            comment_pred[org_idx] = pred
        
        #return
        return np.array(comment_pred)