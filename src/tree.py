from sklearn import tree
import graphviz
import collections
from src import data_loader
import numpy as np
import sys
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold

filename = '../data/preprocessed/PP_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300.csv'
# X_train, X_test, y_train, y_test, _, feature_mapping = data_loader.get_train_test_split(filename)
X, y, _, feature_mapping = data_loader.get_features_and_labels(filename)
config = data_loader.get_features(reddit=False, most_freq=False, pos=False, lexicon=False)
# config['text'] = True

X = data_loader.select_features(X, feature_mapping, config)

# print(len(X[0]))
# X = VarianceThreshold(threshold=0.01).fit_transform(X)
# print(len(X[0]))

clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  # min_samples_split=0.02,
                                  # min_samples_leaf=10,
                                  class_weight='balanced'
                                  )
feature_names = []
feature_names.extend(['period', 'e_mark', 'q_mark', 'hasTripDot', 'url_count', 'tripDotCount', 'q_mark_count',
                      'e_mark_count', 'cap_ratio', 'txt_len', 'tokens_len', 'avg_word_len', 'cap_sequence_max_len'])
# feature_names.extend(['swear_count', 'negation_count', 'positive_smiley_count', 'negative_smiley_count'])
feature_names.extend(['sentiment'])
# feature_names.extend(['karma', 'gold_status', 'employee', 'verif_email', 'upvotes', 'replies',
#                       'is_submitter', 'edited', 'sarcasm', 'quote_count'])
# feature_names.extend(['MFW']*132)
feature_names.extend(['BOW']*13663)

# feature_names.extend(['text']*4)
# feature_names.extend(['sentiment'])
# feature_names.extend(['reddit']*6)
# feature_names.extend(['MFW']*129)
# feature_names.extend(['BOW']*381)
# feature_names.extend(['POS']*16)
# feature_names.extend(['POS']*17)
feature_names.extend(['sim_to_src', 'sim_to_prev', 'sim_to_branch'])
feature_names.extend(['WEMB']*300)
X = pd.DataFrame(X, columns=feature_names)
# X_train = X_train.drop(['url_count', 'tripDotCount', 'period', 'e_mark', 'q_mark',
#                         'hasTripDot', 'e_mark_count', 'cap_sequence_max_len'], axis=1)
clf.fit(X, y)
feature_importances = zip(X.columns, clf.feature_importances_)
feature_importances_sorted = sorted(feature_importances, key=lambda kv: kv[1], reverse=True)
has_value = {}
has_not = {}
for feature, imp in feature_importances_sorted:
    if imp > 0.0:
        if feature in has_value:
            has_value[feature] += 1
        else:
            has_value[feature] = 1
    else:
        if feature in has_not:
            has_not[feature] += 1
        else:
            has_not[feature] = 1

print(has_not)
print(has_value)
print(feature_importances_sorted[:50])

# X_test = pd.DataFrame(X_test, columns=feature_names)
# X_test = X_test.drop(['url_count', 'tripDotCount', 'period', 'e_mark', 'q_mark',
#                       'hasTripDot', 'e_mark_count', 'cap_sequence_max_len'], axis=1)
# y_pred, y_true = clf.predict(X_test), y_test


target_names = ['S', 'D', 'Q', 'C']
# print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=target_names))

# tree.plot_tree(clf)
# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 class_names=target_names,
#                                 feature_names=X.columns,
#                                 filled=True, rounded=True,
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render(filename='tree', format='pdf')

