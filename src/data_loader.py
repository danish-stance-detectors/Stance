import csv
import os
import json
import re

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from nltk import word_tokenize

datafile = '../data/preprocessed/preprocessed.csv'
tab = '\t'
punctuation = re.compile('[^a-zA-ZæøåÆØÅ0-9]')

def get_instances(filename=datafile, delimiter=tab):
    """Load preprocessed data from a csv-file into an iterable
    of instances in the format: [(id, SDQC, feature_vec),...].
    Return the instances as well as a count of the features: (instances, count)"""
    instances = []
    with open(filename, encoding='utf-8', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        header = next(csvreader)
        for row in csvreader:
            c_id = row[0]
            sdqc = int(row[2])  # Submission SDQC
            instance_vec = []
            for feature in row[3:]:
                values = feature.strip("[").strip("]").split(',')
                feature_vec = [float(i.strip()) for i in values]
                instance_vec.append(feature_vec)
            instances.append((c_id, sdqc, instance_vec))
    # Map each feature to its corresponding index in the instance vector
    feature_mapping = {}
    for i, feature_name in enumerate(header[3:]):  # Skip ID and SDQC values
        feature_mapping[feature_name] = i
    # Count the length of the total number of features
    n_features = 0
    for feature_vec in instances[0][2]:
        n_features += len(feature_vec)
    return instances, n_features, feature_mapping


def get_features(all_true=True, text=True, lexicon=True, sentiment=True, reddit=True, most_freq=True,
                 bow=True, pos=True, wembs=True):
    if all_true:
        return {'all': all_true, 'text': text, 'lexicon': lexicon,
                'sentiment': sentiment, 'reddit': reddit, 'most_freq': most_freq,
                'bow': bow, 'pos': pos, 'wembs': wembs}
    return {'all': False, 'text': False, 'lexicon': False,
            'sentiment': False, 'reddit': False, 'most_freq': False,
            'bow': False, 'pos': False, 'wembs': False}


def select_features(data, feature_mapping, config_map, merge=True):
    filtered_data = []
    for instance in data:
        selected_features = []
        if config_map['text']:
            selected_features.append(instance[feature_mapping['text']])
        if config_map['lexicon']:
            selected_features.append(instance[feature_mapping['lexicon']])
        if config_map['sentiment']:
            selected_features.append(instance[feature_mapping['sentiment']])
        if config_map['reddit']:
            selected_features.append(instance[feature_mapping['reddit']])
        if config_map['most_freq']:
            selected_features.append(instance[feature_mapping['most_frequent']])
        if config_map['bow']:
            selected_features.append(instance[feature_mapping['bow']])
        if config_map['pos']:
            selected_features.append(instance[feature_mapping['pos']])
        if config_map['wembs']:
            if 'word2vec' in feature_mapping:
                selected_features.append(instance[feature_mapping['word2vec']])
            else:
                selected_features.append(instance[feature_mapping['fasttext']])
        if merge:
            selected_features = [item for sublist in selected_features for item in sublist]
        filtered_data.append(selected_features)
    if merge:
        return filtered_data
    return filtered_data, [len(v) for v in filtered_data[0]]


def get_features_and_labels(filename=datafile, delimiter=tab, with_ids=False):
    instances, n_features, feature_mapping = get_instances(filename, delimiter)
    X = [x[2] for x in instances]
    y = [x[1] for x in instances]
    ids = [x[0] for x in instances]
    if with_ids:
        return X, y, n_features, feature_mapping, ids
    return X, y, n_features, feature_mapping


def get_train_test_split(filename=datafile, delimiter=tab, test_size=0.2):
    X, y, n_features, feature_mapping = get_features_and_labels(filename, delimiter)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y
    )
    return X_train, X_test, y_train, y_test, n_features, feature_mapping


def load_train_test_data(train_file, test_file, delimiter=tab, split=True):
    X_train, y_train, n_features, feature_mapping = get_features_and_labels(train_file, delimiter=delimiter)
    X_test, y_test, _, _ = get_features_and_labels(test_file, delimiter=delimiter)
    if split:
        return X_train, X_test, y_train, y_test, n_features, feature_mapping
    else:
        #  Concatenate train and test samples
        X_train.extend(X_test)
        y_train.extend(y_test)
        return X_train, y_train, n_features, feature_mapping

def load_bow_list(path):
    print("Finding all unique word in {}\n".format(path))
    dirs = os.listdir(path)
    word_set = set()
    for dir in dirs:
        dir_path = path + dir
        print("adding {} to bow list".format(dir_path))
        if os.path.isdir(dir_path):
            for submission in os.listdir(dir_path):
                file_path = dir_path + '/' + submission

                with open(file_path, 'r', encoding='utf-8') as json_file:
                    json_obj = json.load(json_file)

                    for branch in json_obj['branches']:
                        for comment in branch:
                            text = comment['comment']['text']
                            text_tokens = word_tokenize(text.lower(), language='danish')
                            tokens = [t for t in text_tokens if not punctuation.match(t)]
                            word_set.update(tokens)
    print("Found {} unique words in {}".format(len(word_set), path))
    return word_set

def union_reduce_then_split(x1, x2, th=0.001):
    split = len(x1)
    x_union = []
    x_union.extend(x1)
    x_union.extend(x2)
    x_transformed = VarianceThreshold(threshold=th).fit_transform(x_union)
    return x_transformed[:split], x_transformed[split:]
