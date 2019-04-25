import csv
from sklearn.model_selection import train_test_split

datafile = '../data/preprocessed/preprocessed.csv'
hmm_datafile = '../data/hmm/preprocessed_hmm.csv'
tab = '\t'

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

def get_features():
    return {'text': False, 'lexicon': False,
                    'sentiment': False, 'reddit': False, 'most_freq': False, 'bow': False, 'pos': False, 'wembs': False}

def select_features(data, feature_mapping, text=False, lexicon=False,
                    sentiment=False, reddit=False, most_freq=False, bow=False, pos=False, wembs=False):
    filtered_data = []
    for instance in data:
        selected_features = []
        if text:
            selected_features.extend(instance[feature_mapping['text']])
        if sentiment:
            selected_features.extend(instance[feature_mapping['sentiment']])
        if lexicon:
            selected_features.extend(instance[feature_mapping['lexicon']])
        if reddit:
            selected_features.extend(instance[feature_mapping['reddit']])
        if most_freq:
            selected_features.extend(instance[feature_mapping['most_frequent']])
        if bow:
            selected_features.extend(instance[feature_mapping['bow']])
        if pos:
            selected_features.extend(instance[feature_mapping['pos']])
        if wembs:
            if feature_mapping['word2vec']:
                selected_features.extend(instance[feature_mapping['word2vec']])
            else:
                selected_features.extend(instance[feature_mapping['fasttext']])
        filtered_data.append(selected_features)
    return filtered_data


def get_features_and_labels(filename=datafile, delimiter=tab):
    instances, n_features, feature_mapping = get_instances(filename, delimiter)
    X = [x[2] for x in instances]
    y = [x[1] for x in instances]
    return X, y, n_features, feature_mapping


def get_train_test_split(filename=datafile, delimiter=tab, test_size=0.25):
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

def get_hmm_data(filename=hmm_datafile, delimiter=tab):
    data = []
    max_branch_len = 0
    with open(filename) as file:
        has_header = csv.Sniffer().has_header(file.read(1024))
        file.seek(0) # Rewind
        csvreader = csv.reader(file, delimiter=delimiter)
        if has_header:
            next(csvreader) # Skip header row
        for row in csvreader:
            truth_status = int(row[0])
            values = row[1].strip("[").strip("]").split(',')
            instance_vec = [float(i.strip()) for i in values]
            data.append((truth_status, instance_vec))
            max_branch_len = max(max_branch_len, len(instance_vec))
    
    return data, max_branch_len