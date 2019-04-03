import csv
from sklearn.model_selection import train_test_split

datafile = '../data/preprocessed/preprocessed.csv'
tab = '\t'

def get_instances(filename=datafile, delimiter=tab):
    """Load preprocessed data from a csv-file into an iterable
    of instances in the format: [(id, SDQC, feature_vec),...].
    Return the instances as well as a count of the features: (instances, count)"""
    max_emb = 0
    instances = []
    with open(filename, encoding='utf-8', newline='') as csvfile:
        has_header = csv.Sniffer().has_header(csvfile.read(1024))
        csvfile.seek(0) # Rewind
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        if has_header:
            next(csvreader) # Skip header row
        for row in csvreader:
            c_id = row[0]
            sdqc = int(row[2]) # Submission SDQC
            values = row[3].strip("[").strip("]").split(',')
            instance_vec = [float(i.strip()) for i in values]
            max_emb = max(max_emb, len(instance_vec))
            instances.append((c_id, sdqc, instance_vec))
    return instances, max_emb


def get_features_and_labels(filename=datafile, delimiter=tab):
    instances, emb_size = get_instances(filename, delimiter)
    X = [x[2] for x in instances]
    y = [x[1] for x in instances]
    return X, y, emb_size


def get_train_test_split(filename=datafile, delimiter=tab, test_size=0.25):
    X, y, emb_size = get_features_and_labels(filename, delimiter)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y
    )
    return X_train, X_test, y_train, y_test, emb_size