import csv

def get_instances(filename, delimiter):
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
            sdqc = int(row[2])
            values = row[3].strip("[").strip("]").split(',')
            instance_vec = [float(i.strip()) for i in values]
            max_emb = max(max_emb, len(instance_vec))
            instances.append((c_id, sdqc, instance_vec))
    return (instances, max_emb)